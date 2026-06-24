"""LangGraph node functions for the Retail Knowledge pipeline.

Each function takes a state dict and returns a partial state update.
Shared Retriever / AnswerGenerator singletons are managed in this module
so that the Streamlit app and CLI can both use them.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .chunking import chunk_all
from .indexing import (
    load_chunks, get_embedder, get_qdrant_client,
    recreate_collection, embed_and_index,
)
from .retriever import Retriever
from .generator import AnswerGenerator
from .config import (
    PHASE_1_DIR, OUTPUT_DIR, GROQ_API_KEY, QDRANT_COLLECTION, TOP_K,
)
from .state import IndexState, QueryState, TestState

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Shared singletons (initialised lazily by nodes)
# ──────────────────────────────────────────────────────────────
_retriever: Retriever | None = None
_generator: AnswerGenerator | None = None


def get_shared_retriever(**kwargs: Any) -> Retriever:
    """Return (and optionally create) the shared Retriever."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever(**kwargs)
    return _retriever


def get_shared_generator(**kwargs: Any) -> AnswerGenerator:
    """Return (and optionally create) the shared AnswerGenerator."""
    global _generator
    if _generator is None:
        _generator = AnswerGenerator(**kwargs)
    return _generator


def reset_shared_components() -> None:
    """Reset singletons (call after re-indexing)."""
    global _retriever, _generator
    _retriever = None
    _generator = None


# ──────────────────────────────────────────────────────────────
# Data-preparation node (Phase 0)
# ──────────────────────────────────────────────────────────────

def data_preparation_node(state: IndexState) -> dict:
    """Phase 0 – convert .docx files into structured JSON.

    Imports and calls the reusable ``run_phase_0()`` function from
    the data-preparation module.
    """
    logger.info("=" * 60)
    logger.info("Phase 0: Data Preparation")
    logger.info("=" * 60)
    try:
        from phase_0_data_preparation import run_phase_0
        result = run_phase_0()
        if result is None:
            return {"error": "No .docx files found in data/ directory", "status": "failed"}
        logger.info(
            "Phase 0 complete: %d files → %d elements",
            result["total_files"], result["total_elements"],
        )
        return {"status": "prepared"}
    except Exception as e:
        logger.error("Data preparation failed: %s", e)
        return {"error": str(e), "status": "failed"}


# ──────────────────────────────────────────────────────────────
# Index-building nodes
# ──────────────────────────────────────────────────────────────

def chunking_node(state: IndexState) -> dict:
    """Phase 1 – chunk all documents."""
    logger.info("=" * 60)
    logger.info("Phase 1: Chunking")
    logger.info("=" * 60)
    try:
        chunks = chunk_all()
        return {"chunks": chunks, "status": "chunked"}
    except Exception as e:
        logger.error("Chunking failed: %s", e)
        return {"error": str(e), "status": "failed"}


def embedding_node(state: IndexState) -> dict:
    """Phase 2 – embed & index into Qdrant."""
    logger.info("=" * 60)
    logger.info("Phase 2: Embedding & Indexing")
    logger.info("=" * 60)
    try:
        chunks = load_chunks()
        logger.info("Loaded %d chunks", len(chunks))

        model = get_embedder()
        client = get_qdrant_client()
        recreate_collection(client)
        embed_and_index(chunks, model, client)

        count = client.count(QDRANT_COLLECTION)
        logger.info("Collection has %d vectors", count.count)
        return {
            "embedder": model,
            "qdrant_client": client,
            "vector_count": count.count,
            "status": "indexed",
        }
    except Exception as e:
        logger.error("Embedding/indexing failed: %s", e)
        return {"error": str(e), "status": "failed"}


# ──────────────────────────────────────────────────────────────
# RAG query nodes
# ──────────────────────────────────────────────────────────────

def query_expansion_node(state: QueryState) -> dict:
    """Expand the user question into multiple retrieval queries.

    When chat_history is present, the retriever first rewrites the
    question into a standalone form (resolving pronouns like "it",
    "that", etc.) before generating alternative phrasings.
    """
    question = state["question"]
    chat_history = state.get("chat_history", [])
    logger.info("[QueryExpansion] Expanding: %s (history: %d msgs)", question[:60], len(chat_history))
    try:
        retriever = get_shared_retriever()
        expanded = retriever.expand_queries(question, chat_history=chat_history)
        # The first query is always the standalone (rewritten) form
        rewritten = expanded[0] if expanded else question
        return {"expanded_queries": expanded, "rewritten_question": rewritten}
    except Exception as e:
        logger.warning("Query expansion failed: %s", e)
        return {"expanded_queries": [question], "rewritten_question": question}


def retrieval_node(state: QueryState) -> dict:
    """Retrieve candidate chunks from Qdrant using expanded queries."""
    queries = state.get("expanded_queries", [state["question"]])
    logger.info("[Retrieval] Searching with %d queries", len(queries))
    try:
        retriever = get_shared_retriever()
        candidates = retriever._search_all(queries, TOP_K * 2)
        return {"retrieved_chunks": candidates}
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return {"retrieved_chunks": [], "error": str(e)}


def reranking_node(state: QueryState) -> dict:
    """Re-rank retrieved chunks for the original question."""
    question = state["question"]
    candidates = state.get("retrieved_chunks", [])
    logger.info("[Reranking] %d candidates for: %s", len(candidates), question[:60])
    if not candidates:
        return {"reranked_chunks": []}
    try:
        retriever = get_shared_retriever()
        reranked = retriever._rerank(question, candidates, TOP_K)
        return {"reranked_chunks": reranked}
    except Exception as e:
        logger.error("Reranking failed: %s", e)
        return {"reranked_chunks": candidates[:TOP_K]}


def generation_node(state: QueryState) -> dict:
    """Generate an answer from the reranked chunks.

    Passes chat_history to the generator so the LLM prompt
    includes prior conversation context for follow-up questions.
    """
    question = state["question"]
    chunks = state.get("reranked_chunks", [])
    chat_history = state.get("chat_history", [])
    logger.info("[Generation] Generating answer for: %s (history: %d msgs)", question[:60], len(chat_history))
    try:
        generator = get_shared_generator(api_key=GROQ_API_KEY)
        answer = generator.generate(question, chunks, chat_history=chat_history)
        sources = generator._extract_sources(chunks)
        return {
            "answer": answer,
            "sources": sources,
            "chat_history": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }
    except Exception as e:
        logger.error("Generation failed: %s", e)
        return {
            "answer": f"Error generating answer: {e}",
            "sources": [],
            "error": str(e),
        }


# ──────────────────────────────────────────────────────────────
# Test evaluation nodes
# ──────────────────────────────────────────────────────────────

def load_test_questions_node(state: TestState) -> dict:
    """Load the test question set."""
    from .test_set import get_test_set
    questions = get_test_set()
    logger.info("Loaded %d test questions", len(questions))
    return {"questions": questions}


def run_test_evaluation_node(state: TestState) -> dict:
    """Run all test questions through the RAG pipeline and produce a report."""
    logger.info("=" * 60)
    logger.info("Phase 5: Testing")
    logger.info("=" * 60)

    questions = state.get("questions", [])
    retriever = get_shared_retriever()
    generator = get_shared_generator(api_key=GROQ_API_KEY)

    results: list[dict] = []
    for q in questions:
        t0 = time.time()
        chunks = retriever.retrieve(q)
        answer, sources = generator.generate_with_sources(q, chunks)
        elapsed = time.time() - t0

        results.append({
            "question": q,
            "latency": round(elapsed, 2),
            "num_chunks": len(chunks),
            "avg_score": (
                round(sum(c.get("score", 0) or 0 for c in chunks) / len(chunks), 4)
                if chunks else 0
            ),
            "sources": [s["source_doc"] for s in sources],
        })
        logger.info("  [%.1fs] %s...", elapsed, q[:60])

    report_path = OUTPUT_DIR / "test_report.json"
    avg_lat = round(sum(r["latency"] for r in results) / len(results), 2) if results else 0.0
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_questions": len(results),
            "avg_latency": avg_lat,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    logger.info("Test report saved to %s", report_path)
    logger.info("Average latency: %.2fs", avg_lat)
    return {
        "results": results,
        "report_path": str(report_path),
        "avg_latency": avg_lat,
    }

