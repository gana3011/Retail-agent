"""LangGraph graph definitions for the Retail Knowledge pipeline.

Four graphs are exposed:
- ``full_index_graph`` : full pipeline  (Phase 0 → Phase 1 → Phase 2)
- ``index_graph``      : builds the vector index  (Phase 1 → Phase 2)
- ``query_graph``      : answers a single user question  (expand → retrieve → rerank → generate)
- ``test_graph``       : runs the evaluation test suite
"""

from langgraph.graph import StateGraph, START, END

from .state import IndexState, QueryState, TestState
from .nodes import (
    data_preparation_node,
    chunking_node,
    embedding_node,
    query_expansion_node,
    retrieval_node,
    reranking_node,
    generation_node,
    load_test_questions_node,
    run_test_evaluation_node,
)

# ──────────────────────────────────────────────────────────────
# 1. Index-building graph
# ──────────────────────────────────────────────────────────────

def build_full_index_graph():
    """Phase 0 (data prep) → Phase 1 (chunking) → Phase 2 (embedding & indexing).

    Use this when new documents have been added to data/ and the
    entire pipeline (including extraction) needs to run.
    """
    builder = StateGraph(IndexState)

    builder.add_node("data_preparation", data_preparation_node)
    builder.add_node("chunking", chunking_node)
    builder.add_node("embedding", embedding_node)

    builder.add_edge(START, "data_preparation")
    builder.add_edge("data_preparation", "chunking")
    builder.add_edge("chunking", "embedding")
    builder.add_edge("embedding", END)

    return builder.compile()


def build_index_graph():
    """Phase 1 (chunking) → Phase 2 (embedding & indexing)."""
    builder = StateGraph(IndexState)

    builder.add_node("chunking", chunking_node)
    builder.add_node("embedding", embedding_node)

    builder.add_edge(START, "chunking")
    builder.add_edge("chunking", "embedding")
    builder.add_edge("embedding", END)

    return builder.compile()


# ──────────────────────────────────────────────────────────────
# 2. RAG query graph
# ──────────────────────────────────────────────────────────────

def build_query_graph():
    """expand → retrieve → rerank → generate."""
    builder = StateGraph(QueryState)

    builder.add_node("expand", query_expansion_node)
    builder.add_node("retrieve", retrieval_node)
    builder.add_node("rerank", reranking_node)
    builder.add_node("generate", generation_node)

    builder.add_edge(START, "expand")
    builder.add_edge("expand", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


# ──────────────────────────────────────────────────────────────
# 3. Test evaluation graph
# ──────────────────────────────────────────────────────────────

def build_test_graph():
    """load_questions → evaluate."""
    builder = StateGraph(TestState)

    builder.add_node("load_questions", load_test_questions_node)
    builder.add_node("evaluate", run_test_evaluation_node)

    builder.add_edge(START, "load_questions")
    builder.add_edge("load_questions", "evaluate")
    builder.add_edge("evaluate", END)

    return builder.compile()


# ──────────────────────────────────────────────────────────────
# Pre-compiled graph instances (import-ready)
# ──────────────────────────────────────────────────────────────

index_graph = build_index_graph()
full_index_graph = build_full_index_graph()
query_graph = build_query_graph()
test_graph = build_test_graph()

