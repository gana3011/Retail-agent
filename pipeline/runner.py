import json
import logging
import time
from pathlib import Path

from .chunking import chunk_all
from .indexing import load_chunks, get_embedder, get_qdrant_client, recreate_collection, embed_and_index
from .retriever import Retriever
from .generator import AnswerGenerator
from .config import (
    PHASE_1_DIR, OUTPUT_DIR, GROQ_API_KEY, QDRANT_COLLECTION,
)

logger = logging.getLogger(__name__)


def run_phase_1():
    logger.info("=" * 60)
    logger.info("Phase 1: Chunking")
    logger.info("=" * 60)
    return chunk_all()


def run_phase_2():
    logger.info("=" * 60)
    logger.info("Phase 2: Embedding & Indexing")
    logger.info("=" * 60)
    chunks = load_chunks()
    logger.info("Loaded %d chunks", len(chunks))

    model = get_embedder()
    client = get_qdrant_client()
    recreate_collection(client)
    embed_and_index(chunks, model, client)

    count = client.count(QDRANT_COLLECTION)
    logger.info("Collection has %d vectors", count.count)
    return client


def run_test(retriever: Retriever, generator: AnswerGenerator):
    logger.info("=" * 60)
    logger.info("Phase 5: Testing")
    logger.info("=" * 60)

    from .test_set import get_test_set
    questions = get_test_set()

    results = []
    for q in questions:
        t0 = time.time()
        chunks = retriever.retrieve(q)
        answer, sources = generator.generate_with_sources(q, chunks)
        elapsed = time.time() - t0

        results.append({
            "question": q,
            "latency": round(elapsed, 2),
            "num_chunks": len(chunks),
            "avg_score": round(sum(c.get("score", 0) or 0 for c in chunks) / len(chunks), 4) if chunks else 0,
            "sources": [s["source_doc"] for s in sources],
        })
        logger.info("  [%.1fs] %s...", elapsed, q[:60])

    report_path = OUTPUT_DIR / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_questions": len(results),
            "avg_latency": round(sum(r["latency"] for r in results) / len(results), 2),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    logger.info("Test report saved to %s", report_path)
    avg_lat = sum(r["latency"] for r in results) / len(results)
    logger.info("Average latency: %.2fs", avg_lat)
    return results


def build_pipeline():
    run_phase_1()
    run_phase_2()

    logger.info("=" * 60)
    logger.info("Phase 3 & 4: Retriever + Generator Ready")
    logger.info("=" * 60)

    retriever = Retriever()
    generator = AnswerGenerator(api_key=GROQ_API_KEY)

    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set. Set it in .env file.")
        logger.warning("Get a free key at: https://console.groq.com/keys")

    return retriever, generator


if __name__ == "__main__":
    build_pipeline()
