"""Pipeline runner using LangGraph.

Provides the same public API as the original runner but delegates
to the compiled LangGraph graphs.
"""

import logging

from .graph import index_graph, query_graph, test_graph
from .nodes import (
    get_shared_retriever,
    get_shared_generator,
    reset_shared_components,
)
from .retriever import Retriever
from .generator import AnswerGenerator
from .config import OLLAMA_MODEL

logger = logging.getLogger(__name__)


def run_phase_1():
    """Run chunking only (partial index graph)."""
    from .nodes import chunking_node
    return chunking_node({})


def run_phase_2():
    """Run embedding & indexing only (partial index graph)."""
    from .nodes import embedding_node
    return embedding_node({})


def build_pipeline():
    """Run the full index-building graph (Phase 1 + 2) and return
    a ready-to-use ``(retriever, generator)`` pair.

    This is the drop-in replacement for the original ``build_pipeline()``.
    """
    logger.info("Running index-building graph ...")
    result = index_graph.invoke({})
    logger.info("Index graph finished – status: %s", result.get("status"))

    if result.get("error"):
        logger.error("Index build error: %s", result["error"])

    logger.info("=" * 60)
    logger.info("Phase 3 & 4: Retriever + Generator Ready")
    logger.info("=" * 60)

    # Initialise shared singletons with the fresh client/embedder
    reset_shared_components()
    kwargs = {}
    if result.get("qdrant_client"):
        kwargs["client"] = result["qdrant_client"]
    if result.get("embedder"):
        kwargs["embedder"] = result["embedder"]

    retriever = get_shared_retriever(**kwargs)
    generator = get_shared_generator()

    logger.info("Using Ollama model: %s", OLLAMA_MODEL)

    return retriever, generator


def run_query(question: str, chat_history: list[dict] | None = None) -> dict:
    """Run a single question through the RAG query graph.

    Returns the final ``QueryState`` dict with keys:
        answer, sources, chat_history, etc.
    """
    initial_state: dict = {"question": question}
    if chat_history:
        initial_state["chat_history"] = chat_history
    return query_graph.invoke(initial_state)


def run_test(retriever: Retriever | None = None, generator: AnswerGenerator | None = None):
    """Run the test evaluation graph.

    Parameters are kept for backwards-compatibility but ignored;
    the graph uses the shared singletons from ``nodes.py``.
    """
    logger.info("Running test evaluation graph ...")
    result = test_graph.invoke({})
    return result.get("results", [])


if __name__ == "__main__":
    build_pipeline()
