"""Typed state definitions for the LangGraph pipeline.

Three state schemas are defined — one per graph:
- IndexState  : index-building graph  (Phase 1 + 2)
- QueryState  : RAG query graph       (expand → retrieve → rerank → generate)
- TestState   : test evaluation graph
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


class IndexState(TypedDict, total=False):
    """State for the index-building graph (Phase 1 + 2)."""

    # Phase 1 output
    chunks: list[dict]

    # Phase 2 output
    embedder: Any
    qdrant_client: Any
    vector_count: int

    # Status tracking
    status: str
    error: Optional[str]


class QueryState(TypedDict, total=False):
    """State for the RAG query graph.

    ``chat_history`` uses an ``operator.add`` reducer so that each node
    can *append* new entries without overwriting previous ones.
    """

    # Input
    question: str

    # Node outputs (pipeline stages)
    expanded_queries: list[str]
    rewritten_question: str  # standalone version after context rewrite
    retrieved_chunks: list[dict]
    reranked_chunks: list[dict]

    # Final output
    answer: str
    sources: list[dict]
    error: Optional[str]

    # Chat memory — new entries are appended via operator.add
    chat_history: Annotated[list[dict], operator.add]


class TestState(TypedDict, total=False):
    """State for the test evaluation graph."""

    questions: list[str]
    results: list[dict]
    report_path: str
    avg_latency: float
    error: Optional[str]
