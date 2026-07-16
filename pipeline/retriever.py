"""Retriever — fully local via Ollama (no HuggingFace / no internet required).

Embedding  : Ollama nomic-embed-text  (via /api/embed)
LLM calls  : Ollama llama3.2 / qwen2.5  (via /api/chat)
Reranking  : LLM-based relevance ranking  (single Ollama call, no CrossEncoder)
"""
import json
import logging
from typing import Optional

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .ssl_setup import configure_ssl

configure_ssl()

from .config import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
    EMBEDDING_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL,
    TOP_K, RERANK_TOP_K,
    ENABLE_QUERY_EXPANSION, ENABLE_LLM_RERANK,
)

logger = logging.getLogger(__name__)


def _extract_json(text: str):
    """Extract the first valid JSON value from a string that may contain extra text.

    Uses json.JSONDecoder.raw_decode() which stops at the end of the first
    valid JSON token and ignores any trailing content (explanation sentences,
    newlines, etc.) that small LLMs often append.
    """
    decoder = json.JSONDecoder()
    text = text.strip()
    # Scan forward until we find a '[' or '{' that starts a valid JSON value
    for i, ch in enumerate(text):
        if ch in ("[", "{"):
            try:
                value, _ = decoder.raw_decode(text, i)
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON found in: {text[:120]}")

# ──────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────

QUERY_EXPANSION_PROMPT = """You are a retail knowledge base assistant. Given a user question, generate 2 alternative phrasings that would help retrieve relevant information from different document types (process flows, FAQs, glossary terms, training content, scenarios).

Return a JSON array of 2-3 query strings. Each should capture a different aspect or angle of the question.
Keep each query concise (under 20 words).

Question: {question}
JSON:"""

CONTEXTUAL_REWRITE_PROMPT = """You are a retail knowledge base assistant. The user is asking a follow-up question in a conversation. Rewrite their latest question as a fully self-contained, standalone question that can be understood without the conversation history.

Rules:
- Resolve all pronouns (it, they, that, this, those, etc.) to the specific nouns they refer to.
- Preserve the user's intent exactly.
- If the question is already self-contained, return it unchanged.
- Return ONLY the rewritten question, nothing else.

Conversation history:
{history}

Latest question: {question}

Rewritten standalone question:"""

RERANK_PROMPT = """You are a relevance judge for a retail knowledge base.
Given a question and a list of text chunks, return the indices of the {top_k} most relevant chunks, ordered from MOST to LEAST relevant.

Question: {question}

Chunks:
{chunks_text}

Rules:
- Return ONLY a valid JSON array of integer indices (0-based), e.g. [2, 0, 4, 1, 3]
- Include exactly {top_k} indices.
- Do NOT include any explanation or extra text.

JSON array of top {top_k} indices:"""


class Retriever:
    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        embedder=None,
        ollama_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        # legacy params kept for API compatibility
        groq_client=None,
        model=None,
        reranker=None,
    ):
        if client:
            self.client = client
        else:
            if QDRANT_API_KEY:
                self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            else:
                self.client = QdrantClient(url=QDRANT_URL)

        # Accept any embedder (OllamaEmbedder is the default)
        if embedder is not None:
            self.embedder = embedder
        else:
            from .indexing import OllamaEmbedder
            self.embedder = OllamaEmbedder()

        self.ollama_model = ollama_model or OLLAMA_MODEL
        self.ollama_base_url = (ollama_base_url or OLLAMA_BASE_URL).rstrip("/")

    # ──────────────────────────────────────────────────────────────
    # Ollama chat helper
    # ──────────────────────────────────────────────────────────────

    def _chat(self, prompt: str, temperature: float = 0.3) -> str:
        """Single non-streaming call to Ollama /api/chat."""
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            resp = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception as e:
            logger.warning("Ollama /api/chat failed: %s", e)
            return ""

    # ──────────────────────────────────────────────────────────────
    # Contextual rewrite (follow-up questions)
    # ──────────────────────────────────────────────────────────────

    def rewrite_with_context(self, question: str, chat_history: list[dict]) -> str:
        """Rewrite a follow-up question into a standalone query using chat history."""
        if not chat_history:
            return question

        recent = chat_history[-6:]
        history_lines = []
        for msg in recent:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            if role == "Assistant" and len(content) > 300:
                content = content[:300] + "..."
            history_lines.append(f"{role}: {content}")
        history_str = "\n".join(history_lines)

        prompt = CONTEXTUAL_REWRITE_PROMPT.format(history=history_str, question=question)
        rewritten = self._chat(prompt, temperature=0.0)
        if rewritten:
            logger.info("[ContextRewrite] '%s' → '%s'", question[:60], rewritten[:60])
            return rewritten
        return question

    # ──────────────────────────────────────────────────────────────
    # Multi-query expansion
    # ──────────────────────────────────────────────────────────────

    def expand_queries(
        self, question: str, chat_history: list[dict] | None = None
    ) -> list[str]:
        """Expand a question into 2-3 alternate retrieval queries.

        Skipped when ENABLE_QUERY_EXPANSION=False (default) for speed —
        the single nomic-embed-text query already gives good recall.
        """
        standalone = self.rewrite_with_context(question, chat_history or [])

        if not ENABLE_QUERY_EXPANSION:
            logger.info("[MultiQuery] Expansion disabled — using single query")
            return [standalone]

        prompt = QUERY_EXPANSION_PROMPT.format(question=standalone)
        raw = self._chat(prompt, temperature=0.3)

        try:
            raw_clean = raw.replace("```json", "").replace("```", "").strip()
            expansions = _extract_json(raw_clean)
            if isinstance(expansions, list) and len(expansions) > 0:
                all_queries = [standalone] + [q for q in expansions if isinstance(q, str) and q.strip()]
                logger.info("[MultiQuery] %d queries for: %s", len(all_queries), standalone)
                return all_queries[:4]
        except Exception as e:
            logger.warning("[MultiQuery] Expansion failed for '%s': %s", standalone, e)
        return [standalone]

    # ──────────────────────────────────────────────────────────────
    # Vector search
    # ──────────────────────────────────────────────────────────────

    def _search(self, query_vector: list, k: int) -> list[dict]:
        result = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=k,
            with_payload=True,
        )
        results = []
        for point in result.points:
            results.append({
                "text": point.payload.get("text", ""),
                "score": point.score or 0.0,
                "metadata": {
                    k: v for k, v in point.payload.items() if k != "text"
                },
            })
        return results

    def _search_all(self, queries: list[str], k: int) -> list[dict]:
        """Search Qdrant with all expanded queries, deduplicating results."""
        seen_texts = set()
        all_results = []
        for q in queries:
            vec = self.embedder.encode([q])[0].tolist()
            results = self._search(vec, k)
            for r in results:
                text = r["text"][:200]
                if text not in seen_texts:
                    seen_texts.add(text)
                    all_results.append(r)
        logger.info("[Search] %d queries -> %d unique candidates", len(queries), len(all_results))
        return all_results

    # ──────────────────────────────────────────────────────────────
    # LLM-based reranking (fully local — no HuggingFace CrossEncoder)
    # ──────────────────────────────────────────────────────────────

    def _rerank(self, question: str, chunks: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
        """Rerank chunks using Ollama LLM as relevance judge.

        When ENABLE_LLM_RERANK=False (default) — uses vector similarity scores
        directly (fast, ~0s). Enable for higher relevance precision at ~40s cost.
        """
        if not chunks:
            return []

        # Fast path: sort by Qdrant cosine similarity score — no LLM call needed
        if not ENABLE_LLM_RERANK:
            logger.info("[Rerank] Using vector score ordering (LLM rerank disabled)")
            return sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)[:top_k]

        if len(chunks) <= top_k:
            return chunks

        # Build numbered chunk list (truncate each to 300 chars for prompt efficiency)
        chunks_text = "\n\n".join(
            f"[{i}]: {c['text'][:300].strip()}..."
            for i, c in enumerate(chunks)
        )

        prompt = RERANK_PROMPT.format(
            question=question,
            chunks_text=chunks_text,
            top_k=top_k,
        )

        raw = self._chat(prompt, temperature=0.0)

        try:
            indices = _extract_json(raw)
            if isinstance(indices, list):
                # Validate: must be valid integer indices within range
                valid = [
                    int(i) for i in indices
                    if isinstance(i, (int, float)) and 0 <= int(i) < len(chunks)
                ]
                if valid:
                    seen: set = set()
                    ordered = [chunks[i] for i in valid if not (i in seen or seen.add(i))]
                    logger.info("[LLM-Rerank] Selected indices %s from %d candidates", valid[:top_k], len(chunks))
                    return ordered[:top_k]
        except Exception as e:
            logger.warning("[LLM-Rerank] Parsing failed: %s | raw: %s", e, raw[:200])

        # Fallback: sort by vector score and take top_k
        logger.info("[LLM-Rerank] Fallback to vector score ordering")
        return sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)[:top_k]

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def retrieve(self, question: str, k: int = TOP_K) -> list[dict]:
        queries = self.expand_queries(question)
        candidates = self._search_all(queries, k * 2)
        return self._rerank(question, candidates, k)

    def retrieve_with_filters(
        self, question: str, override_filters: dict, k: int = TOP_K
    ) -> list[dict]:
        queries = self.expand_queries(question)
        query_vectors = [self.embedder.encode([q])[0].tolist() for q in queries]

        filter_conditions = []
        for key, value in override_filters.items():
            if value:
                filter_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        query_filter = None
        if filter_conditions:
            query_filter = models.Filter(must=filter_conditions)

        seen_texts = set()
        all_results = []
        for vec in query_vectors:
            result = self.client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vec,
                query_filter=query_filter,
                limit=k * 2,
                with_payload=True,
            )
            for point in result.points:
                text = (point.payload.get("text", "") or "")[:200]
                if text not in seen_texts:
                    seen_texts.add(text)
                    all_results.append({
                        "text": point.payload.get("text", ""),
                        "score": point.score or 0.0,
                        "metadata": {
                            k: v for k, v in point.payload.items() if k != "text"
                        },
                    })

        return self._rerank(question, all_results, k)
