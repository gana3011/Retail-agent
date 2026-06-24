import json
import logging
from typing import Optional

from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from .ssl_setup import configure_ssl

configure_ssl()

from .config import (
    QDRANT_PATH, QDRANT_COLLECTION, EMBEDDING_MODEL,
    GROQ_API_KEY, GROQ_MODEL, TOP_K, RERANK_TOP_K, RERANKER_MODEL,
)

logger = logging.getLogger(__name__)

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


class Retriever:
    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        embedder: Optional[SentenceTransformer] = None,
        groq_client: Optional[Groq] = None,
        model: Optional[str] = None,
        reranker: Optional[object] = None,
    ):
        self.client = client or QdrantClient(path=str(QDRANT_PATH))
        self.embedder = embedder or SentenceTransformer(
            EMBEDDING_MODEL, trust_remote_code=True
        )
        self.llm_client = groq_client or (Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None)
        self.llm_model = model or GROQ_MODEL
        self.reranker = reranker

    def _load_reranker(self):
        if self.reranker is not None:
            return self.reranker
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANKER_MODEL)
            logger.info("Reranker model '%s' loaded", RERANKER_MODEL)
        except Exception as e:
            logger.warning("Failed to load reranker '%s': %s. Skipping reranking.", RERANKER_MODEL, e)
            self.reranker = False
        return self.reranker

    def rewrite_with_context(
        self, question: str, chat_history: list[dict]
    ) -> str:
        """Rewrite a follow-up question into a standalone query using chat history.

        If no chat history is present or the LLM client is unavailable,
        returns the question unchanged.
        """
        if not chat_history or not self.llm_client:
            return question

        # Build a compact history string (keep last 6 turns = 3 exchanges max)
        recent = chat_history[-6:]
        history_lines = []
        for msg in recent:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            # Truncate long assistant answers to keep the prompt short
            if role == "Assistant" and len(content) > 300:
                content = content[:300] + "..."
            history_lines.append(f"{role}: {content}")
        history_str = "\n".join(history_lines)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{
                    "role": "user",
                    "content": CONTEXTUAL_REWRITE_PROMPT.format(
                        history=history_str, question=question
                    ),
                }],
                temperature=0.0,
                max_tokens=150,
            )
            rewritten = response.choices[0].message.content.strip()
            if rewritten:
                logger.info(
                    "[ContextRewrite] '%s' → '%s'", question[:60], rewritten[:60]
                )
                return rewritten
        except Exception as e:
            logger.warning("[ContextRewrite] Failed for '%s': %s", question, e)

        return question

    def expand_queries(
        self, question: str, chat_history: list[dict] | None = None
    ) -> list[str]:
        """Expand a question into multiple retrieval queries.

        When *chat_history* is provided the question is first rewritten
        into a standalone form so that pronoun references are resolved.
        """
        # Step 1 — contextual rewrite (no-op when history is empty)
        standalone = self.rewrite_with_context(question, chat_history or [])

        if not self.llm_client:
            return [standalone]
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": QUERY_EXPANSION_PROMPT.format(question=standalone)}],
                temperature=0.3,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            expansions = json.loads(raw)
            if isinstance(expansions, list) and len(expansions) > 0:
                all_queries = [standalone] + [q for q in expansions if q.strip()]
                logger.info("[MultiQuery] %d queries for: %s", len(all_queries), standalone)
                return all_queries[:4]
        except Exception as e:
            logger.warning("[MultiQuery] Expansion failed for '%s': %s", standalone, e)
        return [standalone]

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
        seen_texts = set()
        all_results = []
        for q in queries:
            vec = self.embedder.encode(q).tolist()
            results = self._search(vec, k)
            for r in results:
                text = r["text"][:200]
                if text not in seen_texts:
                    seen_texts.add(text)
                    all_results.append(r)
        logger.info("[Search] %d queries -> %d unique candidates", len(queries), len(all_results))
        return all_results

    def _rerank(self, question: str, chunks: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
        reranker = self._load_reranker()
        if not reranker:
            return chunks[:top_k]

        pairs = [(question, c["text"]) for c in chunks]
        scores = reranker.predict(pairs)
        for c, s in zip(chunks, scores):
            c["rerank_score"] = float(s)

        chunks.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        logger.info("[Rerank] top-1 score=%.4f, bottom-1 score=%.4f",
                    chunks[0]["rerank_score"], chunks[-1]["rerank_score"])
        return chunks[:top_k]

    def retrieve(self, question: str, k: int = TOP_K) -> list[dict]:
        queries = self.expand_queries(question)
        candidates = self._search_all(queries, k * 2)
        return self._rerank(question, candidates, k)

    def retrieve_with_filters(
        self, question: str, override_filters: dict, k: int = TOP_K
    ) -> list[dict]:
        queries = self.expand_queries(question)
        query_vectors = [self.embedder.encode(q).tolist() for q in queries]

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
