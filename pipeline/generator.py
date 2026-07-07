"""Answer generator using a local Ollama model."""

import logging
from typing import Generator, Optional

import requests

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        # legacy param kept for API compatibility — ignored
        api_key: Optional[str] = None,
    ):
        self.model = model or OLLAMA_MODEL
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _format_chat_history(chat_history: list[dict], max_turns: int = 6) -> str:
        """Format recent chat history into a string for the prompt."""
        if not chat_history:
            return ""
        recent = chat_history[-max_turns:]
        lines = []
        for msg in recent:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            if role == "Assistant" and len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        question: str,
        chunks: list[dict],
        chat_history: list[dict] | None = None,
    ) -> str:
        context_parts = []
        for i, c in enumerate(chunks, 1):
            meta = c.get("metadata", {})
            source = meta.get("source_doc", "Unknown")
            doc_type = meta.get("doc_type", "")
            domain = meta.get("domain", "")
            title = meta.get("title", "")
            term = meta.get("term", "")
            header = f"[Source: {source}"
            if domain:
                header += f" | Domain: {domain}"
            if doc_type:
                header += f" | Type: {doc_type}"
            if title:
                header += f" | Scenario: {title}"
            if term:
                header += f" | Term: {term}"
            header += "]"
            context_parts.append(f"Chunk {i}:\n{header}\n{c['text']}\n")

        context = "\n---\n".join(context_parts)

        history_section = ""
        if chat_history:
            formatted = self._format_chat_history(chat_history)
            if formatted:
                history_section = f"""\n\nPrevious conversation (use this to understand follow-up questions and maintain context):
{formatted}\n"""

        return f"""You are a helpful retail domain assistant. Answer the user's question thoroughly using the provided context chunks.

GUIDELINES:
- Synthesize information across ALL chunks to give a complete, end-to-end answer.
- If chunks describe sequential steps (e.g., a process flow), present them in order.
- If multiple chunks cover different aspects of the same topic, combine them into one cohesive answer.
- If the context collectively contains the answer across multiple chunks, DO NOT say information is missing — combine the pieces.
- Only say "I don't have enough information to answer that" if no chunk addresses the question at all.
- Be detailed and specific. Include definitions, steps, actors, systems, and examples where available.
- If the user asks a follow-up question, use the conversation history below to understand what they are referring to.
- Keep your answers consistent with what you said previously in the conversation.
{history_section}
Context chunks:
{context}

User Question: {question}

Answer:"""

    # ── Generation ─────────────────────────────────────────────

    def generate(
        self,
        question: str,
        chunks: list[dict],
        chat_history: list[dict] | None = None,
    ) -> str:
        prompt = self._build_prompt(question, chunks, chat_history)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            return f"Error generating answer: {e}"

    def generate_stream(
        self,
        question: str,
        chunks: list[dict],
        chat_history: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        prompt = self._build_prompt(question, chunks, chat_history)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {"temperature": 0.2},
        }
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=180,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    import json as _json
                    data = _json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
        except Exception as e:
            logger.error("Ollama streaming failed: %s", e)
            yield f"Error: {e}"

    def generate_with_sources(
        self, question: str, chunks: list[dict]
    ) -> tuple[str, list[dict]]:
        answer = self.generate(question, chunks)
        sources = self._extract_sources(chunks)
        return answer, sources

    def generate_with_sources_stream(
        self, question: str, chunks: list[dict]
    ) -> Generator[tuple[str, list[dict]], None, None]:
        sources = self._extract_sources(chunks)
        yield from self.generate_stream(question, chunks)

    @staticmethod
    def _extract_sources(chunks: list[dict]) -> list[dict]:
        sources = []
        seen = set()
        for c in chunks:
            meta = c.get("metadata", {})
            src = meta.get("source_doc", "Unknown")
            key = src + meta.get("title", "") + meta.get("term", "")
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source_doc": src,
                    "doc_type": meta.get("doc_type", ""),
                    "domain": meta.get("domain", ""),
                    "title": meta.get("title", ""),
                    "term": meta.get("term", ""),
                    "element_type": meta.get("element_type", ""),
                    "relevance_score": round(c.get("score") or 0, 4),
                })
        return sources
