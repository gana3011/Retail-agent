import logging
from typing import Generator, Optional

from groq import Groq

from .config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or GROQ_API_KEY
        self.model = model or GROQ_MODEL
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    @staticmethod
    def _format_chat_history(chat_history: list[dict], max_turns: int = 6) -> str:
        """Format recent chat history into a string for the prompt.

        Keeps the last *max_turns* messages (user + assistant combined)
        and truncates long assistant answers to avoid blowing up context.
        """
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

        # Build optional conversation history section
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

    def generate(
        self,
        question: str,
        chunks: list[dict],
        chat_history: list[dict] | None = None,
    ) -> str:
        if not self.client:
            return "GROQ_API_KEY not set. Please set the environment variable and restart."

        prompt = self._build_prompt(question, chunks, chat_history)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048,
        )

        return response.choices[0].message.content.strip()

    def generate_stream(
        self,
        question: str,
        chunks: list[dict],
        chat_history: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        if not self.client:
            yield "GROQ_API_KEY not set. Please set the environment variable and restart."
            return

        prompt = self._build_prompt(question, chunks, chat_history)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048,
            stream=True,
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content

    def generate_with_sources(self, question: str, chunks: list[dict]) -> tuple[str, list[dict]]:
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
