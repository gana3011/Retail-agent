from typing import Optional

from groq import Groq

from .config import GROQ_API_KEY, GROQ_MODEL


class AnswerGenerator:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or GROQ_API_KEY
        self.model = model or GROQ_MODEL
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def _build_prompt(self, question: str, chunks: list[dict]) -> str:
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

        return f"""You are a helpful retail domain assistant. Answer the user's question using ONLY the provided context chunks. If the context does not contain enough information, say "I don't have enough information to answer that."

Context chunks:
{context}

User Question: {question}

Answer:"""

    def generate(self, question: str, chunks: list[dict]) -> str:
        if not self.client:
            return "GROQ_API_KEY not set. Please set the environment variable and restart."

        prompt = self._build_prompt(question, chunks)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )

        return response.choices[0].message.content.strip()

    def generate_with_sources(self, question: str, chunks: list[dict]) -> tuple[str, list[dict]]:
        answer = self.generate(question, chunks)
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
                    "relevance_score": round(c.get("score", 0), 4),
                })
        return answer, sources
