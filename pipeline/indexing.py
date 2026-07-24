"""Embedding helpers — supports Ollama (nomic-embed-text) or Sentence Transformers."""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from fastembed import SparseTextEmbedding

from .ssl_setup import configure_ssl

configure_ssl()

from .config import (
    PHASE_1_DIR, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
    EMBEDDING_MODEL, EMBEDDING_DIM, OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Ollama embedding helper
# ──────────────────────────────────────────────────────────────

class OllamaEmbedder:
    """Wraps the Ollama /api/embed endpoint to produce dense vectors."""

    def __init__(self, model: str = EMBEDDING_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dim: Optional[int] = None

    def encode(self, texts, show_progress_bar: bool = False, batch_size: int = 32):
        """Encode a list of strings → numpy-compatible list of float arrays."""
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            payload = {"model": self.model, "input": batch}
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings, dtype="float32")

    def get_sentence_embedding_dimension(self) -> int:
        if self._dim is None:
            sample = self.encode(["hello"])
            self._dim = sample.shape[1]
        return self._dim

# ──────────────────────────────────────────────────────────────
# Public factory: always returns OllamaEmbedder (fully local)
# ──────────────────────────────────────────────────────────────

def get_embedder(model_name: Optional[str] = None):
    """Return an OllamaEmbedder — fully local, no HuggingFace required."""
    name = model_name or EMBEDDING_MODEL
    logger.info("Using Ollama embedder: %s @ %s", name, OLLAMA_BASE_URL)
    embedder = OllamaEmbedder(model=name)
    # Quick connectivity check
    try:
        embedder.encode(["test"])
        logger.info("Ollama embedder ready (model=%s)", name)
    except Exception as e:
        logger.error(
            "Cannot reach Ollama at %s. "
            "Make sure Ollama is running and 'ollama pull %s' has been executed. Error: %s",
            OLLAMA_BASE_URL, name, e,
        )
        raise
    return embedder


def load_chunks() -> list[dict]:
    path = PHASE_1_DIR / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Run chunking first: {path} does not exist")
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def get_qdrant_client() -> QdrantClient:
    port = 443 if QDRANT_URL.startswith("https") else 6333
    if QDRANT_API_KEY:
        client = QdrantClient(url=QDRANT_URL, port=port, api_key=QDRANT_API_KEY)
    else:
        client = QdrantClient(url=QDRANT_URL, port=port)
    return client


def recreate_collection(client: QdrantClient):
    dim = EMBEDDING_DIM

    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=dim,
            distance=models.Distance.COSINE,
        ),
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )

    for field in ("doc_type", "element_type", "domain", "source_doc"):
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    logger.info("Collection '%s' created with dim=%d and indexes", QDRANT_COLLECTION, dim)


def embed_and_index(
    chunks: list[dict],
    model,
    client: QdrantClient,
    batch_size: int = 32,
):
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    logger.info("Generating embeddings for %d chunks...", len(texts))
    t0 = time.time()

    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        batch_metas = metadatas[i: i + batch_size]
        
        embeddings = model.encode(batch_texts, show_progress_bar=False)
        sparse_embeddings = list(sparse_model.embed(batch_texts))

        points = []
        for j, (emb, sparse_emb, meta) in enumerate(zip(embeddings, sparse_embeddings, batch_metas)):
            points.append(
                models.PointStruct(
                    id=i + j,
                    vector={
                        "": emb.tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_emb.indices.tolist(),
                            values=sparse_emb.values.tolist(),
                        )
                    },
                    payload={
                        "text": batch_texts[j],
                        **meta,
                    },
                )
            )

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
            wait=True,
        )

        if (i // batch_size) % 5 == 0:
            logger.info("  Indexed %d/%d chunks", min(i + batch_size, len(texts)), len(texts))

    elapsed = time.time() - t0
    logger.info("Indexing complete: %d chunks in %.1fs", len(texts), elapsed)


def run_indexing(force: bool = True):
    logger.info("=" * 60)
    logger.info("Phase 2: Embedding & Indexing")
    logger.info("=" * 60)

    chunks = load_chunks()
    logger.info("Loaded %d chunks from phase 1", len(chunks))

    model = get_embedder()
    client = get_qdrant_client()
    if force:
        recreate_collection(client)
    embed_and_index(chunks, model, client)

    count = client.count(QDRANT_COLLECTION)
    logger.info("Collection '%s' has %d vectors", QDRANT_COLLECTION, count.count)
    return client


if __name__ == "__main__":
    run_indexing()
