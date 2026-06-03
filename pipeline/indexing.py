import json
import logging
import time
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

from .ssl_setup import configure_ssl

configure_ssl()

from .config import (
    PHASE_1_DIR, QDRANT_PATH, QDRANT_COLLECTION,
    EMBEDDING_MODEL, EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)


def get_embedder(model_name: Optional[str] = None):
    name = model_name or EMBEDDING_MODEL
    logger.info("Loading embedding model: %s ...", name)
    t0 = time.time()
    model = SentenceTransformer(name, trust_remote_code=True)
    logger.info("Model loaded in %.1fs", time.time() - t0)
    return model


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
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))
    return client


def recreate_collection(client: QdrantClient):
    try:
        client.delete_collection(QDRANT_COLLECTION)
    except Exception as e:
        logger.warning("Could not delete existing collection: %s", e)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )

    for field in ("doc_type", "element_type", "domain", "source_doc"):
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    logger.info("Collection '%s' created with indexes", QDRANT_COLLECTION)


def embed_and_index(
    chunks: list[dict],
    model: SentenceTransformer,
    client: QdrantClient,
    batch_size: int = 64,
):
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    logger.info("Generating embeddings for %d chunks...", len(texts))
    t0 = time.time()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        embeddings = model.encode(batch_texts, show_progress_bar=False)

        points = []
        for j, (emb, meta) in enumerate(zip(embeddings, batch_metas)):
            points.append(
                models.PointStruct(
                    id=i + j,
                    vector=emb.tolist(),
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
