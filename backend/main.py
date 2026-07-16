"""FastAPI application entry point for the Retail Knowledge Bot."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# ── Ensure project root is on sys.path before any pipeline imports ────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suppress noisy transformer logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Configure SSL before anything else touches the network
from pipeline.ssl_setup import configure_ssl

configure_ssl()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pipeline.auth import bootstrap_admin
from pipeline.config import QDRANT_COLLECTION
from pipeline.nodes import get_shared_retriever, get_shared_generator

from backend.deps import app_state
from backend.auth_routes import router as auth_router
from backend.chat_routes import router as chat_router
from backend.index_routes import router as index_router, settings_router
from backend.doc_routes import router as doc_router
from backend.admin_routes import router as admin_router

logger = logging.getLogger(__name__)


# ── Lifespan handler ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: bootstrap admin, try loading existing index."""
    logger.info("Starting Retail Knowledge Bot API …")

    # Ensure default admin account exists
    bootstrap_admin()
    logger.info("Admin bootstrap complete")

    # Try to load an existing Qdrant index
    try:
        from pipeline.indexing import get_qdrant_client

        client = get_qdrant_client()
        if client.collection_exists(QDRANT_COLLECTION):
            info = client.get_collection(QDRANT_COLLECTION)
            if info.points_count and info.points_count > 0:
                app_state.retriever = get_shared_retriever()
                app_state.generator = get_shared_generator()
                app_state.indexed = True
                logger.info(
                    "Loaded existing index: %d vectors in '%s'",
                    info.points_count,
                    QDRANT_COLLECTION,
                )
            else:
                logger.info("Index collection exists but is empty — rebuild required")
        else:
            logger.info("No index collection found on server — rebuild required")
    except Exception as exc:
        logger.warning("No existing index found (this is fine on first run): %s", exc)

    yield  # ← application runs here

    logger.info("Shutting down Retail Knowledge Bot API …")


# ── Application setup ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Retail Knowledge Bot API",
    description="REST API wrapping the Retail Knowledge RAG pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Next.js frontend on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ─────────────────────────────────────────────────────────

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(index_router)
app.include_router(settings_router)
app.include_router(doc_router)
app.include_router(admin_router)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["health"])
async def health_check():
    """Simple health-check endpoint."""
    return {
        "status": "healthy",
        "indexed": app_state.indexed,
        "model": app_state.ollama_model,
    }
