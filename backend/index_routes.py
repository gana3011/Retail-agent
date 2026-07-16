"""Index management routes (admin-only): status, rebuild, model change."""

import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import APIRouter, Depends, HTTPException, status

from pipeline.config import QDRANT_COLLECTION, OLLAMA_MODEL, DATA_DIR
from pipeline.graph import build_index_graph, build_full_index_graph
from pipeline.indexing import get_qdrant_client
from pipeline.nodes import reset_shared_components, get_shared_retriever, get_shared_generator

from backend.deps import require_admin, get_current_user, app_state
from backend.models import IndexStatusResponse, ModelChangeRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/index", tags=["index"])
settings_router = APIRouter(tags=["settings"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_collection_stats() -> tuple[int, int]:
    """Return (document_count, vector_count). Document count is from filesystem, vector count from Qdrant."""
    doc_count = 0
    try:
        if DATA_DIR.exists():
            doc_count = sum(1 for f in DATA_DIR.iterdir() if f.is_file() and f.suffix.lower() == ".docx")
    except Exception:
        pass

    vec_count = 0
    try:
        client = get_qdrant_client()
        info = client.get_collection(QDRANT_COLLECTION)
        vec_count = info.vectors_count or info.points_count or 0
    except Exception:
        pass
        
    return doc_count, vec_count


def _refresh_singletons() -> None:
    """Reset and re-initialise shared retriever / generator after index rebuild."""
    reset_shared_components()
    app_state.retriever = get_shared_retriever()
    app_state.generator = get_shared_generator()
    app_state.indexed = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status", response_model=IndexStatusResponse)
async def index_status(user: dict = Depends(get_current_user)):
    """Return current index status including doc/vector counts (any authenticated user)."""
    doc_count, vec_count = _get_collection_stats()
    return IndexStatusResponse(
        indexed=app_state.indexed,
        document_count=doc_count,
        vector_count=vec_count,
        current_model=app_state.ollama_model,
    )


@router.post("/rebuild")
async def rebuild_index(admin: dict = Depends(require_admin)):
    """Rebuild the index (Phase 1 chunking + Phase 2 embedding)."""
    try:
        logger.info("Index rebuild triggered by %s", admin["username"])
        graph = build_index_graph()
        result = graph.invoke({})
        if result and result.get("error"):
            raise Exception(f"Pipeline error: {result['error']}")
        _refresh_singletons()
        doc_count, vec_count = _get_collection_stats()
        return {
            "message": "Index rebuilt successfully",
            "document_count": doc_count,
            "vector_count": vec_count,
        }
    except Exception as exc:
        logger.exception("Index rebuild failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index rebuild failed: {exc}",
        )


@router.post("/full-rebuild")
async def full_rebuild_index(admin: dict = Depends(require_admin)):
    """Full rebuild (Phase 0 data prep + Phase 1 chunking + Phase 2 embedding)."""
    try:
        logger.info("Full index rebuild triggered by %s", admin["username"])
        graph = build_full_index_graph()
        result = graph.invoke({})
        if result and result.get("error"):
            raise Exception(f"Pipeline error: {result['error']}")
        _refresh_singletons()
        doc_count, vec_count = _get_collection_stats()
        return {
            "message": "Full index rebuilt successfully",
            "document_count": doc_count,
            "vector_count": vec_count,
        }
    except Exception as exc:
        logger.exception("Full index rebuild failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Full index rebuild failed: {exc}",
        )


@router.post("/fix-lock")
async def fix_qdrant_lock(admin: dict = Depends(require_admin)):
    """Clear Qdrant lock files that may prevent collection access."""
    # Since we are using a Qdrant Server now, locks are handled internally
    return {"message": "Using Qdrant Server: no local file locks to clear."}


# ── Model settings (mounted separately at /api/settings) ─────────────────────

@settings_router.post("/api/settings/model")
async def change_model(body: ModelChangeRequest, admin: dict = Depends(require_admin)):
    """Change the active Ollama model and re-initialise components."""
    new_model = body.model.strip()
    if not new_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model name cannot be empty",
        )

    old_model = app_state.ollama_model
    app_state.ollama_model = new_model

    # Re-create singletons with the new model
    reset_shared_components()
    try:
        app_state.generator = get_shared_generator(model=new_model)
        app_state.retriever = get_shared_retriever(ollama_model=new_model)
    except Exception as exc:
        # Rollback on failure
        app_state.ollama_model = old_model
        reset_shared_components()
        logger.exception("Model change failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to switch model: {exc}",
        )

    logger.info("Model changed: %s → %s by %s", old_model, new_model, admin["username"])
    return {
        "message": f"Model changed to '{new_model}'",
        "previous_model": old_model,
        "current_model": new_model,
    }
