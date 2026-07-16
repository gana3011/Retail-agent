"""Document management routes (admin-only): list, upload, delete."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from pydantic import BaseModel

from pipeline.config import DATA_DIR
from pipeline.graph import build_full_index_graph
from pipeline.nodes import reset_shared_components, get_shared_retriever, get_shared_generator

from backend.deps import require_admin, app_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


class DeleteRequest(BaseModel):
    filenames: list[str]


def _trigger_full_rebuild() -> dict:
    """Run the full index pipeline and refresh shared components."""
    graph = build_full_index_graph()
    graph.invoke({})
    reset_shared_components()
    app_state.retriever = get_shared_retriever()
    app_state.generator = get_shared_generator()
    app_state.indexed = True
    return {"rebuild": "success"}


@router.get("/")
async def list_documents(admin: dict = Depends(require_admin)):
    """List all .docx files in the data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for f in sorted(DATA_DIR.iterdir()):
        if f.suffix.lower() == ".docx" and f.is_file():
            files.append({
                "filename": f.name,
                "size_bytes": f.stat().st_size,
            })
    return {"documents": files, "count": len(files)}


@router.post("/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    admin: dict = Depends(require_admin),
):
    """Upload .docx files and trigger a full index rebuild."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    uploaded = []

    for file in files:
        if not file.filename:
            continue
        if not file.filename.lower().endswith(".docx"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Only .docx files are supported. Got: '{file.filename}'",
            )

        dest = DATA_DIR / file.filename
        content = await file.read()
        dest.write_bytes(content)
        uploaded.append(file.filename)
        logger.info("Uploaded document: %s (%d bytes)", file.filename, len(content))

    if not uploaded:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid files provided",
        )

    # Trigger full rebuild after upload
    try:
        _trigger_full_rebuild()
    except Exception as exc:
        logger.exception("Rebuild after upload failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Files uploaded but rebuild failed: {exc}",
        )

    return {
        "message": f"Uploaded {len(uploaded)} file(s) and rebuilt index",
        "uploaded": uploaded,
    }


@router.delete("/")
async def delete_documents(
    body: DeleteRequest,
    admin: dict = Depends(require_admin),
):
    """Delete specified documents and trigger a rebuild."""
    deleted = []
    not_found = []

    for filename in body.filenames:
        filepath = DATA_DIR / filename
        if filepath.exists() and filepath.is_file():
            filepath.unlink()
            deleted.append(filename)
            logger.info("Deleted document: %s", filename)
        else:
            not_found.append(filename)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No files found to delete: {not_found}",
        )

    # Trigger rebuild after deletion
    try:
        _trigger_full_rebuild()
    except Exception as exc:
        logger.exception("Rebuild after delete failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Files deleted but rebuild failed: {exc}",
        )

    return {
        "message": f"Deleted {len(deleted)} file(s) and rebuilt index",
        "deleted": deleted,
        "not_found": not_found,
    }
