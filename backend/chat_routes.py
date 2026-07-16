"""Chat route: SSE-streamed RAG responses."""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from pipeline.nodes import get_shared_retriever, get_shared_generator
from pipeline.generator import AnswerGenerator

from backend.deps import get_current_user, app_state
from backend.models import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


async def _sse_stream(question: str, chat_history: list[dict]):
    """Generator that yields SSE-formatted events for the RAG pipeline."""
    try:
        retriever = get_shared_retriever()
        generator = get_shared_generator()

        # 1. Retrieve relevant chunks
        chunks = retriever.retrieve(question)

        # 2. Extract source metadata
        sources = AnswerGenerator._extract_sources(chunks)

        # 3. Stream answer tokens
        for token in generator.generate_stream(question, chunks, chat_history=chat_history):
            payload = json.dumps({"type": "token", "content": token})
            yield f"data: {payload}\n\n"

        # 4. Send sources
        sources_payload = json.dumps({"type": "sources", "sources": sources})
        yield f"data: {sources_payload}\n\n"

        # 5. Signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as exc:
        logger.exception("Chat streaming error")
        error_payload = json.dumps({"type": "error", "content": str(exc)})
        yield f"data: {error_payload}\n\n"


@router.post("/chat")
async def chat(body: ChatRequest, user: dict = Depends(get_current_user)):
    """Stream a RAG-powered answer via Server-Sent Events."""
    if not app_state.indexed:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Index not built yet. Ask an admin to build the index first.",
        )

    return StreamingResponse(
        _sse_stream(body.question, body.chat_history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
