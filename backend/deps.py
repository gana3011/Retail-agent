"""Shared dependencies: JWT handling, app state, and auth dependencies."""

import os
import sys
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ── Ensure project root is importable ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from pipeline.config import OLLAMA_MODEL

# ── JWT configuration ─────────────────────────────────────────────────────────

JWT_SECRET = os.environ.get("JWT_SECRET", secrets.token_urlsafe(64))
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

security = HTTPBearer()


def create_access_token(username: str, role: str) -> str:
    """Create a JWT access token with username and role claims."""
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    """Decode and validate a JWT token. Raises on failure."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: Optional[str] = payload.get("sub")
        role: Optional[str] = payload.get("role")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
            )
        return {"username": username, "role": role}
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
        )


# ── FastAPI dependencies ──────────────────────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Extract and validate the current user from the Authorization header."""
    return _decode_token(credentials.credentials)


async def require_admin(
    user: dict = Depends(get_current_user),
) -> dict:
    """Ensure the current user has admin privileges."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return user


# ── Application state singleton ───────────────────────────────────────────────

class AppState:
    """Mutable application state shared across all routes."""

    def __init__(self) -> None:
        self.indexed: bool = False
        self.retriever = None
        self.generator = None
        self.ollama_model: str = OLLAMA_MODEL


app_state = AppState()
