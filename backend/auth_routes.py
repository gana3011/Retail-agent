"""Authentication routes: sign-in, sign-up, and current-user info."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import APIRouter, Depends, HTTPException, status

from pipeline.auth import signin, signup

from backend.deps import create_access_token, get_current_user
from backend.models import SignInRequest, SignUpRequest, TokenResponse, UserInfo

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/signin", response_model=TokenResponse)
async def sign_in(body: SignInRequest):
    """Authenticate a user and return a JWT token."""
    success, message, role = signin(body.username, body.password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message,
        )
    token = create_access_token(body.username, role)
    return TokenResponse(
        access_token=token,
        username=body.username,
        role=role,
    )


@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def sign_up(body: SignUpRequest):
    """Register a new user account."""
    success, message = signup(body.username, body.password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=message,
        )
    return {"message": message}


@router.get("/me", response_model=UserInfo)
async def get_me(user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's info."""
    return UserInfo(username=user["username"], role=user.get("role", "user"))
