"""Admin routes: user management (list, change role, delete)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import APIRouter, Depends, HTTPException, status

from pipeline.auth import list_users, change_role, delete_user

from backend.deps import require_admin
from backend.models import ChangeRoleRequest, UserInfo

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/users", response_model=list[UserInfo])
async def get_users(admin: dict = Depends(require_admin)):
    """List all registered users."""
    users = list_users()
    return [UserInfo(username=u["username"], role=u["role"]) for u in users]


@router.put("/users/{username}/role")
async def update_user_role(
    username: str,
    body: ChangeRoleRequest,
    admin: dict = Depends(require_admin),
):
    """Change a user's role (admin-only)."""
    if body.role not in ("admin", "user"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be 'admin' or 'user'",
        )

    success, message = change_role(username, body.role)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message,
        )
    return {"message": message}


@router.delete("/users/{username}")
async def remove_user(
    username: str,
    admin: dict = Depends(require_admin),
):
    """Delete a user account (admin-only)."""
    # Prevent self-deletion
    if username == admin["username"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    success, message = delete_user(username)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message,
        )
    return {"message": message}
