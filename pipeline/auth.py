"""
auth.py – lightweight file-backed authentication for the Retail KB app.

Users are stored in data/users.json as:
    {
      "username": {
        "password_hash": "<bcrypt-hash>",
        "role": "admin" | "user"
      }
    }

A default admin account is created on first run if no users exist.
"""

import json
import os
from pathlib import Path

import bcrypt

# Path to the user store (sibling of data/)
_USERS_FILE = Path(__file__).parent.parent / "data" / "users.json"

# Default admin credentials (change via the app or environment variables)
_DEFAULT_ADMIN_USERNAME = os.environ.get("DEFAULT_ADMIN_USERNAME", "admin")
_DEFAULT_ADMIN_PASSWORD = os.environ.get("DEFAULT_ADMIN_PASSWORD", "Admin@1234")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_users() -> dict:
    """Load user store from disk. Returns empty dict if file missing."""
    _USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not _USERS_FILE.exists():
        return {}
    with open(_USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: dict) -> None:
    """Persist user store to disk."""
    _USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_admin() -> None:
    """Create the default admin account if no users exist yet."""
    users = _load_users()
    if not users:
        users[_DEFAULT_ADMIN_USERNAME] = {
            "password_hash": _hash_password(_DEFAULT_ADMIN_PASSWORD),
            "role": "admin",
        }
        _save_users(users)


# ── Public API ────────────────────────────────────────────────────────────────

def signup(username: str, password: str, role: str = "user") -> tuple[bool, str]:
    """
    Register a new user.

    Returns (success: bool, message: str).
    """
    username = username.strip()
    if not username:
        return False, "Username cannot be empty."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = _load_users()
    if username in users:
        return False, f"Username '{username}' is already taken."

    users[username] = {
        "password_hash": _hash_password(password),
        "role": role,
    }
    _save_users(users)
    return True, f"Account created successfully. Welcome, {username}!"


def signin(username: str, password: str) -> tuple[bool, str, str | None]:
    """
    Authenticate a user.

    Returns (success: bool, message: str, role: str | None).
    """
    users = _load_users()
    user = users.get(username.strip())
    if user is None:
        return False, "Invalid username or password.", None
    if not _check_password(password, user["password_hash"]):
        return False, "Invalid username or password.", None
    return True, f"Welcome back, {username}!", user.get("role", "user")


def is_admin(role: str | None) -> bool:
    """Return True if the given role is 'admin'."""
    return role == "admin"


def list_users() -> list[dict]:
    """Return a list of {username, role} dicts (no passwords)."""
    users = _load_users()
    return [{"username": u, "role": v["role"]} for u, v in users.items()]


def change_role(username: str, new_role: str) -> tuple[bool, str]:
    """Admin utility: change a user's role."""
    users = _load_users()
    if username not in users:
        return False, f"User '{username}' not found."
    users[username]["role"] = new_role
    _save_users(users)
    return True, f"Role for '{username}' updated to '{new_role}'."


def delete_user(username: str) -> tuple[bool, str]:
    """Admin utility: delete a user account."""
    users = _load_users()
    if username not in users:
        return False, f"User '{username}' not found."
    del users[username]
    _save_users(users)
    return True, f"User '{username}' deleted."
