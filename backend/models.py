"""Pydantic schemas for the Retail Knowledge Bot API."""

from pydantic import BaseModel, Field


# ── Auth schemas ──────────────────────────────────────────────────────────────

class SignInRequest(BaseModel):
    username: str
    password: str


class SignUpRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


class UserInfo(BaseModel):
    username: str
    role: str


# ── Chat schemas ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    chat_history: list[dict] = Field(default_factory=list)


class SourceInfo(BaseModel):
    source_doc: str = ""
    title: str = ""
    term: str = ""
    domain: str = ""
    relevance_score: float = 0.0


# ── Index / Admin schemas ────────────────────────────────────────────────────

class IndexStatusResponse(BaseModel):
    indexed: bool
    document_count: int = 0
    vector_count: int = 0
    current_model: str = ""


class ChangeRoleRequest(BaseModel):
    role: str


class ModelChangeRequest(BaseModel):
    model: str
