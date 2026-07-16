import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
PHASE_0_DIR = OUTPUT_DIR / "phase_0"
PHASE_0_JSON = PHASE_0_DIR / "retail_knowledge_base.json"
PHASE_1_DIR = OUTPUT_DIR / "phase_1"
QDRANT_URL = os.environ.get("QDRANT_URL", "https://cd18c5ab-228a-4070-b018-05602213c45c.us-east-2-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

# ── Embedding: Ollama nomic-embed-text (768-dim, fully local) ─────────────────
EMBEDDING_MODEL = "nomic-embed-text"   # ollama pull nomic-embed-text
EMBEDDING_DIM = 768

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── Ollama LLM (fully local, no API key needed) ───────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")  # or qwen2.5:7b

# ── Speed controls ────────────────────────────────────────────────────────────
# Query expansion: adds ~10-15s per query (extra LLM call). Disable for speed.
ENABLE_QUERY_EXPANSION = os.environ.get("ENABLE_QUERY_EXPANSION", "false").lower() == "true"
# LLM reranking: adds ~40-50s per query (extra LLM call). Disable for speed.
# When False, chunks are ordered by Qdrant vector similarity score (still good quality).
ENABLE_LLM_RERANK = os.environ.get("ENABLE_LLM_RERANK", "false").lower() == "true"

# Reranking is done via Ollama LLM prompt (no HuggingFace CrossEncoder needed)

QDRANT_COLLECTION = "retail_chunks"
TOP_K = 6           # reduced from 8 for faster generation prompts
RERANK_TOP_K = 5
