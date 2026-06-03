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
QDRANT_PATH = OUTPUT_DIR / "qdrant_storage"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

QDRANT_COLLECTION = "retail_chunks"
TOP_K = 8
RERANK_TOP_K = 6
