"""
This file is an application configeration
"""

from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "policies"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
STORAGE_DIR = PROJECT_ROOT / "storage"

BANK_DB = DATA_DIR / "bank.db"
ACTION_LOG = STORAGE_DIR / "action_log.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

MAX_REVISION_ATTEMPTS: int = 2
MAX_CLARIFICATIONS: int = 3
MAX_TOOL_ATTEMPTS: int = 3
MAX_APPROVAL_LIMIT: float = 50.0  # A single authomatic path in the agent can approve refund below this amount

BANK_NAME = "Everyone's Bank"
COLLECTION_NAME = "banking_policies"

# Chroma chunking (build_vector_db + must stay stable for a given index)
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

FORCE_BAD_DRAFTS = 0
FORCE_TOOL_FAILURE = ""
FORCE_LLM_FAILURE = False
