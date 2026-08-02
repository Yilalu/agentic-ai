"""
This file is an application configeration
"""

from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "policies"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
ACCOUNT_DATA = DATA_DIR / "account_data.csv"
CUSTOMER_DATA = DATA_DIR / "customer_data.csv"
FEE_REQUESTS_DATA = DATA_DIR / "fee_requests_data.csv"
FRAUD_ALERTS_DATA = DATA_DIR / "fraud_alerts_data.csv"
SUPPORT_CASES_DATA = DATA_DIR / "support_cases_data.csv"
TRANSACTIONS_DATA = DATA_DIR / "transactions_data.csv"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
MAX_REVISION_ATTEMPTS = int(os.getenv("MAX_REVISION_ATTEMPTS", "2"))

BANK_NAME = "Everyone's Bank"
COLLECTION_NAME = "banking_policies"
ALLOWED_CATEGORIES = {
    "card_dispute",
    "fee_refund",
    "failed_transfer",
    "account_lockout",
    "fraud",
    "loan"
    "identity_verification",
    "general",
}
    