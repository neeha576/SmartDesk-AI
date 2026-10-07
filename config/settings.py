"""Central settings – the ONLY module that reads environment variables (.env). No secrets in code.

Model and provider names have no fallback defaults: they must be set in .env, and a missing
value fails fast with a clear message instead of silently using a different model.
Numeric tuning knobs (chunk size, batch size, thresholds) keep sensible defaults.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def required(name: str) -> str:
    """Return a required env var, or raise a clear error if it's missing/empty."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to your .env file (see .env.example).")
    return value


def optional(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


# --- LLM (read when the LLM is created, so scripts that don't use it don't need it) ---
def llm_provider() -> str:
    return required("LLM_PROVIDER").lower()


def llm_model() -> str:
    return required("LLM_MODEL")


def llm_fallback_model() -> str | None:
    """Optional second model tried once if LLM_MODEL keeps failing (e.g. LLM_FALLBACK_MODEL=gpt-4o-mini)."""
    return optional("LLM_FALLBACK_MODEL")


LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))


# --- Embeddings ---
def embedding_provider() -> str:
    return required("EMBEDDING_PROVIDER").lower()


def embedding_model() -> str:
    return required("EMBEDDING_MODEL")


def sparse_model() -> str:
    return required("SPARSE_MODEL")


EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# --- Qdrant ---
QDRANT_URL = optional("QDRANT_URL")
QDRANT_API_KEY = optional("QDRANT_API_KEY")
QDRANT_PATH = os.getenv("QDRANT_PATH", "data/vector_store")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "smartdesk_kb")

# --- Retrieval ---
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.30"))  # min top dense cosine score (calibrate with try_agent --eval)
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))  # past messages sent to the LLM

# --- Semantic cache (memory/semantic_cache.py) -------------------------------------------------
SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
CACHE_COLLECTION = os.getenv("CACHE_COLLECTION", "smartdesk_cache")
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.93"))   # min cosine similarity to reuse an answer
CACHE_TTL_HOURS = float(os.getenv("CACHE_TTL_HOURS", "168"))    # cached answers expire after a week

# --- Ticketing (Jira) ---
USE_MOCK_TICKETING = os.getenv("USE_MOCK_TICKETING", "false").lower() == "true"
JIRA_ISSUE_TYPE = os.getenv("JIRA_ISSUE_TYPE", "Task")
JIRA_TIMEOUT_SECONDS = float(os.getenv("JIRA_TIMEOUT_SECONDS", "15"))
JIRA_MAX_RETRIES = int(os.getenv("JIRA_MAX_RETRIES", "3"))
TICKET_DB_PATH = os.getenv("TICKET_DB_PATH", "data/tickets/tickets.db")


def jira_base_url() -> str:
    return required("JIRA_BASE_URL").rstrip("/")


def jira_email() -> str:
    return required("JIRA_EMAIL")


def jira_api_token() -> str:
    return required("JIRA_API_TOKEN")


def jira_project_key(category: str) -> str:
    """Jira project for a ticket category: JIRA_IT_PROJECT_KEY or JIRA_HR_PROJECT_KEY."""
    return required(f"JIRA_{category.upper()}_PROJECT_KEY")
