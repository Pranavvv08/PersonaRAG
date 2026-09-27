import hashlib
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
DATA_PATH = BASE_DIR / "Data" / "documents.jsonl"
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "data" / "documents.jsonl"

CHROMA_DB_PATH = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
CHAT_MODEL = "gpt-4o-mini"

COLLECTION_NAME = "portfolio"
CACHE_COLLECTION_NAME = "semantic_cache"
TOP_K = 3
MIN_SIMILARITY = 0.4
SEMANTIC_CACHE_THRESHOLD = 0.92

# Global Content Hash for Cache Invalidation
CONTENT_HASH = "unknown"
if DATA_PATH.exists():
    with open(DATA_PATH, "rb") as f:
        CONTENT_HASH = hashlib.sha256(f.read()).hexdigest()

_RAW_TYPE_KEYWORDS = {
    "project": ["project", "projects", "built", "build", "app", "apps", "application", "applications"],
    "skill": ["technology", "technologies", "skill", "skills", "know", "language", "languages", "framework", "frameworks", "tool", "tools"],
    "education": ["education", "degree", "college", "university", "cgpa", "school"],
    "certification": ["certification", "certifications", "certificate", "certificates", "certified"],
    "course": ["course", "courses", "learned", "studied"],
    "contact": ["contact", "email", "linkedin", "github", "reach"],
    "about": ["located", "location", "based", "role", "roles", "looking for"],
}

TYPE_KEYWORDS = {
    k: re.compile(r'\b(' + '|'.join(re.escape(w) for w in v) + r')\b', re.IGNORECASE)
    for k, v in _RAW_TYPE_KEYWORDS.items()
}

_RAW_LIST_KEYWORDS = ["all", "list", "every", "each", "what are", "how many"]
LIST_KEYWORDS = re.compile(r'\b(' + '|'.join(re.escape(w) for w in _RAW_LIST_KEYWORDS) + r')\b', re.IGNORECASE)

MAX_HISTORY_TURNS = 6
