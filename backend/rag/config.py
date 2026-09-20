from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
DATA_PATH = BASE_DIR / "data" / "documents.jsonl"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
CHAT_MODEL = "gpt-4o-mini"

COLLECTION_NAME = "portfolio"
CACHE_COLLECTION_NAME = "semantic_cache"
TOP_K = 3
MIN_SIMILARITY = 0.4
SEMANTIC_CACHE_THRESHOLD = 0.92

TYPE_KEYWORDS = {
    "project": ["project", "built", "build", "app", "application"],
    "skill": ["technology", "technologies", "skill", "know", "language", "framework", "tool"],
    "education": ["education", "degree", "college", "university", "cgpa", "school"],
    "certification": ["certification", "certificate", "certified"],
    "course": ["course", "learned", "studied"],
    "contact": ["contact", "email", "linkedin", "github", "reach"],
    "about": ["located", "location", "based", "role", "roles", "looking for"],
}

LIST_KEYWORDS = ["all", "list", "every", "each", "what are", "how many"]
MAX_HISTORY_TURNS = 6
