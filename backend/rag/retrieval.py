import hashlib
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import (
    CACHE_COLLECTION_NAME,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LIST_KEYWORDS,
    SEMANTIC_CACHE_THRESHOLD,
    TOP_K,
    TYPE_KEYWORDS,
)

# Loaded once at process startup, reused across requests.
_model = SentenceTransformer(EMBEDDING_MODEL)
_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

_collection = None
_cache_collection = None

def get_collection():
    global _collection
    if _collection is None:
        try:
            _collection = _client.get_collection(COLLECTION_NAME)
        except Exception as e:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found at {CHROMA_DB_PATH}. "
                "Run `uv run python -m rag.ingest` first to build the index."
            ) from e
    return _collection

def get_cache_collection():
    global _cache_collection
    if _cache_collection is None:
        _cache_collection = _client.get_or_create_collection(CACHE_COLLECTION_NAME)
    return _cache_collection

def generate_embedding(text: str) -> list[float]:
    return _model.encode(text, normalize_embeddings=True).tolist()

def semantic_cache_search(query_embedding: list[float]) -> Optional[str]:
    results = get_cache_collection().query(
        query_embeddings=[query_embedding],
        n_results=1
    )
    if not results["ids"][0]:
        return None
        
    dist = results["distances"][0][0]
    if (1 - dist) >= SEMANTIC_CACHE_THRESHOLD:
        return results["metadatas"][0][0]["answer"]
    return None

def save_to_semantic_cache(query: str, query_embedding: list[float], answer: str):
    doc_id = hashlib.sha256(query.encode('utf-8')).hexdigest()
    # add() will error if ID exists, upsert() gracefully handles duplicates
    get_cache_collection().upsert(
        ids=[doc_id],
        embeddings=[query_embedding],
        documents=[query],
        metadatas=[{"answer": answer}]
    )


def infer_type(query: str) -> Optional[str]:
    q = query.lower()
    for type_, keywords in TYPE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return type_
    return None


def is_list_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in LIST_KEYWORDS)


def search(query: str, query_embedding: list[float], top_k: int = TOP_K):
    """Similarity-ranked search — best for 'tell me about X' style queries."""
    inferred = infer_type(query)


    where = {"type": inferred} if inferred else None
    results = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )

    if not results["ids"][0]:
        return []

    # Chroma returns cosine *distance* (0 = identical); convert to similarity.
    return [
        (doc_id, text, 1 - dist)
        for doc_id, text, dist in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        )
    ]


def fetch_all(type_: Optional[str] = None):
    """Exhaustive metadata fetch — no similarity ranking, no top_k cap.
    Use for 'list all X' style queries where completeness matters more
    than relevance ordering. type_=None fetches the whole corpus."""
    where = {"type": type_} if type_ else None
    results = get_collection().get(where=where)
    if not results["ids"]:
        return []
    return list(zip(results["ids"], results["documents"]))
