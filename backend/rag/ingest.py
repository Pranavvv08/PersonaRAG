import json

import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import CHROMA_DB_PATH, COLLECTION_NAME, CACHE_COLLECTION_NAME, DATA_PATH, EMBEDDING_MODEL


def load_documents():
    docs = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            docs.append(
                {
                    "id": data["id"],
                    "text": data["text"],
                    "metadata": {"id": data.get("id"), **data.get("metadata", {})},
                }
            )
    return docs


def build_index():
    docs = load_documents()
    print(f"Loaded {len(docs)} documents (no splitting)")

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    print("Embeddings shape:", embeddings.shape)

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass
    
    try:
        client.delete_collection(CACHE_COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[d["id"] for d in docs],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[d["metadata"] for d in docs],
    )
    print(f"Loaded {collection.count()} chunks into ChromaDB at {CHROMA_DB_PATH}")


if __name__ == "__main__":
    build_index()
