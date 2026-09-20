# Personal RAG — Backend

A **production-ready RAG (Retrieval-Augmented Generation) API** that powers an AI chatbot for answering questions about my portfolio, skills, and projects. Built with vanilla Python — no LangChain — to demonstrate clean AI systems architecture.

## Architecture

```
User Query
    │
    ▼
Rate Limit + Budget Check (Upstash Redis)
    │
    ▼
Exact-match Cache (Redis hash) ──► hit → return instantly
    │ miss
    ▼
Semantic Cache (ChromaDB, cosine ≥ 0.92) ──► hit → return in ~0.2s
    │ miss
    ▼
Query Rewrite (follow-up resolution via LLM)
    │
    ▼
Embedding → ChromaDB Retrieval (top-3 chunks)
    │
    ▼
Relevance Threshold Check ──► below 0.4 → "I don't have that info"
    │ pass
    ▼
LLM Generation (gpt-4o-mini, hardened system prompt)
    │
    ▼
Cache answer → Return with sources
```

## Stack

| Component | Choice |
|-----------|--------|
| API | FastAPI |
| Embeddings | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` (local, zero cost) |
| Vector DB | ChromaDB (local on-disk) |
| LLM | `gpt-4o-mini` |
| Cache Backend | Upstash Redis (exact-match + rate limiting) |
| Package Manager | [uv](https://github.com/astral-sh/uv) |

## Setup

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) — `pip install uv` or see [uv docs](https://docs.astral.sh/uv/)

### Install

```bash
cd backend

# Install all dependencies from lockfile (reproducible)
uv sync
```

### Configure

```bash
# Copy the template and fill in your secrets
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...
UPSTASH_REDIS_REST_URL=https://your-db.upstash.io
UPSTASH_REDIS_REST_TOKEN=your-token
FRONTEND_URL=http://localhost:5173   # or your deployed domain
```

### Build the Vector Index

Run once after setup (or whenever `Data/documents.jsonl` changes):

```bash
uv run python -c "from rag.ingest import build_index; build_index()"
```

This creates the local `chroma_db/` directory (gitignored).

### Run the API

```bash
uv run uvicorn main:app --reload
```

API will be available at `http://localhost:8000`.

- `POST /chat` — `{ "query": "..." }` → `{ "answer": "..." }`
- `GET /health` — health check

### Test in the console (no frontend needed)

```bash
uv run python test_chat.py
```

## Safety Features

- **Prompt injection hardening** — system prompt instructs the LLM to ignore embedded instructions
- **Input length cap** — 500 characters max per query
- **Per-IP rate limiting** — 10 req/min via Redis sliding window
- **Daily budget cap** — hard spend counter checked before every LLM call
- **CORS lock-down** — only the configured `FRONTEND_URL` is allowed

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `UPSTASH_REDIS_REST_URL` | ⚠️ Optional | Redis URL — caching/rate-limiting disabled if missing |
| `UPSTASH_REDIS_REST_TOKEN` | ⚠️ Optional | Redis token |
| `FRONTEND_URL` | ⚠️ Optional | CORS origin (defaults to `http://localhost:5173`) |
