from dotenv import load_dotenv
load_dotenv(override=True)

import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.generation import answer_query

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Personal RAG API")

# Default origins: local Vite dev + production Vercel portfolio
default_origins = [
    "http://localhost:5173",
    "https://pranavsasank-portfolio.vercel.app",
]

# Additional origins from FRONTEND_URL env var (comma-separated if multiple)
env_origins = os.getenv("FRONTEND_URL", "")
if env_origins:
    for url in env_origins.split(","):
        cleaned = url.strip().rstrip("/")
        if cleaned and cleaned not in default_origins:
            default_origins.append(cleaned)

app.add_middleware(
    CORSMiddleware,
    allow_origins=default_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., max_length=500)


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    ip_address = request.headers.get("x-forwarded-for")
    if ip_address:
        ip_address = ip_address.split(",")[0].strip()
    else:
        ip_address = request.client.host if request.client else "unknown"

    try:
        answer = answer_query(req.query, ip_address=ip_address)
        return ChatResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error processing chat query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate answer. Please try again shortly."
        )


@app.get("/health")
def health():
    return {"status": "ok"}
