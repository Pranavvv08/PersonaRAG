from dotenv import load_dotenv
load_dotenv()

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.generation import answer_query

app = FastAPI(title="Personal RAG API")

# Vite's default dev server port. Add your deployed frontend origin too.
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., max_length=500)


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    ip_address = request.client.host if request.client else "unknown"
    return ChatResponse(answer=answer_query(req.query, ip_address=ip_address))


@app.get("/health")
def health():
    return {"status": "ok"}
