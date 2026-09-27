FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install CPU-only PyTorch first (keeps image small, under 500MB instead of 3GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install app dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Pre-cache the embedding model during the build phase
# (This ensures production boots in 1s without hanging on Hugging Face downloads)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')"

# 4. Copy the backend code
COPY . .

# 5. Tell Hugging Face to stay offline (use pre-cached model)
ENV HF_HUB_OFFLINE=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
