import os
import hashlib
from typing import Optional
from datetime import datetime, timezone

from upstash_redis import Redis
from upstash_redis.errors import UpstashError

from rag.config import CONTENT_HASH

_redis = None
try:
    if os.getenv("UPSTASH_REDIS_REST_URL") and os.getenv("UPSTASH_REDIS_REST_TOKEN"):
        _redis = Redis.from_env()
    else:
        print("Warning: Upstash Redis env vars missing. Caching and rate limiting are disabled.")
except Exception as e:
    print(f"Warning: Upstash Redis init failed: {e}")

DAILY_SPEND_CAP = int(os.getenv("DAILY_LLM_BUDGET", "500"))
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_SECS = 60

class RateLimitExceeded(Exception):
    pass

class BudgetExceeded(Exception):
    pass

def check_rate_limit(ip_address: str):
    if not _redis or ip_address == "unknown":
        return
    
    key = f"rate_limit:{ip_address}"
    try:
        pipeline = _redis.pipeline()
        pipeline.incr(key)
        pipeline.expire(key, RATE_LIMIT_WINDOW_SECS)
        results = pipeline.exec()
        current = results[0]
    except UpstashError as e:
        print(f"Warning: Redis rate limiting failed: {e}")
        return
        
    if current > RATE_LIMIT_REQUESTS:
        raise RateLimitExceeded("You're sending too many requests. Please wait a moment.")

def increment_and_check_budget():
    if not _redis:
        return
        
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"daily_budget:{today}"
    
    try:
        pipeline = _redis.pipeline()
        pipeline.incr(key)
        pipeline.expire(key, 86400)
        results = pipeline.exec()
        current = results[0]
    except UpstashError as e:
        print(f"Warning: Redis budget check failed: {e}")
        return
        
    if current > DAILY_SPEND_CAP:
        raise BudgetExceeded("I'm currently resting for today. Please try again tomorrow!")

def get_cache_key(query: str) -> str:
    normalized = query.strip().lower()
    query_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    return f"cache:{CONTENT_HASH}:{query_hash}"

def get_cached_answer(query: str) -> Optional[str]:
    if not _redis:
        return None
        
    key = get_cache_key(query)
    try:
        return _redis.get(key)
    except UpstashError as e:
        print(f"Warning: Redis get_cached_answer failed: {e}")
        return None

def set_cached_answer(query: str, answer: str):
    if not _redis:
        return
        
    key = get_cache_key(query)
    try:
        _redis.set(key, answer, ex=604800)
    except UpstashError as e:
        print(f"Warning: Redis set_cached_answer failed: {e}")
