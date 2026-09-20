import os
import hashlib
from typing import Optional
from datetime import datetime, timezone

from upstash_redis import Redis

_redis = None
try:
    if os.getenv("UPSTASH_REDIS_REST_URL") and os.getenv("UPSTASH_REDIS_REST_TOKEN"):
        _redis = Redis.from_env()
    else:
        print("Warning: Upstash Redis env vars missing. Caching and rate limiting are disabled.")
except Exception as e:
    print(f"Warning: Upstash Redis init failed: {e}")

DAILY_SPEND_CAP = 50
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
    current = _redis.incr(key)
    if current == 1:
        _redis.expire(key, RATE_LIMIT_WINDOW_SECS)
        
    if current > RATE_LIMIT_REQUESTS:
        raise RateLimitExceeded("You're sending too many requests. Please wait a moment.")

def increment_and_check_budget():
    if not _redis:
        return
        
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"daily_budget:{today}"
    
    current = _redis.incr(key)
    if current == 1:
        _redis.expire(key, 86400)
        
    if current > DAILY_SPEND_CAP:
        raise BudgetExceeded("I'm currently resting for today. Please try again tomorrow!")

def get_cached_answer(query: str) -> Optional[str]:
    if not _redis:
        return None
        
    normalized = query.strip().lower()
    key = f"cache:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"
    return _redis.get(key)

def set_cached_answer(query: str, answer: str):
    if not _redis:
        return
        
    key = f"cache:{hashlib.sha256(query.encode('utf-8')).hexdigest()}"
    _redis.set(key, answer, ex=604800)
