"""
Optional Redis caching for the entitlement summary.

Degrades gracefully: if REDIS_URL isn't set (e.g. local dev without a
Redis container running), every call is a clean no-op cache miss --
nothing errors, it just always recomputes. This is what earns the
"Redis" line in the stack table instead of just documenting it as a
future intention.
"""
import os
import json

REDIS_URL = os.environ.get("REDIS_URL")
_client = None

if REDIS_URL:
    import redis
    _client = redis.from_url(REDIS_URL, decode_responses=True)

CACHE_TTL_SECONDS = 60  # short TTL: correctness over staleness for financial figures


def get_cached_summary(user_id: int):
    if not _client:
        return None
    raw = _client.get(f"summary:{user_id}")
    return json.loads(raw) if raw else None


def set_cached_summary(user_id: int, summary: dict):
    if not _client:
        return
    _client.setex(f"summary:{user_id}", CACHE_TTL_SECONDS, json.dumps(summary))


def invalidate_summary(user_id: int):
    """Call this whenever a claim is written -- a cached unclaimed-value
    figure must never outlive the claim that changed it by more than the
    TTL above."""
    if not _client:
        return
    _client.delete(f"summary:{user_id}")
