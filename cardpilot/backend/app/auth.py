"""
Minimal API access control -- Phase 1 placeholder.

This is deliberately NOT real per-user authentication. It's a shared-
secret gate (a single API key) that stops the API from being wide open
to anyone who guesses a user ID -- the gap flagged in the Phase 1 audit.

Real per-user auth (JWT issued at login, scoped so a user can only ever
read their own user_id) is required before this API is exposed to real
users or real bank data. Do not treat this as sufficient beyond local
development / demo purposes.
"""
import os
from fastapi import Header, HTTPException

API_KEY = os.environ.get("CARDPILOT_API_KEY", "dev-only-placeholder-key")


def require_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True
