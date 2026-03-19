"""
Simple token-based auth.

Set APP_PASSWORD in env to enable. If unset, auth is disabled and
all requests are allowed through (useful for localhost dev).

Tokens are random 32-byte URL-safe strings stored in memory.
They expire after TOKEN_TTL seconds. On server restart all sessions
are invalidated (expected behaviour for a personal app).
"""

import os
import secrets
import time

APP_PASSWORD: str = os.getenv("APP_PASSWORD", "")
AUTH_ENABLED: bool = bool(APP_PASSWORD)

TOKEN_TTL = 7 * 24 * 3600  # 7 days

# token → expiry timestamp
_tokens: dict[str, float] = {}


def login(password: str) -> str:
    """Verify password and return a new session token. Raises ValueError on failure."""
    if not AUTH_ENABLED:
        raise ValueError("Auth is disabled on this server")
    if not secrets.compare_digest(password, APP_PASSWORD):
        raise ValueError("Invalid password")
    token = secrets.token_urlsafe(32)
    _tokens[token] = time.time() + TOKEN_TTL
    return token


def verify_token(token: str) -> bool:
    """Return True if token is valid and not expired."""
    if not AUTH_ENABLED:
        return True
    if not token:
        return False
    exp = _tokens.get(token)
    if exp is None:
        return False
    if time.time() > exp:
        _tokens.pop(token, None)
        return False
    return True


def revoke_token(token: str) -> None:
    _tokens.pop(token, None)
