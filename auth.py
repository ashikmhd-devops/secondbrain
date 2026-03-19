"""
Simple token-based auth.

Set APP_PASSWORD in env to enable. If unset, auth is disabled and
all requests are allowed through (useful for localhost dev).

Tokens are random 32-byte URL-safe strings persisted in the SQLite DB.
They expire after TOKEN_TTL seconds. Expired tokens are pruned on startup
and on each verify call.
"""

import os
import secrets
import time

from database import get_conn

APP_PASSWORD: str = os.getenv("APP_PASSWORD", "")
AUTH_ENABLED: bool = bool(APP_PASSWORD)

TOKEN_TTL = 7 * 24 * 3600  # 7 days


def _init_token_table() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_tokens (
                token      TEXT PRIMARY KEY,
                expires_at REAL NOT NULL
            )
            """
        )
        # Prune any already-expired tokens on startup
        conn.execute("DELETE FROM auth_tokens WHERE expires_at <= ?", (time.time(),))


def login(password: str) -> str:
    """Verify password and return a new session token. Raises ValueError on failure."""
    if not AUTH_ENABLED:
        raise ValueError("Auth is disabled on this server")
    if not secrets.compare_digest(password, APP_PASSWORD):
        raise ValueError("Invalid password")
    token = secrets.token_urlsafe(32)
    expires_at = time.time() + TOKEN_TTL
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO auth_tokens (token, expires_at) VALUES (?, ?)",
            (token, expires_at),
        )
    return token


def verify_token(token: str) -> bool:
    """Return True if token is valid and not expired."""
    if not AUTH_ENABLED:
        return True
    if not token:
        return False
    now = time.time()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT expires_at FROM auth_tokens WHERE token = ?", (token,)
        ).fetchone()
        if row is None:
            return False
        if now > row["expires_at"]:
            conn.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))
            return False
    return True


def revoke_token(token: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))
