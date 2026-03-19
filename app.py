import json
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from auth import AUTH_ENABLED, login, revoke_token, verify_token
from database import get_conn, init_db
from enricher import enrich_memory
from ollama import OLLAMA_HOST, get_embedding, ollama_chat

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _emb_to_blob(embedding: list[float]) -> bytes:
    return np.array(embedding, dtype=np.float32).tobytes()


def _blob_to_emb(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _row_to_dict(row) -> dict:
    return dict(row)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def require_auth(authorization: str = Header(default="")) -> None:
    """FastAPI dependency — enforces token auth when APP_PASSWORD is set."""
    if not AUTH_ENABLED:
        return
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization[len("Bearer "):]
    if not verify_token(token):
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MemoryCreate(BaseModel):
    raw_text: str


class MemoryUpdate(BaseModel):
    raw_text: str


class RecallRequest(BaseModel):
    question: str
    top_k: int = 5
    threshold: float = 0.25


class SearchRequest(BaseModel):
    query: str
    limit: int = 20


class LoginRequest(BaseModel):
    password: str


# ---------------------------------------------------------------------------
# App & routers
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# All /api/* routes share this router; require_auth is enforced on every one.
api = APIRouter(prefix="/api", dependencies=[Depends(require_auth)])


# ---------------------------------------------------------------------------
# Public routes (no auth)
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "secondbrain API"}


# ---------------------------------------------------------------------------
# Auth routes (public)
# ---------------------------------------------------------------------------

@app.get("/auth/status")
def auth_status():
    return {"auth_enabled": AUTH_ENABLED}


@app.post("/auth/login")
def auth_login(body: LoginRequest):
    if not AUTH_ENABLED:
        return {"token": None, "auth_enabled": False}
    try:
        token = login(body.password)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid password")
    return {"token": token, "auth_enabled": True}


@app.post("/auth/logout")
def auth_logout(authorization: str = Header(default="")):
    if authorization.startswith("Bearer "):
        revoke_token(authorization[len("Bearer "):])
    return {"ok": True}


# ---------------------------------------------------------------------------
# GET /api/models  (protected)
# ---------------------------------------------------------------------------

@api.get("/models")
async def check_models():
    required = {"nomic-embed-text", "llama3.2:latest"}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
        available = {m["name"] for m in r.json().get("models", [])}
    except Exception as e:
        return {"ollama_reachable": False, "error": str(e), "missing": list(required)}

    missing = [m for m in required if not any(a == m or a.startswith(m.split(":")[0] + ":") for a in available)]
    return {
        "ollama_reachable": True,
        "ollama_host": OLLAMA_HOST,
        "available": sorted(available),
        "missing": missing,
        "ready": len(missing) == 0,
    }


# ---------------------------------------------------------------------------
# POST /api/memory
# ---------------------------------------------------------------------------

@api.post("/memory", status_code=201)
async def create_memory(body: MemoryCreate):
    enriched = await enrich_memory(body.raw_text)
    embedding = await get_embedding(body.raw_text)

    mem_id = str(uuid.uuid4())
    tags = json.dumps(enriched.get("tags", []))
    data_types = json.dumps(enriched.get("data_types", []))
    extracted_entities = json.dumps(enriched.get("extracted_entities", {}))
    is_sensitive = int(bool(enriched.get("is_sensitive", False)))
    title = enriched.get("title", body.raw_text[:80])
    category = enriched.get("category", "General")

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO memory
                (id, raw_text, title, category, tags, data_types,
                 extracted_entities, is_sensitive, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mem_id, body.raw_text, title, category,
                tags, data_types, extracted_entities,
                is_sensitive, _emb_to_blob(embedding),
            ),
        )

    return {
        "id": mem_id,
        "title": title,
        "category": category,
        "tags": enriched.get("tags", []),
        "data_types": enriched.get("data_types", []),
        "is_sensitive": bool(is_sensitive),
    }


# ---------------------------------------------------------------------------
# GET /api/memory
# ---------------------------------------------------------------------------

@api.get("/memory")
def list_memories(
    category: Optional[str] = None,
    tag: Optional[str] = None,
    sensitive: Optional[bool] = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    clauses = []
    params: list = []

    if category:
        clauses.append("category = ?")
        params.append(category)
    if sensitive is not None:
        clauses.append("is_sensitive = ?")
        params.append(int(sensitive))
    if tag:
        clauses.append("EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)")
        params.append(tag)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    with get_conn() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM memory {where}", params
        ).fetchone()[0]

        rows = conn.execute(
            f"""
            SELECT id, raw_text, title, category, tags, data_types,
                   is_sensitive, created_at, updated_at
            FROM memory {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        ).fetchall()

    items = []
    for row in rows:
        d = _row_to_dict(row)
        d["tags"] = json.loads(d["tags"] or "[]")
        d["data_types"] = json.loads(d["data_types"] or "[]")
        d["is_sensitive"] = bool(d["is_sensitive"])
        items.append(d)

    return {"total": total, "items": items}


# ---------------------------------------------------------------------------
# GET /api/memory/{id}
# ---------------------------------------------------------------------------

@api.get("/memory/{mem_id}")
def get_memory(mem_id: str):
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, raw_text, title, category, tags, data_types,
                   extracted_entities, is_sensitive, created_at, updated_at
            FROM memory WHERE id = ?
            """,
            (mem_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Memory not found")

    d = _row_to_dict(row)
    d["tags"] = json.loads(d["tags"] or "[]")
    d["data_types"] = json.loads(d["data_types"] or "[]")
    d["extracted_entities"] = json.loads(d["extracted_entities"] or "{}")
    d["is_sensitive"] = bool(d["is_sensitive"])
    return d


# ---------------------------------------------------------------------------
# PUT /api/memory/{id}
# ---------------------------------------------------------------------------

@api.put("/memory/{mem_id}")
async def update_memory(mem_id: str, body: MemoryUpdate):
    with get_conn() as conn:
        exists = conn.execute(
            "SELECT 1 FROM memory WHERE id = ?", (mem_id,)
        ).fetchone()

    if not exists:
        raise HTTPException(status_code=404, detail="Memory not found")

    enriched = await enrich_memory(body.raw_text)
    embedding = await get_embedding(body.raw_text)

    tags = json.dumps(enriched.get("tags", []))
    data_types = json.dumps(enriched.get("data_types", []))
    extracted_entities = json.dumps(enriched.get("extracted_entities", {}))
    is_sensitive = int(bool(enriched.get("is_sensitive", False)))
    title = enriched.get("title", body.raw_text[:80])
    category = enriched.get("category", "General")

    with get_conn() as conn:
        conn.execute(
            """
            UPDATE memory SET
                raw_text = ?, title = ?, category = ?, tags = ?,
                data_types = ?, extracted_entities = ?, is_sensitive = ?,
                embedding = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (
                body.raw_text, title, category, tags,
                data_types, extracted_entities, is_sensitive,
                _emb_to_blob(embedding), mem_id,
            ),
        )

    return {
        "id": mem_id,
        "title": title,
        "category": category,
        "tags": enriched.get("tags", []),
        "data_types": enriched.get("data_types", []),
        "is_sensitive": bool(is_sensitive),
    }


# ---------------------------------------------------------------------------
# DELETE /api/memory/{id}
# ---------------------------------------------------------------------------

@api.delete("/memory/{mem_id}", status_code=204)
def delete_memory(mem_id: str):
    with get_conn() as conn:
        result = conn.execute(
            "DELETE FROM memory WHERE id = ?", (mem_id,)
        )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Memory not found")


# ---------------------------------------------------------------------------
# POST /api/recall
# ---------------------------------------------------------------------------

@api.post("/recall")
async def recall(body: RecallRequest):
    q_emb = np.array(await get_embedding(body.question), dtype=np.float32)

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, category, is_sensitive, embedding FROM memory WHERE embedding IS NOT NULL"
        ).fetchall()

    scored = []
    for row in rows:
        emb = _blob_to_emb(row["embedding"])
        score = _cosine(q_emb, emb)
        if score >= body.threshold:
            scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: body.top_k]

    if not top:
        return {
            "answer": "I couldn't find anything relevant in your second brain.",
            "sources": [],
        }

    top_ids = [r["id"] for _, r in top]
    placeholders = ",".join("?" * len(top_ids))
    with get_conn() as conn:
        detail_rows = conn.execute(
            f"SELECT id, raw_text FROM memory WHERE id IN ({placeholders})",
            top_ids,
        ).fetchall()

    id_to_text = {r["id"]: r["raw_text"] for r in detail_rows}

    context_parts = []
    for i, (score, row) in enumerate(top, 1):
        context_parts.append(f"[{i}] {row['title']}\n{id_to_text.get(row['id'], '')}")
    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a personal memory assistant. Answer using ONLY the memories provided. "
                "If the answer contains exact values like account numbers, API keys, amounts, "
                "passwords, or dates — reproduce them exactly as stored. Do not paraphrase."
            ),
        },
        {
            "role": "user",
            "content": f"Memories:\n{context}\n\nQuestion: {body.question}",
        },
    ]

    answer = await ollama_chat(messages)

    sources = [
        {
            "id": row["id"],
            "title": row["title"],
            "category": row["category"],
            "similarity": round(score, 4),
            "is_sensitive": bool(row["is_sensitive"]),
        }
        for score, row in top
    ]

    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# POST /api/search
# ---------------------------------------------------------------------------

@api.post("/search")
def search(body: SearchRequest):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT m.id, m.raw_text, m.title, m.category, m.tags,
                   m.data_types, m.is_sensitive, m.created_at, m.updated_at
            FROM memory_fts f
            JOIN memory m ON m.id = f.id
            WHERE memory_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (body.query, body.limit),
        ).fetchall()

    items = []
    for row in rows:
        d = _row_to_dict(row)
        d["tags"] = json.loads(d["tags"] or "[]")
        d["data_types"] = json.loads(d["data_types"] or "[]")
        d["is_sensitive"] = bool(d["is_sensitive"])
        items.append(d)

    return {"items": items}


# ---------------------------------------------------------------------------
# GET /api/stats
# ---------------------------------------------------------------------------

@api.get("/stats")
def stats():
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        sensitive = conn.execute(
            "SELECT COUNT(*) FROM memory WHERE is_sensitive = 1"
        ).fetchone()[0]
        by_category = conn.execute(
            "SELECT category, COUNT(*) as count FROM memory GROUP BY category ORDER BY count DESC"
        ).fetchall()
        recent_rows = conn.execute(
            """
            SELECT id, title, category, is_sensitive, created_at
            FROM memory ORDER BY created_at DESC LIMIT 5
            """
        ).fetchall()

    return {
        "total": total,
        "sensitive": sensitive,
        "by_category": [{"category": r["category"], "count": r["count"]} for r in by_category],
        "recent": [
            {
                "id": r["id"],
                "title": r["title"],
                "category": r["category"],
                "is_sensitive": bool(r["is_sensitive"]),
                "created_at": r["created_at"],
            }
            for r in recent_rows
        ],
    }


# ---------------------------------------------------------------------------
# GET /api/tags
# ---------------------------------------------------------------------------

@api.get("/tags")
def list_tags():
    with get_conn() as conn:
        rows = conn.execute("SELECT tags FROM memory WHERE tags IS NOT NULL").fetchall()

    counts: Counter = Counter()
    for row in rows:
        try:
            tags = json.loads(row["tags"] or "[]")
            for tag in tags:
                if isinstance(tag, str) and tag.strip():
                    counts[tag.strip()] += 1
        except (json.JSONDecodeError, TypeError):
            pass

    return [{"tag": tag, "count": count} for tag, count in counts.most_common()]


# ---------------------------------------------------------------------------
# GET /api/events/upcoming
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
    "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
    "%Y/%m/%d", "%d.%m.%Y",
]


def _try_parse_date(s: str) -> date | None:
    s = s.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


@api.get("/events/upcoming")
def upcoming_events(limit: int = Query(default=10, ge=1, le=50)):
    today = date.today()

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, title, category, extracted_entities, is_sensitive, raw_text
            FROM memory
            WHERE category = 'Event'
              AND extracted_entities IS NOT NULL
            ORDER BY created_at DESC
            """
        ).fetchall()

    events = []
    for row in rows:
        try:
            entities = json.loads(row["extracted_entities"] or "{}")
        except (json.JSONDecodeError, TypeError):
            entities = {}

        dates_raw = entities.get("dates", [])
        if not isinstance(dates_raw, list):
            dates_raw = [dates_raw] if dates_raw else []

        seen_dates: set[str] = set()
        for ds in dates_raw:
            if not isinstance(ds, str):
                continue
            parsed = _try_parse_date(ds)
            if parsed and parsed >= today and parsed.isoformat() not in seen_dates:
                seen_dates.add(parsed.isoformat())
                days_left = (parsed - today).days
                events.append({
                    "id": row["id"],
                    "title": row["title"] or row["raw_text"][:80],
                    "category": row["category"],
                    "date_str": ds,
                    "date_iso": parsed.isoformat(),
                    "days_left": days_left,
                    "is_sensitive": bool(row["is_sensitive"]),
                })

    events.sort(key=lambda x: x["days_left"])
    return {"events": events[:limit]}


# ---------------------------------------------------------------------------

app.include_router(api)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8765, reload=True)
