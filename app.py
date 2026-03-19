import asyncio
import json
import re
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import markdown as md_lib

import httpx
import numpy as np
from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from auth import AUTH_ENABLED, _init_token_table, login, revoke_token, verify_token
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


def _render_md(text: str) -> str:
    """Render markdown to HTML (safe subset — no raw HTML pass-through)."""
    return md_lib.markdown(text, extensions=["nl2br", "tables", "fenced_code"])


def _fts_query(text: str) -> str | None:
    """Convert free text to a safe FTS5 OR query over individual words."""
    words = re.findall(r'\b\w{2,}\b', text)
    if not words:
        return None
    return " OR ".join(f'"{w}"' for w in words[:20])


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
    expires_at: Optional[str] = None  # ISO datetime string, e.g. "2026-03-20T10:00:00"


class MemoryUpdate(BaseModel):
    raw_text: str
    expires_at: Optional[str] = None


class RecallRequest(BaseModel):
    question: str
    top_k: int = 5
    threshold: float = 0.25


class SearchRequest(BaseModel):
    query: str
    limit: int = 20


class LoginRequest(BaseModel):
    password: str


class BulkDeleteRequest(BaseModel):
    ids: list[str]


class ImportRequest(BaseModel):
    memories: list[dict]


# ---------------------------------------------------------------------------
# App & routers
# ---------------------------------------------------------------------------

async def _auto_archive_loop():
    """Background task: archive expired memories every 60 seconds."""
    while True:
        try:
            with get_conn() as conn:
                conn.execute(
                    """
                    UPDATE memory SET is_archived = 1
                    WHERE expires_at IS NOT NULL
                      AND expires_at <= datetime('now')
                      AND is_archived = 0
                    """
                )
        except Exception:
            pass
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _init_token_table()
    task = asyncio.create_task(_auto_archive_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "An internal error occurred."})


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

    rendered = _render_md(body.raw_text)

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO memory
                (id, raw_text, rendered_text, title, category, tags, data_types,
                 extracted_entities, is_sensitive, embedding, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mem_id, body.raw_text, rendered, title, category,
                tags, data_types, extracted_entities,
                is_sensitive, _emb_to_blob(embedding),
                body.expires_at,
            ),
        )

    return {
        "id": mem_id,
        "title": title,
        "category": category,
        "tags": enriched.get("tags", []),
        "data_types": enriched.get("data_types", []),
        "is_sensitive": bool(is_sensitive),
        "expires_at": body.expires_at,
    }


# ---------------------------------------------------------------------------
# POST /api/memory/check-duplicate
# ---------------------------------------------------------------------------

@api.post("/memory/check-duplicate")
async def check_duplicate(body: MemoryCreate, threshold: float = 0.75):
    q_emb = np.array(await get_embedding(body.raw_text), dtype=np.float32)

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, category, is_sensitive, embedding FROM memory WHERE embedding IS NOT NULL"
        ).fetchall()

    matches = []
    for row in rows:
        score = _cosine(q_emb, _blob_to_emb(row["embedding"]))
        if score >= threshold:
            matches.append({
                "id": row["id"],
                "title": row["title"],
                "category": row["category"],
                "similarity": round(score, 4),
                "is_sensitive": bool(row["is_sensitive"]),
            })

    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return {"duplicates": matches[:5]}


# ---------------------------------------------------------------------------
# GET /api/memory
# ---------------------------------------------------------------------------

@api.get("/memory")
def list_memories(
    category: Optional[str] = None,
    tag: Optional[str] = None,
    sensitive: Optional[bool] = None,
    archived: Optional[bool] = None,
    sort: Optional[str] = Query(default=None, pattern="^(recent|accessed|popular)$"),
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
    # By default hide archived; pass archived=true to show only archived
    if archived is True:
        clauses.append("is_archived = 1")
    else:
        clauses.append("is_archived = 0")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    if sort == "accessed":
        order = "is_pinned DESC, last_accessed_at DESC NULLS LAST"
    elif sort == "popular":
        order = "is_pinned DESC, access_count DESC"
    else:
        order = "is_pinned DESC, created_at DESC"

    with get_conn() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM memory {where}", params
        ).fetchone()[0]

        rows = conn.execute(
            f"""
            SELECT id, raw_text, title, category, tags, data_types,
                   is_sensitive, is_pinned, created_at, updated_at,
                   last_accessed_at, access_count, expires_at, is_archived
            FROM memory {where}
            ORDER BY {order}
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
        d["is_pinned"] = bool(d["is_pinned"])
        d["is_archived"] = bool(d["is_archived"])
        d["access_count"] = d.get("access_count") or 0
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
            SELECT id, raw_text, rendered_text, title, category, tags, data_types,
                   extracted_entities, is_sensitive, is_pinned, created_at, updated_at,
                   last_accessed_at, access_count, expires_at, is_archived
            FROM memory WHERE id = ?
            """,
            (mem_id,),
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Memory not found")

        conn.execute(
            """
            UPDATE memory
            SET access_count = COALESCE(access_count, 0) + 1,
                last_accessed_at = datetime('now')
            WHERE id = ?
            """,
            (mem_id,),
        )

    d = _row_to_dict(row)
    d["tags"] = json.loads(d["tags"] or "[]")
    d["data_types"] = json.loads(d["data_types"] or "[]")
    d["extracted_entities"] = json.loads(d["extracted_entities"] or "{}")
    d["is_sensitive"] = bool(d["is_sensitive"])
    d["is_pinned"] = bool(d["is_pinned"])
    d["is_archived"] = bool(d["is_archived"])
    d["access_count"] = d.get("access_count") or 0
    return d


# ---------------------------------------------------------------------------
# POST /api/memory/{id}/archive  — toggle archive
# ---------------------------------------------------------------------------

@api.post("/memory/{mem_id}/archive")
def toggle_archive(mem_id: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT is_archived FROM memory WHERE id = ?", (mem_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Memory not found")
        new_val = 0 if row["is_archived"] else 1
        conn.execute(
            "UPDATE memory SET is_archived = ? WHERE id = ?", (new_val, mem_id)
        )
    return {"is_archived": bool(new_val)}


# ---------------------------------------------------------------------------
# PUT /api/memory/{id}
# ---------------------------------------------------------------------------

@api.put("/memory/{mem_id}")
async def update_memory(mem_id: str, body: MemoryUpdate):
    with get_conn() as conn:
        old = conn.execute(
            "SELECT raw_text, title FROM memory WHERE id = ?", (mem_id,)
        ).fetchone()

    if not old:
        raise HTTPException(status_code=404, detail="Memory not found")

    enriched = await enrich_memory(body.raw_text)
    embedding = await get_embedding(body.raw_text)
    rendered = _render_md(body.raw_text)

    tags = json.dumps(enriched.get("tags", []))
    data_types = json.dumps(enriched.get("data_types", []))
    extracted_entities = json.dumps(enriched.get("extracted_entities", {}))
    is_sensitive = int(bool(enriched.get("is_sensitive", False)))
    title = enriched.get("title", body.raw_text[:80])
    category = enriched.get("category", "General")

    with get_conn() as conn:
        # Snapshot the previous version before overwriting
        conn.execute(
            "INSERT INTO memory_history (memory_id, raw_text, title) VALUES (?, ?, ?)",
            (mem_id, old["raw_text"], old["title"]),
        )
        conn.execute(
            """
            UPDATE memory SET
                raw_text = ?, rendered_text = ?, title = ?, category = ?, tags = ?,
                data_types = ?, extracted_entities = ?, is_sensitive = ?,
                embedding = ?, updated_at = datetime('now'),
                expires_at = ?, is_archived = 0
            WHERE id = ?
            """,
            (
                body.raw_text, rendered, title, category, tags,
                data_types, extracted_entities, is_sensitive,
                _emb_to_blob(embedding), body.expires_at, mem_id,
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

    # --- Semantic scores ---
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, category, is_sensitive, is_pinned, embedding FROM memory WHERE embedding IS NOT NULL AND is_archived = 0"
        ).fetchall()

    row_map: dict = {}
    sem_scores: dict[str, float] = {}
    for row in rows:
        emb = _blob_to_emb(row["embedding"])
        sem_scores[row["id"]] = _cosine(q_emb, emb)
        row_map[row["id"]] = row

    # --- FTS scores (best-effort; silently ignored on parse errors) ---
    fts_scores: dict[str, float] = {}
    fts_q = _fts_query(body.question)
    if fts_q:
        try:
            with get_conn() as conn:
                fts_rows = conn.execute(
                    """
                    SELECT m.id, memory_fts.rank
                    FROM memory_fts
                    JOIN memory m ON m.id = memory_fts.id
                    WHERE memory_fts MATCH ? AND m.is_archived = 0
                    ORDER BY memory_fts.rank
                    LIMIT 50
                    """,
                    (fts_q,),
                ).fetchall()
            if fts_rows:
                ranks = [abs(r["rank"]) for r in fts_rows]
                max_r = max(ranks) or 1.0
                for r in fts_rows:
                    fts_scores[r["id"]] = abs(r["rank"]) / max_r
                # Fetch metadata for FTS-only hits (no embedding)
                fts_only = set(fts_scores) - set(row_map)
                if fts_only:
                    ph = ",".join("?" * len(fts_only))
                    with get_conn() as conn:
                        for r in conn.execute(
                            f"SELECT id, title, category, is_sensitive, is_pinned FROM memory WHERE id IN ({ph})",
                            list(fts_only),
                        ).fetchall():
                            row_map[r["id"]] = r
        except Exception:
            pass  # fall back to semantic-only

    # --- Hybrid merge (α=0.7 semantic, 0.3 FTS) ---
    alpha = 0.7
    scored = []
    for mid in set(sem_scores) | set(fts_scores):
        if mid not in row_map:
            continue
        hybrid = alpha * sem_scores.get(mid, 0.0) + (1 - alpha) * fts_scores.get(mid, 0.0)
        if hybrid >= body.threshold:
            scored.append((hybrid, row_map[mid]))

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
        conn.execute(
            f"""
            UPDATE memory
            SET access_count = COALESCE(access_count, 0) + 1,
                last_accessed_at = datetime('now')
            WHERE id IN ({placeholders})
            """,
            top_ids,
        )

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
            "is_pinned": bool(row["is_pinned"]) if "is_pinned" in row.keys() else False,
        }
        for score, row in top
    ]

    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# POST /api/memory/{id}/pin  — toggle pin
# ---------------------------------------------------------------------------

@api.post("/memory/{mem_id}/pin")
def toggle_pin(mem_id: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT is_pinned FROM memory WHERE id = ?", (mem_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    new_val = 0 if row["is_pinned"] else 1
    with get_conn() as conn:
        conn.execute("UPDATE memory SET is_pinned = ? WHERE id = ?", (new_val, mem_id))
    return {"is_pinned": bool(new_val)}


# ---------------------------------------------------------------------------
# POST /api/search
# ---------------------------------------------------------------------------

@api.post("/search")
def search(body: SearchRequest):
    fts_q = _fts_query(body.query)
    if not fts_q:
        return {"items": []}

    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT m.id, m.raw_text, m.title, m.category, m.tags,
                       m.data_types, m.is_sensitive, m.created_at, m.updated_at
                FROM memory_fts
                JOIN memory m ON m.id = memory_fts.id
                WHERE memory_fts MATCH ? AND m.is_archived = 0
                ORDER BY rank
                LIMIT ?
                """,
                (fts_q, body.limit),
            ).fetchall()
    except Exception:
        return {"items": []}

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
# POST /api/upload  — extract text from PDF / image / plain text
# ---------------------------------------------------------------------------

_PDF_EXTS   = {"pdf"}
_IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "tiff", "tif", "bmp", "gif"}
_TEXT_EXTS  = {"txt", "md", "csv", "log"}
_MAX_BYTES  = 20 * 1024 * 1024  # 20 MB


def _extract_pdf(content: bytes) -> str:
    try:
        from io import BytesIO
        from pdfminer.high_level import extract_text
        text = extract_text(BytesIO(content))
        return (text or "").strip()
    except ImportError:
        raise HTTPException(status_code=501, detail="PDF support unavailable. Run: pip install pdfminer.six")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {e}")


def _extract_image(content: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
        from io import BytesIO
        import os
        _TESS_CANDIDATES = [
            "/opt/homebrew/bin/tesseract",   # macOS Apple Silicon (Homebrew)
            "/usr/local/bin/tesseract",       # macOS Intel (Homebrew)
            "/usr/bin/tesseract",             # Linux
        ]
        for _p in _TESS_CANDIDATES:
            if os.path.exists(_p):
                pytesseract.pytesseract.tesseract_cmd = _p
                break
        img = Image.open(BytesIO(content))
        text = pytesseract.image_to_string(img)
        return (text or "").strip()
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="OCR unavailable. Install dependencies: pip install pytesseract Pillow  |  brew install tesseract",
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read image: {e}")


@api.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB)")

    filename = (file.filename or "").strip()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext in _PDF_EXTS:
        text = _extract_pdf(content)
    elif ext in _IMAGE_EXTS:
        text = _extract_image(content)
    elif ext in _TEXT_EXTS:
        text = content.decode("utf-8", errors="replace").strip()
    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '.{ext}'. Supported: pdf, png, jpg, jpeg, webp, tiff, bmp, txt, md, csv",
        )

    if not text:
        raise HTTPException(status_code=422, detail="No text could be extracted from this file.")

    return {"text": text, "filename": filename, "chars": len(text)}


# ---------------------------------------------------------------------------
# DELETE /api/memory  — bulk delete
# ---------------------------------------------------------------------------

@api.delete("/memory", status_code=200)
def bulk_delete(body: BulkDeleteRequest):
    if not body.ids:
        return {"deleted": 0}
    ph = ",".join("?" * len(body.ids))
    with get_conn() as conn:
        result = conn.execute(f"DELETE FROM memory WHERE id IN ({ph})", body.ids)
    return {"deleted": result.rowcount}


# ---------------------------------------------------------------------------
# GET /api/memory/{id}/history
# ---------------------------------------------------------------------------

@api.get("/memory/{mem_id}/history")
def get_history(mem_id: str):
    with get_conn() as conn:
        exists = conn.execute("SELECT 1 FROM memory WHERE id = ?", (mem_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="Memory not found")
        rows = conn.execute(
            """
            SELECT id, raw_text, title, saved_at
            FROM memory_history WHERE memory_id = ?
            ORDER BY saved_at DESC
            LIMIT 20
            """,
            (mem_id,),
        ).fetchall()
    return {"history": [_row_to_dict(r) for r in rows]}


# ---------------------------------------------------------------------------
# GET /api/export
# ---------------------------------------------------------------------------

@api.get("/export")
def export_memories(format: str = Query(default="json", pattern="^(json|markdown)$")):
    from fastapi.responses import Response

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, raw_text, title, category, tags, data_types,
                   extracted_entities, is_sensitive, is_pinned,
                   expires_at, is_archived, access_count,
                   created_at, updated_at
            FROM memory
            ORDER BY created_at ASC
            """
        ).fetchall()

    memories = []
    for row in rows:
        d = _row_to_dict(row)
        d["tags"] = json.loads(d["tags"] or "[]")
        d["data_types"] = json.loads(d["data_types"] or "[]")
        d["extracted_entities"] = json.loads(d["extracted_entities"] or "{}")
        d["is_sensitive"] = bool(d["is_sensitive"])
        d["is_pinned"] = bool(d["is_pinned"])
        d["is_archived"] = bool(d["is_archived"])
        memories.append(d)

    if format == "markdown":
        lines = ["# SecondBrain Export\n"]
        for m in memories:
            lines.append(f"## {m['title'] or 'Untitled'}")
            lines.append(f"**Category:** {m['category']}  ")
            if m["tags"]:
                lines.append(f"**Tags:** {', '.join(m['tags'])}  ")
            lines.append(f"**Created:** {m['created_at']}  ")
            if m["expires_at"]:
                lines.append(f"**Expires:** {m['expires_at']}  ")
            lines.append("")
            lines.append(m["raw_text"])
            lines.append("\n---\n")
        content = "\n".join(lines)
        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": "attachment; filename=secondbrain_export.md"},
        )

    content = json.dumps({"memories": memories}, indent=2, ensure_ascii=False)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=secondbrain_export.json"},
    )


# ---------------------------------------------------------------------------
# POST /api/import
# ---------------------------------------------------------------------------

_IMPORT_CONCURRENCY = 5   # max simultaneous embedding calls
_IMPORT_ITEM_TIMEOUT = 30  # seconds per embedding call


async def _import_one(item: dict, sem: asyncio.Semaphore) -> bool:
    """Embed and insert a single import item. Returns True on success."""
    try:
        raw_text = str(item.get("raw_text", "")).strip()
        if not raw_text:
            return False

        mem_id = str(uuid.uuid4())
        title = item.get("title") or raw_text[:80]
        category = item.get("category") or "General"
        tags = json.dumps(item.get("tags") or [])
        data_types = json.dumps(item.get("data_types") or [])
        extracted_entities = json.dumps(item.get("extracted_entities") or {})
        is_sensitive = int(bool(item.get("is_sensitive", False)))
        is_pinned = int(bool(item.get("is_pinned", False)))
        expires_at = item.get("expires_at")
        created_at = item.get("created_at")
        rendered = _render_md(raw_text)

        async with sem:
            embedding = await asyncio.wait_for(
                get_embedding(raw_text), timeout=_IMPORT_ITEM_TIMEOUT
            )

        cols = "(id, raw_text, rendered_text, title, category, tags, data_types, extracted_entities, is_sensitive, is_pinned, embedding, expires_at"
        vals = "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
        params: list = [mem_id, raw_text, rendered, title, category,
                        tags, data_types, extracted_entities,
                        is_sensitive, is_pinned, _emb_to_blob(embedding), expires_at]
        if created_at:
            cols += ", created_at)"
            vals += ", ?)"
            params.append(created_at)
        else:
            cols += ")"
            vals += ")"

        with get_conn() as conn:
            conn.execute(f"INSERT INTO memory {cols} {vals}", params)
        return True
    except Exception:
        return False


@api.post("/import")
async def import_memories(body: ImportRequest):
    items = body.memories[:500]  # hard cap
    sem = asyncio.Semaphore(_IMPORT_CONCURRENCY)
    results = await asyncio.gather(*(_import_one(item, sem) for item in items))
    imported = sum(1 for r in results if r)
    return {"imported": imported, "failed": len(results) - imported}


# ---------------------------------------------------------------------------

app.include_router(api)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8765, reload=True)
