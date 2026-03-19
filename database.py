import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "secondbrain.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory (
                id                  TEXT PRIMARY KEY,
                raw_text            TEXT NOT NULL,
                title               TEXT,
                category            TEXT DEFAULT 'General',
                tags                TEXT DEFAULT '[]',
                data_types          TEXT DEFAULT '[]',
                extracted_entities  TEXT DEFAULT '{}',
                is_sensitive        INTEGER DEFAULT 0,
                embedding           BLOB,
                created_at          TEXT DEFAULT (datetime('now')),
                updated_at          TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                id UNINDEXED, raw_text, title, tags,
                content='memory', content_rowid='rowid'
            );

            CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
                INSERT INTO memory_fts(rowid, id, raw_text, title, tags)
                VALUES (new.rowid, new.id, new.raw_text, new.title, new.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, id, raw_text, title, tags)
                VALUES ('delete', old.rowid, old.id, old.raw_text, old.title, old.tags);
                INSERT INTO memory_fts(rowid, id, raw_text, title, tags)
                VALUES (new.rowid, new.id, new.raw_text, new.title, new.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, id, raw_text, title, tags)
                VALUES ('delete', old.rowid, old.id, old.raw_text, old.title, old.tags);
            END;
        """)
