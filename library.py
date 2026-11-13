"""Persistent recording catalog, FTS5 index, and summary vectors (SQLite)."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime

import numpy as np

import config
import embeddings
import rules_loader


@contextmanager
def connection():
    """Open a SQLite connection that commits on success and always closes."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(config.LIBRARY_DB, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        with db:
            yield db
    finally:
        db.close()


def initialize():
    """Create the schema and audio folder if missing; safe to call repeatedly."""
    config.LIBRARY_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with connection() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS recordings (
                id TEXT PRIMARY KEY, file TEXT NOT NULL, duration REAL NOT NULL,
                stored_audio_path TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL, language TEXT NOT NULL DEFAULT '',
                transcript TEXT NOT NULL DEFAULT '', summary TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '', flags TEXT NOT NULL DEFAULT '[]',
                analysis_version TEXT NOT NULL, analysis_error INTEGER NOT NULL DEFAULT 0,
                index_state TEXT NOT NULL DEFAULT 'pending', index_error TEXT NOT NULL DEFAULT '',
                embedding_model TEXT, embedding_dimensions INTEGER, embedding BLOB,
                embedding_text_hash TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS recording_categories (
                recording_id TEXT NOT NULL, category_id INTEGER NOT NULL,
                PRIMARY KEY (recording_id, category_id)
            );
            CREATE INDEX IF NOT EXISTS category_lookup ON recording_categories(category_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS recording_fts USING fts5(
                recording_id UNINDEXED, file, summary, transcript, tokenize='unicode61'
            );
        """)
        # Additive migration for libraries created before the path metadata field.
        columns = {row[1] for row in db.execute("PRAGMA table_info(recordings)")}
        if "stored_audio_path" not in columns:
            db.execute(
                "ALTER TABLE recordings ADD COLUMN stored_audio_path TEXT NOT NULL DEFAULT ''"
            )
        db.execute(
            "UPDATE recordings SET stored_audio_path='audio/' || id || '.wav' WHERE stored_audio_path=''"
        )


def valid_id(value):
    """Recording IDs are lowercase SHA-256 hex digests of the audio bytes."""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def audio_path(recording_id):
    """Managed WAV path for an ID; never built from user-supplied file names."""
    if not valid_id(recording_id):
        raise ValueError("Invalid recording ID")
    return config.LIBRARY_AUDIO_DIR / f"{recording_id}.wav"


def _record(row):
    """Database row -> API record (flags decoded, vector omitted, audio URL added)."""
    if row is None:
        return None
    result = dict(row)
    result.pop("embedding", None)
    result["triggered_rules"] = json.loads(result.pop("flags"))
    result["categories"] = [r["id"] for r in result["triggered_rules"]]
    result["audio_url"] = f"/api/recordings/{result['id']}/audio"
    result["content_hash"] = result["id"]
    result["analysis_error"] = bool(result["analysis_error"])
    return result


def get(recording_id):
    """One recording by ID, or None for an unknown or malformed ID."""
    if not valid_id(recording_id):
        return None
    with connection() as db:
        return _record(
            db.execute("SELECT * FROM recordings WHERE id=?", (recording_id,)).fetchone()
        )


def get_many(recording_ids):
    """Several recordings in one query, keyed by ID (unknown IDs are absent)."""
    ids = list(dict.fromkeys(rid for rid in recording_ids if valid_id(rid)))
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    with connection() as db:
        rows = db.execute(f"SELECT * FROM recordings WHERE id IN ({placeholders})", ids).fetchall()
    return {row["id"]: _record(row) for row in rows}


def save(result):
    """Replace one analysis and its FTS/category entries atomically; invalidate its vector."""
    now = datetime.now(UTC).isoformat()
    with connection() as db:
        db.execute(
            """
            INSERT INTO recordings
            (id,file,duration,status,language,transcript,summary,reason,flags,
             analysis_version,analysis_error,index_state,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                file=excluded.file,duration=excluded.duration,status=excluded.status,
                language=excluded.language,transcript=excluded.transcript,summary=excluded.summary,
                reason=excluded.reason,flags=excluded.flags,analysis_version=excluded.analysis_version,
                analysis_error=excluded.analysis_error,index_state=excluded.index_state,
                index_error='',embedding=NULL,embedding_model=NULL,embedding_dimensions=NULL,
                embedding_text_hash=NULL,updated_at=excluded.updated_at
        """,
            (
                result["id"],
                result["file"],
                result["duration"],
                result["status"],
                result.get("language", ""),
                result.get("transcript", ""),
                result.get("summary", ""),
                result.get("reason", ""),
                json.dumps(result.get("triggered_rules", []), ensure_ascii=False),
                result["analysis_version"],
                int(result.get("analysis_error", False)),
                "excluded" if result["status"] == "skipped" else "pending",
                now,
                now,
            ),
        )
        db.execute("DELETE FROM recording_categories WHERE recording_id=?", (result["id"],))
        db.execute(
            "UPDATE recordings SET stored_audio_path=? WHERE id=?",
            (f"audio/{result['id']}.wav", result["id"]),
        )
        db.executemany(
            "INSERT INTO recording_categories VALUES (?,?)",
            [(result["id"], r["id"]) for r in result.get("triggered_rules", [])],
        )
        db.execute("DELETE FROM recording_fts WHERE recording_id=?", (result["id"],))
        if result["status"] != "skipped":
            db.execute(
                "INSERT INTO recording_fts VALUES (?,?,?,?)",
                (
                    result["id"],
                    result["file"],
                    result.get("summary", ""),
                    result.get("transcript", ""),
                ),
            )


def summary_hash(text):
    """Hash of the embedded summary, used to detect stale vectors."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_embedding(recording_id, summary, vector):
    """Store a summary vector, unless the summary changed since it was embedded."""
    data = np.asarray(vector, dtype="<f4")
    if not embeddings.is_valid(data):
        raise ValueError("Invalid embedding vector")
    with connection() as db:
        # An older in-flight indexing request must not overwrite a newer analysis.
        db.execute(
            """UPDATE recordings SET embedding=?,embedding_model=?,embedding_dimensions=?,
                   embedding_text_hash=?,index_state='ready',index_error=''
                   WHERE id=? AND summary=? AND status!='skipped'""",
            (
                data.tobytes(),
                config.EMBEDDING_MODEL,
                config.EMBEDDING_DIMENSIONS,
                summary_hash(summary),
                recording_id,
                summary,
            ),
        )


def index_failed(recording_id, message):
    """Mark a recording's vector as pending, keeping the error for a later retry."""
    with connection() as db:
        db.execute(
            "UPDATE recordings SET index_state='pending',index_error=?,embedding=NULL WHERE id=?",
            (message, recording_id),
        )


def _where(status=None, categories=None, searchable=False):
    """SQL filter clause and parameters; categories match as a union."""
    clauses, args = [], []
    if searchable:
        clauses.append("r.status!='skipped'")
    if status:
        clauses.append("r.status=?")
        args.append(status)
    if categories:
        placeholders = ",".join("?" for _ in categories)
        clauses.append(
            f"EXISTS (SELECT 1 FROM recording_categories c WHERE c.recording_id=r.id AND c.category_id IN ({placeholders}))"
        )
        args.extend(categories)
    return " AND ".join(clauses) or "1=1", args


def list_recordings(status=None, categories=None, page=1, page_size=24):
    """One page of recordings for the library view, with the total count."""
    where, args = _where(status, categories)
    with connection() as db:
        total = db.execute(f"SELECT COUNT(*) FROM recordings r WHERE {where}", args).fetchone()[0]
        rows = db.execute(
            f"SELECT r.* FROM recordings r WHERE {where} ORDER BY r.file,r.id LIMIT ? OFFSET ?",
            [*args, page_size, (page - 1) * page_size],
        ).fetchall()
    return {
        "recordings": [_record(r) for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


def searchable_records(status=None, categories=None):
    """Full records of every searchable (not skipped) recording matching the filters."""
    where, args = _where(status, categories, searchable=True)
    with connection() as db:
        rows = db.execute(
            f"SELECT r.* FROM recordings r WHERE {where} ORDER BY r.id", args
        ).fetchall()
    return [_record(r) for r in rows]


def searchable_ids(status=None, categories=None):
    """IDs only, for membership checks that do not need transcripts."""
    where, args = _where(status, categories, searchable=True)
    with connection() as db:
        return {row[0] for row in db.execute(f"SELECT r.id FROM recordings r WHERE {where}", args)}


def vectors(status=None, categories=None):
    """(id, vector) pairs embedded with the current model and dimensions."""
    where, args = _where(status, categories, searchable=True)
    with connection() as db:
        rows = db.execute(
            f"""SELECT r.id,r.embedding FROM recordings r WHERE {where}
            AND r.index_state='ready' AND r.embedding_model=? AND r.embedding_dimensions=? ORDER BY r.id""",
            [*args, config.EMBEDDING_MODEL, config.EMBEDDING_DIMENSIONS],
        ).fetchall()
    return [(r["id"], np.frombuffer(r["embedding"], dtype="<f4")) for r in rows]


def keywords(query, status=None, categories=None, limit=20):
    """BM25 full-text matches over file name, summary and transcript."""
    # Raw user text never becomes FTS syntax. OR supports natural language queries.
    tokens = list(dict.fromkeys(re.findall(r"\w+", query, re.UNICODE)))[:48]
    if not tokens:
        return []
    expression = " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)
    where, args = _where(status, categories, searchable=True)
    with connection() as db:
        rows = db.execute(
            f"""SELECT r.id,bm25(recording_fts,0,2,2,1) AS score
            FROM recording_fts JOIN recordings r ON r.id=recording_fts.recording_id
            WHERE recording_fts MATCH ? AND {where} ORDER BY score,r.id LIMIT ?""",
            [expression, *args, limit],
        ).fetchall()
    return [(r["id"], r["score"]) for r in rows]


def category_stats():
    """Category definitions with member counts, plus status and index totals."""
    with connection() as db:
        counts = dict(
            db.execute(
                "SELECT category_id,COUNT(DISTINCT recording_id) FROM recording_categories GROUP BY category_id"
            )
        )
        statuses = dict(db.execute("SELECT status,COUNT(*) FROM recordings GROUP BY status"))
        ready = db.execute(
            """SELECT COUNT(*) FROM recordings WHERE index_state='ready'
            AND status!='skipped' AND embedding_model=? AND embedding_dimensions=?""",
            (config.EMBEDDING_MODEL, config.EMBEDDING_DIMENSIONS),
        ).fetchone()[0]
    return {
        "categories": [
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "source": r.source,
                "fallback": r.fallback,
                "count": counts.get(r.id, 0),
            }
            for r in rules_loader.get_active().rules
        ],
        "statuses": statuses,
        "total": sum(statuses.values()),
        "searchable": ready,
    }
