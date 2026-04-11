"""
SQLite-backed metadata store for document state tracking.
Tracks ingestion status, processing state, wiki output paths, and confidence scores.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


@dataclass
class DocumentRecord:
    id: str                          # SHA-256 hash of content
    path: str                        # Relative path in /raw
    status: str = "pending"          # pending|processing|done|failed|review
    created_at: str = ""
    updated_at: str = ""
    wiki_path: str = ""
    confidence: float = 0.0
    agent_outputs: dict[str, Any] = field(default_factory=dict)
    error_msg: str = ""
    file_size: int = 0
    mime_type: str = ""

    def __post_init__(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


class MetadataStore:
    """SQLite metadata store. Thread-safe via connection-per-call pattern."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS documents (
        id          TEXT PRIMARY KEY,
        path        TEXT NOT NULL UNIQUE,
        status      TEXT NOT NULL DEFAULT 'pending',
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL,
        wiki_path   TEXT DEFAULT '',
        confidence  REAL DEFAULT 0.0,
        agent_outputs TEXT DEFAULT '{}',
        error_msg   TEXT DEFAULT '',
        file_size   INTEGER DEFAULT 0,
        mime_type   TEXT DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_status ON documents(status);
    CREATE INDEX IF NOT EXISTS idx_path ON documents(path);
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)

    def upsert(self, record: DocumentRecord) -> None:
        record.updated_at = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO documents
                   (id, path, status, created_at, updated_at, wiki_path,
                    confidence, agent_outputs, error_msg, file_size, mime_type)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     status=excluded.status,
                     updated_at=excluded.updated_at,
                     wiki_path=excluded.wiki_path,
                     confidence=excluded.confidence,
                     agent_outputs=excluded.agent_outputs,
                     error_msg=excluded.error_msg
                """,
                (
                    record.id, record.path, record.status,
                    record.created_at, record.updated_at,
                    record.wiki_path, record.confidence,
                    json.dumps(record.agent_outputs),
                    record.error_msg, record.file_size, record.mime_type,
                ),
            )

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id=?", (doc_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_path(self, path: str) -> DocumentRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE path=?", (path,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_by_status(self, status: str) -> list[DocumentRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE status=? ORDER BY created_at", (status,)
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_all(self) -> list[DocumentRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def update_status(self, doc_id: str, status: str, error_msg: str = "") -> None:
        updated = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE documents SET status=?, updated_at=?, error_msg=? WHERE id=?",
                (status, updated, error_msg, doc_id),
            )

    def update_agent_output(
        self, doc_id: str, agent: str, output: dict[str, Any], confidence: float
    ) -> None:
        updated = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT agent_outputs FROM documents WHERE id=?", (doc_id,)
            ).fetchone()
            if row:
                outputs = json.loads(row["agent_outputs"] or "{}")
                outputs[agent] = output
                conn.execute(
                    "UPDATE documents SET agent_outputs=?, confidence=?, updated_at=? WHERE id=?",
                    (json.dumps(outputs), confidence, updated, doc_id),
                )

    def stats(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as n FROM documents GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            path=row["path"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            wiki_path=row["wiki_path"] or "",
            confidence=row["confidence"] or 0.0,
            agent_outputs=json.loads(row["agent_outputs"] or "{}"),
            error_msg=row["error_msg"] or "",
            file_size=row["file_size"] or 0,
            mime_type=row["mime_type"] or "",
        )
