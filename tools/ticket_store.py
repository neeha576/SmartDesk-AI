"""Local SQLite record of tickets created by SmartDesk: employee email -> ticket keys.

Jira can't search issues by an external requester's email, so every ticket we create is also
saved here. The status-check flow looks up an employee's ticket keys here, then reads each
ticket's live status from Jira.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_key  TEXT PRIMARY KEY,
    email       TEXT NOT NULL,
    category    TEXT NOT NULL,
    summary     TEXT NOT NULL,
    priority    TEXT,
    url         TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_email ON tickets(email);
"""


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    path = Path(db_path or settings.TICKET_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def save_ticket(email: str, ticket_key: str, category: str, summary: str,
                priority: str | None, url: str, db_path: str | None = None) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ticket_key, email.lower(), category, summary, priority, url,
             datetime.now(timezone.utc).isoformat()))


def tickets_for_email(email: str, db_path: str | None = None) -> list[dict]:
    """All tickets SmartDesk created for this email, newest first."""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute("SELECT * FROM tickets WHERE email = ? ORDER BY created_at DESC",
                            (email.lower(),)).fetchall()
    return [dict(r) for r in rows]
