"""
database.py
-----------
Lightweight SQLite persistence layer for the Analysis History feature.
Uses plain sqlite3 (no ORM) to keep the project easy to explain in a
viva.
"""

import sqlite3
from datetime import datetime

DB_PATH = "history.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_text TEXT NOT NULL,
            source TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            credibility_score REAL NOT NULL,
            suspicious_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_analysis(news_text, source, prediction, confidence, credibility_score, suspicious_count):
    conn = get_connection()
    conn.execute(
        """INSERT INTO analyses
           (news_text, source, prediction, confidence, credibility_score, suspicious_count, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            news_text.strip(),
            source.strip() if source else "",
            prediction,
            confidence,
            credibility_score,
            suspicious_count,
            datetime.utcnow().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    new_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    conn.close()
    return new_id


def get_history(limit=50):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_analysis_by_id(analysis_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_stats():
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) AS c FROM analyses").fetchone()["c"]
    fake = conn.execute("SELECT COUNT(*) AS c FROM analyses WHERE prediction = 'FAKE'").fetchone()["c"]
    real = total - fake
    conn.close()
    return {"total": total, "fake": fake, "real": real}


def clear_history():
    conn = get_connection()
    conn.execute("DELETE FROM analyses")
    conn.commit()
    conn.close()
