import sqlite3
import uuid
from datetime import datetime
from typing import Any
import json

from core.config import history_db

# Database setup
DB_PATH = history_db

def init_db():
    """Initialize the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                title TEXT NOT NULL,
                chat_history TEXT NOT NULL
            )
        """)
        conn.commit()


def save_chat_history(history: list[dict], session_id: str = None, title: str = "Untitled Session") -> str:
    """Save chat history to the database."""
    session_id = session_id or str(uuid.uuid4())
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO sessions (id, start_time, title, chat_history) 
            VALUES (?, ?, ?, ?)
        """, (session_id, datetime.now().isoformat(), title, json.dumps(history)))  # Save history as JSON
        conn.commit()
        return session_id

def get_chat_history(session_id: str = None):
    """Retrieve chat history."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        if session_id:
            cursor.execute("SELECT id, start_time, title, chat_history FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                row = row[:-1] + (json.loads(row[-1]),)  
            return row
        else:
            cursor.execute("SELECT id, start_time, title, chat_history FROM sessions")
            rows = cursor.fetchall()
            return [(row[0], row[1], row[2], len(json.loads(row[3]))) for row in rows]
