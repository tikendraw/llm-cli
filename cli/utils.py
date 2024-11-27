import sqlite3
from datetime import datetime
from typing import Any
from core.config import history_db

# Database setup
DB_PATH = history_db

def init_db():
    """Initialize the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                chat_history TEXT NOT NULL
            )
        """)
        conn.commit()


def save_chat_history(history:list[dict])->str:
    """Save chat history to the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sessions (start_time, chat_history) VALUES (?, ?)", 
                       (datetime.now().isoformat(), history))
        conn.commit()
        return cursor.lastrowid

def get_chat_history(session_id:str=None)-> Any:
    """Retrieve chat history."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        if session_id:
            cursor.execute("SELECT id, start_time, chat_history FROM sessions WHERE id = ?", (session_id,))
            return cursor.fetchone()
        else:
            cursor.execute("SELECT id, start_time, LENGTH(chat_history) FROM sessions")
            return cursor.fetchall()
