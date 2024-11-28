import json
import sqlite3
import uuid
from datetime import datetime
from typing import Any

import click
from rich.console import Console
from rich.markdown import Markdown

from core.chat import is_vision_llm, parse_image, unparse_image
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
        """, (session_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), title, json.dumps(history)))  # Save history as JSON
        conn.commit()
        return session_id


def get_chat_history(session_id: str = None):
    """Retrieve chat history."""
    try: 
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
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return None


def delete_chat_session(session_id:str=None):
    """Delete a chat session by ID."""
    
    if session_id.strip() == 'all':
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions")
            conn.commit()
        
    else:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
        
    return
    

def print_markdown(message: str, console: Console) -> None:
    """Render a message as Markdown or plain text."""
    try:
        md = Markdown(message)
        console.print(md)
    except Exception:
        console.print(message)
    console.print("\n")


def show_messages(messages: list[dict], console: Console, show_system_prompt: bool = False) -> None:
    """Display chat messages in the console."""
    for message in messages:
        role = message['role']
        
        if role == 'system' and not show_system_prompt:
            continue

        content = message['content']
        image_path = None
        if isinstance(content, list):
            content, image_path = unparse_image(content)



        if role == 'user':
            console.print("[blue]You 👦:[/blue]")
        elif role == 'assistant':
            console.print("[green]LLM 🤖:[/green]")
        elif role == 'system':
            console.print("[cyan]System 🤖:[/cyan]")
        else:
            console.print("[yellow]Unknown Role:[/yellow]")

        if image_path:
            console.print(f'[yellow]Image is saved here: file:///{image_path}[/yellow]')
            
        print_markdown(content, console)
