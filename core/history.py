import sqlite3
import json
import uuid
import datetime
from pathlib import Path

DB_PATH = "chat_history.db"

class ChatHistoryManager:
    def __init__(self, db_path: str = "data/chat_history.db"):
        self.db_path = db_path
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT,
                updated_at TEXT,
                pinned INTEGER DEFAULT 0
            )
            ''')
            
            # Add pinned column if it doesn't exist (migration for existing DBs)
            try:
                cursor.execute('ALTER TABLE sessions ADD COLUMN pinned INTEGER DEFAULT 0')
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Messages table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            ''')
            
            conn.commit()
        finally:
            conn.close()

    def create_session(self, title="New Chat"):
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        now = datetime.datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO sessions (id, title, created_at, updated_at, pinned) VALUES (?, ?, ?, ?, ?)',
                (session_id, title, now, now, 0)
            )
            conn.commit()
        finally:
            conn.close()
        return session_id

    def update_session_title(self, session_id, title):
        """Update the title of a session."""
        now = datetime.datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?',
                (title, now, session_id)
            )
            conn.commit()
        finally:
            conn.close()

    def toggle_pin(self, session_id):
        """Toggle the pinned status of a session. Returns the new pinned state."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Get current pinned state
            cursor.execute('SELECT pinned FROM sessions WHERE id = ?', (session_id,))
            row = cursor.fetchone()
            current_pinned = row[0] if row else 0
            new_pinned = 0 if current_pinned else 1
            
            # Update
            cursor.execute('UPDATE sessions SET pinned = ? WHERE id = ?', (new_pinned, session_id))
            conn.commit()
            return bool(new_pinned)
        finally:
            conn.close()

    def add_message(self, session_id, role, content):
        """Add a message to a session."""
        now = datetime.datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)',
                (session_id, role, content, now)
            )
            # Update session timestamp
            cursor.execute(
                'UPDATE sessions SET updated_at = ? WHERE id = ?',
                (now, session_id)
            )
            conn.commit()
        finally:
            conn.close()

    def get_sessions(self):
        """Get all sessions, ordered by pinned first, then most recent update."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT id, title, created_at, pinned FROM sessions ORDER BY pinned DESC, updated_at DESC')
            sessions = [
                {'id': row[0], 'title': row[1], 'created_at': row[2], 'pinned': bool(row[3])}
                for row in cursor.fetchall()
            ]
            return sessions
        finally:
            conn.close()

    def get_messages(self, session_id):
        """Get all messages for a session."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC',
                (session_id,)
            )
            messages = [
                {'role': row[0], 'content': row[1]}
                for row in cursor.fetchall()
            ]
            return messages
        finally:
            conn.close()

    def delete_session(self, session_id):
        """Delete a session and all its messages."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
            cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
            conn.commit()
        finally:
            conn.close()

# Global Instance
history_manager = ChatHistoryManager()
