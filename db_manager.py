import sqlite3
import hashlib

class DatabaseManager:
    def __init__(self, db_name="app.db"):
        self.db_name = db_name
        self._init_tables()

    def _get_conn(self):
        # Enables row factory for dict-like access
        conn = sqlite3.connect(self.db_name, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self._get_conn() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, role TEXT, message TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
            conn.execute("CREATE TABLE IF NOT EXISTS profile (username TEXT PRIMARY KEY, name TEXT)")
            conn.commit()

    def verify_user(self, username, password):
        hashed = hashlib.sha256(password.encode()).hexdigest()
        with self._get_conn() as conn:
            user = conn.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed)).fetchone()
            return user is not None

    def add_user(self, username, password):
        hashed = hashlib.sha256(password.encode()).hexdigest()
        try:
            with self._get_conn() as conn:
                conn.execute("INSERT INTO users VALUES (?, ?)", (username, hashed))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False

    def save_message(self, username, role, message):
        with self._get_conn() as conn:
            conn.execute("INSERT INTO chats (username, role, message) VALUES (?, ?, ?)", (username, role, message))
            conn.commit()

    def get_history(self, username):
        with self._get_conn() as conn:
            return conn.execute("SELECT role, message FROM chats WHERE username=? ORDER BY timestamp ASC", (username,)).fetchall()
