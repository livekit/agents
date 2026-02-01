"""
User Memory System
Remembers users across sessions - preferences, history, patterns.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)


class UserMemory:
    """
    Persistent user memory that remembers:
    - User preferences
    - Scraping history
    - Conversation context
    - Learned patterns
    """

    def __init__(self, user_id: str, db_path: str = "./data/user_memory.db"):
        """
        Initialize user memory.

        Args:
            user_id: Unique user identifier
            db_path: Path to SQLite database
        """
        self.user_id = user_id
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db: Optional[aiosqlite.Connection] = None
        self.preferences: Dict[str, Any] = {}
        self.profile: Dict[str, Any] = {}

    async def load(self):
        """Load user memory from database"""
        self.db = await aiosqlite.connect(str(self.db_path))
        self.db.row_factory = aiosqlite.Row

        # Create tables
        await self._create_tables()

        # Load user profile
        await self._load_profile()

        # Load preferences
        await self._load_preferences()

        logger.info(f"Loaded memory for user: {self.user_id}")

    async def _create_tables(self):
        """Create database tables"""
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                created_at TEXT,
                last_seen TEXT,
                metadata TEXT
            )
        """)

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT,
                key TEXT,
                value TEXT,
                updated_at TEXT,
                PRIMARY KEY (user_id, key)
            )
        """)

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                user_message TEXT,
                assistant_message TEXT,
                timestamp TEXT,
                metadata TEXT
            )
        """)

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS scraping_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                url TEXT,
                success INTEGER,
                data TEXT,
                timestamp TEXT
            )
        """)

        await self.db.commit()

    async def _load_profile(self):
        """Load user profile"""
        cursor = await self.db.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?",
            (self.user_id,)
        )
        row = await cursor.fetchone()

        if row:
            self.profile = {
                "user_id": row["user_id"],
                "name": row["name"],
                "created_at": row["created_at"],
                "last_seen": row["last_seen"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
            }
        else:
            # Create new profile
            self.profile = {
                "user_id": self.user_id,
                "name": None,
                "created_at": datetime.utcnow().isoformat(),
                "last_seen": datetime.utcnow().isoformat(),
                "metadata": {}
            }
            await self._save_profile()

    async def _save_profile(self):
        """Save user profile"""
        await self.db.execute("""
            INSERT OR REPLACE INTO user_profiles
            (user_id, name, created_at, last_seen, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            self.profile["user_id"],
            self.profile.get("name"),
            self.profile["created_at"],
            self.profile["last_seen"],
            json.dumps(self.profile.get("metadata", {}))
        ))
        await self.db.commit()

    async def _load_preferences(self):
        """Load user preferences"""
        cursor = await self.db.execute(
            "SELECT key, value FROM user_preferences WHERE user_id = ?",
            (self.user_id,)
        )
        rows = await cursor.fetchall()

        self.preferences = {
            row["key"]: json.loads(row["value"])
            for row in rows
        }

    async def get_preferences(self) -> Dict[str, Any]:
        """Get all user preferences"""
        return self.preferences.copy()

    async def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a specific preference"""
        return self.preferences.get(key, default)

    async def set_preference(self, key: str, value: Any):
        """Set a user preference"""
        self.preferences[key] = value

        await self.db.execute("""
            INSERT OR REPLACE INTO user_preferences
            (user_id, key, value, updated_at)
            VALUES (?, ?, ?, ?)
        """, (
            self.user_id,
            key,
            json.dumps(value),
            datetime.utcnow().isoformat()
        ))
        await self.db.commit()

        logger.info(f"Set preference {key}={value} for user {self.user_id}")

    async def add_conversation(
        self,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save a conversation turn"""
        await self.db.execute("""
            INSERT INTO conversation_history
            (user_id, user_message, assistant_message, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            self.user_id,
            user_message,
            assistant_message,
            datetime.utcnow().isoformat(),
            json.dumps(metadata or {})
        ))
        await self.db.commit()

    async def get_conversation_history(
        self,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get conversation history"""
        cursor = await self.db.execute("""
            SELECT user_message, assistant_message, timestamp, metadata
            FROM conversation_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (self.user_id, limit))

        rows = await cursor.fetchall()

        return [
            {
                "user_message": row["user_message"],
                "assistant_message": row["assistant_message"],
                "timestamp": row["timestamp"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
            }
            for row in reversed(rows)
        ]

    async def add_scraping_history(
        self,
        url: str,
        success: bool,
        data: Dict[str, Any]
    ):
        """Record a scraping action"""
        await self.db.execute("""
            INSERT INTO scraping_history
            (user_id, url, success, data, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            self.user_id,
            url,
            1 if success else 0,
            json.dumps(data),
            datetime.utcnow().isoformat()
        ))
        await self.db.commit()

    async def get_scraping_history(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get scraping history"""
        cursor = await self.db.execute("""
            SELECT url, success, data, timestamp
            FROM scraping_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (self.user_id, limit))

        rows = await cursor.fetchall()

        return [
            {
                "url": row["url"],
                "success": bool(row["success"]),
                "data": json.loads(row["data"]),
                "timestamp": row["timestamp"]
            }
            for row in rows
        ]

    async def get_profile(self) -> Dict[str, Any]:
        """Get user profile"""
        return self.profile.copy()

    async def update_profile(self, **kwargs):
        """Update user profile"""
        self.profile.update(kwargs)
        self.profile["last_seen"] = datetime.utcnow().isoformat()
        await self._save_profile()

    async def save(self):
        """Save all data and close database"""
        if self.db:
            await self.db.commit()
            await self.db.close()

    async def clear_history(self):
        """Clear conversation and scraping history"""
        await self.db.execute(
            "DELETE FROM conversation_history WHERE user_id = ?",
            (self.user_id,)
        )
        await self.db.execute(
            "DELETE FROM scraping_history WHERE user_id = ?",
            (self.user_id,)
        )
        await self.db.commit()

        logger.info(f"Cleared history for user {self.user_id}")
