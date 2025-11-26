"""
Advanced Memory Manager
Handles persistent conversation history and user context.
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger("memory-manager")


@dataclass
class ConversationMessage:
    """Represents a single conversation message."""

    user_id: str
    message: str
    role: str  # "user" or "assistant"
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "message": self.message,
            "role": self.role,
            "timestamp": self.timestamp.isoformat(),
            "metadata": json.dumps(self.metadata) if self.metadata else "{}",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMessage":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            message=data["message"],
            role=data["role"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=json.loads(data.get("metadata", "{}")),
        )


class MemoryManager:
    """Manages persistent conversation memory with SQLite."""

    def __init__(
        self,
        db_path: str = "./data/memory.db",
        window_size: int = 10,
    ):
        """
        Initialize the memory manager.

        Args:
            db_path: Path to SQLite database
            window_size: Number of recent messages to keep in context
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self.conn: Optional[sqlite3.Connection] = None

        logger.info(f"Memory Manager initialized with db: {db_path}")

    async def initialize(self):
        """Initialize the database and create tables."""
        try:
            # Connect to database
            self.conn = await asyncio.to_thread(
                sqlite3.connect, str(self.db_path), check_same_thread=False
            )
            self.conn.row_factory = sqlite3.Row

            # Create tables
            await asyncio.to_thread(self._create_tables)

            logger.info("✓ Memory database initialized")

        except Exception as e:
            logger.error(f"Error initializing memory database: {e}")
            raise

    def _create_tables(self):
        """Create database tables."""
        cursor = self.conn.cursor()

        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                role TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # User profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                preferences TEXT DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_user_id
            ON conversations(user_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
            ON conversations(timestamp)
        """)

        self.conn.commit()

    async def save_message(
        self,
        user_id: str,
        message: str,
        role: str,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Save a conversation message.

        Args:
            user_id: User identifier
            message: Message content
            role: Message role (user/assistant)
            timestamp: Message timestamp
            metadata: Optional metadata

        Returns:
            Message ID
        """
        if not self.conn:
            raise ValueError("Memory manager not initialized")

        try:
            msg = ConversationMessage(
                user_id=user_id,
                message=message,
                role=role,
                timestamp=timestamp,
                metadata=metadata or {},
            )

            cursor = await asyncio.to_thread(self.conn.cursor)
            await asyncio.to_thread(
                cursor.execute,
                """
                INSERT INTO conversations (user_id, message, role, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    msg.user_id,
                    msg.message,
                    msg.role,
                    msg.timestamp.isoformat(),
                    json.dumps(msg.metadata),
                ),
            )

            await asyncio.to_thread(self.conn.commit)

            msg_id = cursor.lastrowid
            logger.debug(f"Saved message {msg_id} for user {user_id}")

            return msg_id

        except Exception as e:
            logger.error(f"Error saving message: {e}")
            raise

    async def get_recent_history(
        self, user_id: str, limit: int = None
    ) -> List[Dict[str, str]]:
        """
        Get recent conversation history.

        Args:
            user_id: User identifier
            limit: Maximum number of messages (defaults to window_size)

        Returns:
            List of conversation messages
        """
        if not self.conn:
            return []

        limit = limit or self.window_size

        try:
            cursor = await asyncio.to_thread(self.conn.cursor)
            rows = await asyncio.to_thread(
                cursor.execute,
                """
                SELECT message, role, timestamp, metadata
                FROM conversations
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (user_id, limit),
            )

            rows = await asyncio.to_thread(cursor.fetchall)

            # Reverse to get chronological order
            messages = []
            for row in reversed(rows):
                messages.append(
                    {
                        "content": row["message"],
                        "role": row["role"],
                        "timestamp": row["timestamp"],
                    }
                )

            logger.debug(f"Retrieved {len(messages)} messages for user {user_id}")
            return messages

        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []

    async def get_summary(self, user_id: str, max_messages: int = 50) -> str:
        """
        Get a summary of conversation history.

        Args:
            user_id: User identifier
            max_messages: Maximum messages to include in summary

        Returns:
            Summary text
        """
        try:
            history = await self.get_recent_history(user_id, limit=max_messages)

            if not history:
                return ""

            # Build summary
            summary_lines = [
                f"Conversation history ({len(history)} messages):\n"
            ]

            for msg in history[-10:]:  # Last 10 messages
                role_emoji = "👤" if msg["role"] == "user" else "🤖"
                summary_lines.append(
                    f"{role_emoji} {msg['role']}: {msg['content'][:100]}..."
                )

            return "\n".join(summary_lines)

        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return ""

    async def delete_conversation(self, user_id: str) -> bool:
        """
        Delete all messages for a user.

        Args:
            user_id: User identifier

        Returns:
            Success status
        """
        if not self.conn:
            return False

        try:
            cursor = await asyncio.to_thread(self.conn.cursor)
            await asyncio.to_thread(
                cursor.execute,
                "DELETE FROM conversations WHERE user_id = ?",
                (user_id,),
            )
            await asyncio.to_thread(self.conn.commit)

            logger.info(f"Deleted conversation for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
            return False

    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics for a user.

        Args:
            user_id: User identifier

        Returns:
            User statistics
        """
        if not self.conn:
            return {}

        try:
            cursor = await asyncio.to_thread(self.conn.cursor)

            # Get message count
            row = await asyncio.to_thread(
                cursor.execute,
                """
                SELECT COUNT(*) as count, MIN(timestamp) as first_message,
                       MAX(timestamp) as last_message
                FROM conversations
                WHERE user_id = ?
                """,
                (user_id,),
            )

            row = await asyncio.to_thread(cursor.fetchone)

            return {
                "user_id": user_id,
                "message_count": row["count"],
                "first_message": row["first_message"],
                "last_message": row["last_message"],
            }

        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {}

    async def save_user_profile(
        self, user_id: str, name: str = None, preferences: Dict[str, Any] = None
    ) -> bool:
        """
        Save or update user profile.

        Args:
            user_id: User identifier
            name: User name
            preferences: User preferences

        Returns:
            Success status
        """
        if not self.conn:
            return False

        try:
            cursor = await asyncio.to_thread(self.conn.cursor)
            await asyncio.to_thread(
                cursor.execute,
                """
                INSERT OR REPLACE INTO user_profiles (user_id, name, preferences, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (user_id, name, json.dumps(preferences or {})),
            )
            await asyncio.to_thread(self.conn.commit)

            logger.info(f"Saved profile for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving user profile: {e}")
            return False

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user profile.

        Args:
            user_id: User identifier

        Returns:
            User profile or None
        """
        if not self.conn:
            return None

        try:
            cursor = await asyncio.to_thread(self.conn.cursor)
            row = await asyncio.to_thread(
                cursor.execute,
                "SELECT * FROM user_profiles WHERE user_id = ?",
                (user_id,),
            )

            row = await asyncio.to_thread(cursor.fetchone)

            if row:
                return {
                    "user_id": row["user_id"],
                    "name": row["name"],
                    "preferences": json.loads(row["preferences"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }

            return None

        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None

    async def close(self):
        """Close database connection."""
        if self.conn:
            await asyncio.to_thread(self.conn.close)
            logger.info("Memory database connection closed")
