"""
Thread/session management for Backboard conversations.

Maps user identities to Backboard thread IDs. Threads are created
on demand and cached in memory for the duration of the agent session.
"""

from typing import Dict, Optional

import httpx
from livekit.agents import utils

logger = utils.log.logger


class SessionStore:
    """
    Manages user -> thread_id mappings for Backboard conversations.

    Creates threads on demand via the Backboard API and caches them
    in memory for fast lookups within a session.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        assistant_id: str,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._assistant_id = assistant_id
        self._cache: Dict[str, str] = {}
        self._client: Optional[httpx.AsyncClient] = None

    def set_assistant_id(self, assistant_id: str) -> None:
        """Update the assistant ID."""
        self._assistant_id = assistant_id

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client

    async def _create_thread(self) -> str:
        """Create a new Backboard thread on the configured assistant."""
        client = self._get_client()
        resp = await client.post(
            f"{self._base_url}/assistants/{self._assistant_id}/threads",
            headers={
                "X-API-Key": self._api_key,
                "Content-Type": "application/json",
            },
            json={},
        )
        resp.raise_for_status()
        return resp.json()["thread_id"]

    async def get_or_create_thread(self, user_id: str) -> str:
        """Get an existing thread or create a new one for the given user."""
        if user_id in self._cache:
            return self._cache[user_id]

        thread_id = await self._create_thread()
        self._cache[user_id] = thread_id
        logger.info(f"Created Backboard thread {thread_id} for user {user_id}")
        return thread_id

    def set_thread(self, user_id: str, thread_id: str) -> None:
        """Manually set a thread ID for a user (e.g. from external storage)."""
        self._cache[user_id] = thread_id

    def get_thread(self, user_id: str) -> Optional[str]:
        """Get thread_id for user from cache, or None."""
        return self._cache.get(user_id)

    def clear(self, user_id: str) -> None:
        """Remove a user's cached thread."""
        self._cache.pop(user_id, None)

    async def aclose(self) -> None:
        """Clean up HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
