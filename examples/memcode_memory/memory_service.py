"""Caller-scoped MemCode service, owned by the authenticated application."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass

from memcode_sdk import AsyncMemcodeV2Client, MemcodeSDKError
from memcode_sdk.v2_types import V2IngestResult


class MemoryUnavailable(Exception):
    """A safe error with no provider response body or credential."""


@dataclass(frozen=True)
class MemoryService:
    client: AsyncMemcodeV2Client
    space_id: str
    user_id: str
    actor_id: str
    timeout: float = 0.5

    def __post_init__(self) -> None:
        if any(
            not isinstance(v, str) or not v.strip()
            for v in (self.space_id, self.user_id, self.actor_id)
        ):
            raise ValueError("An authorized user, actor and space are required")
        if self.timeout <= 0:
            raise ValueError("Memory timeout must be positive")

    async def recall(self, query: str) -> list[str]:
        if not isinstance(query, str) or not query.strip() or len(query) > 5000:
            raise ValueError("Query must contain 1 to 5000 characters")
        try:
            result = await asyncio.wait_for(
                self.client.search(
                    context_space_id=self.space_id,
                    actor_id=self.actor_id,
                    query=query,
                    scope="context_only",
                    mode="memories",
                    include_original_chunks=False,
                    top_k=5,
                ),
                timeout=self.timeout,
            )
            return [
                item.content[:1000]
                for item in result.results
                if item.space.id == self.space_id and item.metadata.get("user_id") == self.user_id
            ][:5]
        except (MemcodeSDKError, asyncio.TimeoutError, ValueError, TypeError, AttributeError):
            raise MemoryUnavailable("Memory is temporarily unavailable") from None

    async def save_approved_fact(self, content: str) -> V2IngestResult:
        """Application call after the human approves this exact fact, never an LLM tool."""
        if not isinstance(content, str) or not content.strip() or len(content) > 16000:
            raise ValueError("Approved fact must contain 1 to 16000 characters")
        digest = hashlib.sha256(
            json.dumps(
                [self.space_id, self.user_id, content],
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        try:
            return await asyncio.wait_for(
                self.client.ingest(
                    space_id=self.space_id,
                    actor_id=self.actor_id,
                    content=content,
                    metadata={"user_id": self.user_id, "source": "user", "scope": "user"},
                    idempotency_key=f"livekit:{digest}",
                ),
                timeout=self.timeout,
            )
        except (MemcodeSDKError, asyncio.TimeoutError, ValueError, TypeError, AttributeError):
            raise MemoryUnavailable("Memory save is temporarily unavailable") from None
