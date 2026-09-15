from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .. import llm
    from .speech_handle import SpeechHandle


class ReplyGuard(Protocol):
    """Control replies without owning speech scheduling.

    Each turn retains its guard until reply processing finishes, even if the
    guard detaches from the session while the customer hook runs.
    """

    def tools_for_reply(
        self, tools: list[llm.Tool | llm.Toolset]
    ) -> list[llm.Tool | llm.Toolset]: ...

    async def should_reply(self, turn_id: int | None, chat_ctx: llm.ChatContext) -> bool:
        """Wait for permission to reply and optionally update the reply context."""
        ...

    def on_reply_created(self, handle: SpeechHandle, turn_id: int | None) -> None: ...
