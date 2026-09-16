from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .. import llm
    from .speech_handle import SpeechHandle


class ReplyGuard(Protocol):
    """Control replies without owning speech scheduling.

    A guard returned at turn commit stays bound to that turn until reply
    processing finishes, even if it detaches while the customer hook runs.
    """

    def tools_for_reply(
        self, tools: list[llm.Tool | llm.Toolset]
    ) -> list[llm.Tool | llm.Toolset]: ...

    async def should_reply(self, chat_ctx: llm.ChatContext) -> bool:
        """Wait for permission to reply and optionally update the reply context."""
        ...

    def on_reply_created(self, handle: SpeechHandle) -> None: ...
