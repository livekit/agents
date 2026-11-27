from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .. import llm
    from .speech_handle import SpeechHandle


class TurnHooks(Protocol):
    """Observe committed turns and control their replies.

    Hooks returned at user-turn commit stay bound to that turn until reply
    processing finishes, even if they detach while the customer hook runs.

    Preemptive generation only calls ``on_reply_generation`` until its reply
    handle is accepted for output.
    """

    @property
    def reply_instructions(self) -> str | None:
        """Temporary realtime instructions, available after ``should_reply`` allows a reply."""
        ...

    def on_user_turn_committed(self, transcript: str, end_of_turn_delay: float | None) -> TurnHooks:
        """Bind hooks to the accepted user turn before the customer hook runs.
        Useful for per-turn classification task.
        """
        ...

    def on_user_turn_completed(self) -> None:
        """Observe completion of user-turn processing, including cancellation.
        Useful for per-turn idle tracking (non-human), unlike user away timeout (human).
        """
        ...

    def on_reply_generation(
        self, tools: list[llm.Tool | llm.Toolset]
    ) -> list[llm.Tool | llm.Toolset]:
        """Prepare tools for reply generation, including preemptive and manual replies.
        Useful for injecting the DTMF tool."""
        ...

    async def should_reply(self, chat_ctx: llm.ChatContext) -> bool:
        """Wait for permission to reply and optionally update the reply context.
        Useful for waiting through advertisement."""
        ...

    def on_agent_turn_committed(self, handle: SpeechHandle) -> None:
        """Observe agent's reply handle after it is created or adopted for output.
        Useful for voicemail playout tracking."""
        ...
