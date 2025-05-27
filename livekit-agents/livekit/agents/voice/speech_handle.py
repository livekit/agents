from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Generator
from typing import Any, Callable

from .. import llm, utils


class SpeechHandle:
    SPEECH_PRIORITY_LOW = 0
    """Priority for messages that should be played after all other messages in the queue"""
    SPEECH_PRIORITY_NORMAL = 5
    """Every speech generates by the VoiceAgent defaults to this priority."""
    SPEECH_PRIORITY_HIGH = 10
    """Priority for important messages that should be played before others."""

    def __init__(
        self,
        *,
        speech_id: str,
        allow_interruptions: bool,
        step_index: int,
        parent: SpeechHandle | None,
    ) -> None:
        self._id = speech_id
        self._step_index = step_index
        self._allow_interruptions = allow_interruptions
        self._interrupt_fut = asyncio.Future[None]()
        self._authorize_fut = asyncio.Future[None]()
        self._playout_done_fut = asyncio.Future[None]()
        self._parent = parent

        self._chat_message: llm.ChatMessage | None = None

    @staticmethod
    def create(
        allow_interruptions: bool = True,
        step_index: int = 0,
        parent: SpeechHandle | None = None,
    ) -> SpeechHandle:
        return SpeechHandle(
            speech_id=utils.shortuuid("speech_"),
            allow_interruptions=allow_interruptions,
            step_index=step_index,
            parent=parent,
        )

    @property
    def id(self) -> str:
        return self._id

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def interrupted(self) -> bool:
        return self._interrupt_fut.done()

    @property
    def allow_interruptions(self) -> bool:
        return self._allow_interruptions

    @property
    def chat_message(self) -> llm.ChatMessage | None:
        """
        Returns the assistant's generated chat message associated with this speech handle.

        Only available once the speech playout is complete.
        """
        return self._chat_message

    # TODO(theomonnom): should we introduce chat_items property as well for generated tools?

    @property
    def parent(self) -> SpeechHandle | None:
        """
        The parent handle that initiated the creation of the current speech handle.
        This happens when a tool call is made, a new SpeechHandle will be created for the tool response.
        """  # noqa: E501
        return self._parent

    def done(self) -> bool:
        return self._playout_done_fut.done()

    def interrupt(self) -> SpeechHandle:
        """Interrupt the current speech generation.

        Raises:
            RuntimeError: If this speech handle does not allow interruptions.

        Returns:
            SpeechHandle: The same speech handle that was interrupted.
        """
        if not self._allow_interruptions:
            raise RuntimeError("This generation handle does not allow interruptions")

        if self.done():
            return self

        with contextlib.suppress(asyncio.InvalidStateError):
            self._interrupt_fut.set_result(None)

        return self

    async def wait_for_playout(self) -> None:
        await asyncio.shield(self._playout_done_fut)

    def __await__(self) -> Generator[None, None, SpeechHandle]:
        async def _await_impl() -> SpeechHandle:
            await self.wait_for_playout()
            return self

        return _await_impl().__await__()

    def add_done_callback(self, callback: Callable[[SpeechHandle], None]) -> None:
        self._playout_done_fut.add_done_callback(lambda _: callback(self))

    async def wait_if_not_interrupted(self, aw: list[asyncio.futures.Future[Any]]) -> None:
        fs: list[asyncio.Future[Any]] = [
            asyncio.gather(*aw, return_exceptions=True),
            self._interrupt_fut,
        ]
        await asyncio.wait(fs, return_when=asyncio.FIRST_COMPLETED)

    def _authorize_playout(self) -> None:
        self._authorize_fut.set_result(None)

    async def _wait_for_authorization(self) -> None:
        await asyncio.shield(self._authorize_fut)

    def _mark_playout_done(self) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            # will raise InvalidStateError if the future is already done (interrupted)
            self._playout_done_fut.set_result(None)

    def _set_chat_message(self, chat_message: llm.ChatMessage) -> None:
        if self.done():
            raise RuntimeError("Cannot set chat message after speech has been played")

        if self._chat_message is not None:
            raise RuntimeError("Chat message already set")

        self._chat_message = chat_message
