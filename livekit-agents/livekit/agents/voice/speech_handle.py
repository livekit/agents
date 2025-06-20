from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from .. import llm, utils


Run_T = TypeVar("Run_T")


@dataclass
class RunResult(Generic[Run_T]):
    final_output: Run_T

    def __init__(self) -> None:
        pass

    def __await__(self) -> None:
        pass


class SpeechHandle:
    SPEECH_PRIORITY_LOW = 0
    """Priority for messages that should be played after all other messages in the queue"""
    SPEECH_PRIORITY_NORMAL = 5
    """Every speech generates by the VoiceAgent defaults to this priority."""
    SPEECH_PRIORITY_HIGH = 10
    """Priority for important messages that should be played before others."""

    def __init__(self, *, speech_id: str, allow_interruptions: bool) -> None:
        self._id = speech_id
        self._allow_interruptions = allow_interruptions

        self._interrupt_fut = asyncio.Future[None]()
        self._done_fut = asyncio.Future[None]()
        self._generation_fut = asyncio.Future[None]()
        self._authorize_event = asyncio.Event()

        # internal tasks used by this generation
        self._tasks: list[asyncio.Task] = []
        self._chat_items: list[llm.ChatItem] = []
        self._num_steps = 0

    @staticmethod
    def create(
        allow_interruptions: bool = True,
    ) -> SpeechHandle:
        return SpeechHandle(
            speech_id=utils.shortuuid("speech_"), allow_interruptions=allow_interruptions
        )

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def id(self) -> str:
        return self._id

    @property
    def interrupted(self) -> bool:
        return self._interrupt_fut.done()

    @property
    def allow_interruptions(self) -> bool:
        return self._allow_interruptions

    @property
    def chat_items(self) -> list[llm.ChatItem]:
        return self._chat_items

    def done(self) -> bool:
        return self._done_fut.done()

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
        await asyncio.shield(self._done_fut)

    def __await__(self) -> Generator[None, None, SpeechHandle]:
        async def _await_impl() -> SpeechHandle:
            await self.wait_for_playout()
            return self

        return _await_impl().__await__()

    def add_done_callback(self, callback: Callable[[SpeechHandle], None]) -> None:
        self._done_fut.add_done_callback(lambda _: callback(self))

    async def wait_if_not_interrupted(self, aw: list[asyncio.futures.Future[Any]]) -> None:
        fs: list[asyncio.Future[Any]] = [
            asyncio.gather(*aw, return_exceptions=True),
            self._interrupt_fut,
        ]
        await asyncio.wait(fs, return_when=asyncio.FIRST_COMPLETED)

    def _authorize_generation(self) -> None:
        self._generation_fut = asyncio.Future()
        self._authorize_event.set()

    def _clear_authorization(self) -> None:
        self._authorize_event.clear()

    async def _wait_for_authorization(self) -> None:
        await self._authorize_event.wait()

    def _mark_generation_done(self) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            self._generation_fut.set_result(None)

    async def _wait_for_generation(self) -> None:
        await asyncio.shield(self._generation_fut)

    def _mark_done(self) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            # will raise InvalidStateError if the future is already done (interrupted)
            self._done_fut.set_result(None)
