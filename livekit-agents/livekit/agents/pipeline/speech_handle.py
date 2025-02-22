from __future__ import annotations

import asyncio
import contextlib
from typing import Callable

from .. import utils


class SpeechHandle:
    SPEECH_PRIORITY_LOW = 0
    """Priority for messages that should be played after all other messages in the queue"""
    SPEECH_PRIORITY_NORMAL = 5
    """Every speech generates by the PipelineAgent defaults to this priority."""
    SPEECH_PRIORITY_HIGH = 10
    """Priority for important messages that should be played before others."""

    def __init__(
        self, *, speech_id: str, allow_interruptions: bool, step_index: int
    ) -> None:
        self._id = speech_id
        self._step_index = step_index
        self._allow_interruptions = allow_interruptions
        self._interrupt_fut = asyncio.Future()
        self._authorize_fut = asyncio.Future()
        self._playout_done_fut = asyncio.Future()

    @staticmethod
    def create(allow_interruptions: bool = True, step_index: int = 0) -> SpeechHandle:
        return SpeechHandle(
            speech_id=utils.shortuuid("speech_"),
            allow_interruptions=allow_interruptions,
            step_index=step_index,
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

    def done(self) -> bool:
        return self._playout_done_fut.done()

    def interrupt(self) -> None:
        if not self._allow_interruptions:
            raise ValueError("This generation handle does not allow interruptions")

        if self.done():
            return

        self._interrupt_fut.set_result(None)

    async def wait_for_playout(self) -> None:
        await asyncio.shield(self._playout_done_fut)

    def __await__(self):
        async def _await_impl() -> SpeechHandle:
            await self.wait_for_playout()
            return self

        return _await_impl().__await__()

    def add_done_callback(self, callback: Callable[[SpeechHandle], None]) -> None:
        self._playout_done_fut.add_done_callback(lambda _: callback(self))

    async def wait_if_not_interrupted(self, aw: list[asyncio.futures.Future]) -> None:
        await asyncio.wait(
            [asyncio.gather(*aw, return_exceptions=True), self._interrupt_fut],
            return_when=asyncio.FIRST_COMPLETED,
        )

    def _authorize_playout(self) -> None:
        self._authorize_fut.set_result(None)

    async def _wait_for_authorization(self) -> None:
        await asyncio.shield(self._authorize_fut)

    def _mark_playout_done(self) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            # will raise InvalidStateError if the future is already done (interrupted)
            self._playout_done_fut.set_result(None)
