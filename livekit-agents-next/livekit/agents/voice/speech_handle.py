from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Generator

from .. import _core, llm


class SpeechHandle:
    """One speech of the session, over the core's handle: its state is the core's, and awaiting
    it waits for the core to finish playing it out. Awaiting never raises; read
    ``exception()`` for a failure."""

    def __init__(self, core: _core.SpeechHandle) -> None:
        self._core = core
        self._callbacks: set[asyncio.Task[None]] = set()

    @property
    def id(self) -> str:
        return self._core.id

    @property
    def num_steps(self) -> int:
        return self._core.num_steps

    @property
    def scheduled(self) -> bool:
        return self._core.scheduled

    @property
    def interrupted(self) -> bool:
        return self._core.interrupted

    @property
    def allow_interruptions(self) -> bool:
        return self._core.allow_interruptions

    @allow_interruptions.setter
    def allow_interruptions(self, value: bool) -> None:
        self._core.allow_interruptions = value

    @property
    def chat_items(self) -> list[llm.ChatItem]:
        from .._core_view import from_core_item

        return [from_core_item(i) for i in self._core.chat_items]

    def done(self) -> bool:
        return self._core.done()

    def exception(self) -> BaseException | None:
        return self._core.exception()

    def interrupt(self, *, force: bool = False) -> SpeechHandle:
        self._core.interrupt(force=force)
        return self

    async def wait_for_playout(self) -> None:
        # a session that closed under the speech has nothing left to play
        with contextlib.suppress(RuntimeError):
            await self._core.wait_for_playout()

    async def _wait_for_generation(self, step_idx: int = -1) -> None:
        await self._core.wait_for_generation(None if step_idx < 0 else step_idx)

    def __await__(self) -> Generator[None, None, SpeechHandle]:
        async def _await() -> SpeechHandle:
            await self.wait_for_playout()
            return self

        return _await().__await__()

    def add_done_callback(self, callback: Callable[[SpeechHandle], None]) -> None:
        async def _call() -> None:
            await self.wait_for_playout()
            callback(self)

        task = asyncio.ensure_future(_call())
        self._callbacks.add(task)
        task.add_done_callback(self._callbacks.discard)

    def __repr__(self) -> str:
        return f"SpeechHandle(id={self.id!r}, done={self.done()}, interrupted={self.interrupted})"
