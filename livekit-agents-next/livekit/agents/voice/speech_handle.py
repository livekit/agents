from __future__ import annotations

import asyncio
import contextlib
import weakref
from collections.abc import Awaitable, Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Literal

from .. import _core, llm
from .._generated import speech_defaults
from ..log import logger


@dataclass
class InputDetails:
    modality: Literal["text", "audio"]


DEFAULT_INPUT_DETAILS = InputDetails(modality=speech_defaults.DEFAULT_INPUT_MODALITY)  # type: ignore[arg-type]

InterruptionSource = Literal["audio_activity", "user_turn", "programmatic"]
"""Why a speech was interrupted, as the reference's ``agent_turn`` trace names it."""


class SpeechHandle:
    """One speech of the session, over the core's handle: its state is the core's, and awaiting
    it waits for the core to finish playing it out. Awaiting never raises; read
    ``exception()`` for a failure. The core hands this object back wherever the speech is named
    (``speech_created``, a tool's ``RunContext``), for as long as it lives."""

    SPEECH_PRIORITY_LOW = speech_defaults.SPEECH_PRIORITY_LOW
    """Priority for messages that should be played after all other messages in the queue"""
    SPEECH_PRIORITY_NORMAL = speech_defaults.SPEECH_PRIORITY_NORMAL
    """Every speech generates by the VoiceAgent defaults to this priority."""
    SPEECH_PRIORITY_HIGH = speech_defaults.SPEECH_PRIORITY_HIGH
    """Priority for important messages that should be played before others."""

    def __init__(self, core: _core.SpeechHandle) -> None:
        self._core = core
        self._done_callbacks: list[Callable[[SpeechHandle], None]] = []
        core.host = weakref.ref(self)

    @staticmethod
    def _of(core: _core.SpeechHandle) -> SpeechHandle:
        """The handle over ``core``: the one already made, else a new one."""
        ref = core.host
        return (ref() if ref is not None else None) or SpeechHandle(core)

    @property
    def id(self) -> str:
        return self._core.id

    @property
    def num_steps(self) -> int:
        return self._core.num_steps

    @property
    def input_details(self) -> InputDetails:
        """How the input this speech answers arrived: the core runs every speech as audio."""
        return DEFAULT_INPUT_DETAILS

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
        """The error the speech failed with, or ``None``.

        Raises:
            asyncio.InvalidStateError: If the speech is not done yet.
        """
        if not self.done():
            raise asyncio.InvalidStateError("SpeechHandle is not done yet")
        return self._core.exception()

    def interrupt(
        self, *, force: bool = False, source: InterruptionSource = "programmatic"
    ) -> SpeechHandle:
        """Interrupt the speech. ``source`` is accepted for the reference's signature: the trace
        records the cause the core observed."""
        self._core.interrupt(force=force)
        return self

    async def wait_for_playout(self) -> None:
        from .agent_session import _origin

        await self._core.wait_for_playout(origin=_origin())

    async def _wait_for_generation(self, step_idx: int = -1) -> None:
        await self._core.wait_for_generation(None if step_idx < 0 else step_idx)

    def __await__(self) -> Generator[None, None, SpeechHandle]:
        async def _await() -> SpeechHandle:
            await self.wait_for_playout()
            return self

        return _await().__await__()

    def add_done_callback(self, callback: Callable[[SpeechHandle], None]) -> None:
        """``callback(handle)`` once the speech is done; soon, if it is already."""
        if self.done():
            asyncio.get_running_loop().call_soon(callback, self)
            return
        self._done_callbacks.append(callback)

    def remove_done_callback(self, callback: Callable[[SpeechHandle], None]) -> None:
        with contextlib.suppress(ValueError):
            self._done_callbacks.remove(callback)

    def _mark_done(self) -> None:
        """The core reported the speech done (``speech_done``): its callbacks run, in order."""
        callbacks, self._done_callbacks = self._done_callbacks, []
        for callback in callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.warning(f"error in done_callback: {callback}", exc_info=e)

    async def wait_if_not_interrupted(self, aw: Sequence[Awaitable[object]]) -> None:
        """Wait for ``aw``, unless the speech is interrupted first; ``aw`` is never cancelled."""
        gathered = asyncio.gather(*[asyncio.shield(fut) for fut in aw], return_exceptions=True)
        interrupted = asyncio.ensure_future(self._core.wait_for_interrupted())
        try:
            await asyncio.wait({gathered, interrupted}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for fut in (gathered, interrupted):
                if not fut.done():
                    fut.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await fut

    def __repr__(self) -> str:
        return f"SpeechHandle(id={self.id!r}, done={self.done()}, interrupted={self.interrupted})"
