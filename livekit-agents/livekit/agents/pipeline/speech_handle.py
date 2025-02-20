from __future__ import annotations

import asyncio
import contextlib
from typing import Callable

from .. import utils


class SpeechHandle:
    """
    Represents an individual speech generation request in the pipeline.
    
    Manages:
    - Lifecycle tracking of speech generation/playback
    - Interruption handling
    - Priority queuing
    - Coordination between input processing and output playout
    
    Each speech operation (TTS generation, LLM response) creates a SpeechHandle
    that tracks its progress through the system.
    """
    
    SPEECH_PRIORITY_LOW = 0
    """Priority for non-urgent messages that should be played after others"""
    SPEECH_PRIORITY_NORMAL = 5
    """Default priority for most speech generations"""
    SPEECH_PRIORITY_HIGH = 10
    """Priority for urgent messages that should interrupt current speech"""

    def __init__(
        self, *, speech_id: str, allow_interruptions: bool, step_index: int
    ) -> None:
        self._id = speech_id  # Unique identifier for this speech handle
        self._step_index = step_index  # Track function call recursion depth
        self._allow_interruptions = allow_interruptions  # Can this be interrupted?
        self._interrupt_fut = asyncio.Future()  # Signals interruption
        self._authorize_fut = asyncio.Future()  # Controls when playback starts
        self._playout_done_fut = asyncio.Future()  # Signals completion

    @staticmethod
    def create(allow_interruptions: bool = True, step_index: int = 0) -> SpeechHandle:
        """
        Factory method to create a new SpeechHandle.
        
        Args:
            allow_interruptions: Whether user input can interrupt this speech
            step_index: Tracks function call recursion depth (for LLM tool calls)
        """
        return SpeechHandle(
            speech_id=utils.shortuuid("speech_"),
            allow_interruptions=allow_interruptions,
            step_index=step_index,
        )

    @property
    def id(self) -> str:
        """Unique identifier for this speech instance"""
        return self._id

    @property
    def step_index(self) -> int:
        """Tracks depth of function call recursion (for LLM tool handling)"""
        return self._step_index

    @property
    def interrupted(self) -> bool:
        """True if this speech was interrupted before completion"""
        return self._interrupt_fut.done()

    @property
    def allow_interruptions(self) -> bool:
        """Whether this speech can be interrupted by user input"""
        return self._allow_interruptions

    def done(self) -> bool:
        """True if speech generation/playout has completed"""
        return self._playout_done_fut.done()

    def interrupt(self) -> None:
        """Attempt to cancel this speech generation"""
        if not self._allow_interruptions:
            raise ValueError("This generation handle does not allow interruptions")

        if self.done():
            return  # Already completed

        self._interrupt_fut.set_result(None)

    async def wait_for_playout(self) -> None:
        """Wait until speech generation and playout completes or is interrupted"""
        await asyncio.shield(self._playout_done_fut)

    def __await__(self):
        """Allows await syntax directly on SpeechHandle instances"""
        async def _await_impl() -> SpeechHandle:
            await self.wait_for_playout()
            return self

        return _await_impl().__await__()

    def add_done_callback(self, callback: Callable[[SpeechHandle], None]) -> None:
        """Add a callback to be executed when speech completes"""
        self._playout_done_fut.add_done_callback(lambda _: callback(self))

    async def wait_if_not_interrupted(self, aw: list[asyncio.futures.Future]) -> None:
        """
        Wait for a set of futures unless interrupted.
        
        Used to coordinate between speech generation tasks and potential interruptions.
        """
        await asyncio.wait(
            [asyncio.gather(*aw, return_exceptions=True), self._interrupt_fut],
            return_when=asyncio.FIRST_COMPLETED,
        )

    def _authorize_playout(self) -> None:
        """Internal method to authorize playback start (called by scheduler)"""
        self._authorize_fut.set_result(None)

    async def _wait_for_authorization(self) -> None:
        """Internal method to wait for playback authorization"""
        await asyncio.shield(self._authorize_fut)

    def _mark_playout_done(self) -> None:
        """Internal method to mark speech as completed"""
        with contextlib.suppress(asyncio.InvalidStateError):
            # Handle case where future was already completed (interrupted)
            self._playout_done_fut.set_result(None)
