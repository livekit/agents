from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Generator, Sequence
from typing import Any, Callable

from .. import llm, utils


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
        self._scheduled_fut = asyncio.Future[None]()
        self._authorize_event = asyncio.Event()

        self._generations: list[asyncio.Future[None]] = []

        # indicate if the speech was interrupted by a user turn
        self._interrupted_by_user: bool = False

        # internal tasks used by this generation
        self._tasks: list[asyncio.Task] = []
        self._chat_items: list[llm.ChatItem] = []
        self._num_steps = 1

        self._item_added_callbacks: set[Callable[[llm.ChatItem], None]] = set()
        self._done_callbacks: set[Callable[[SpeechHandle], None]] = set()

        def _on_done(_: asyncio.Future[None]) -> None:
            for cb in self._done_callbacks:
                cb(self)

        self._done_fut.add_done_callback(_on_done)
        self._maybe_run_final_output: Any = None  # kept private

    @staticmethod
    def create(allow_interruptions: bool = True) -> SpeechHandle:
        return SpeechHandle(
            speech_id=utils.shortuuid("speech_"),
            allow_interruptions=allow_interruptions,
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

    @allow_interruptions.setter
    def allow_interruptions(self, value: bool) -> None:
        """Allow or disallow interruptions on this SpeechHandle.

        When set to False, the SpeechHandle will no longer accept any incoming
        interruption requests until re-enabled. If the handle is already
        interrupted, clearing interruptions is not allowed.

        Args:
            value (bool): True to allow interruptions, False to disallow.

        Raises:
            RuntimeError: If attempting to disable interruptions when already interrupted.
        """
        if self.interrupted and not value:
            raise RuntimeError(
                "Cannot set allow_interruptions to False, the SpeechHandle is already interrupted"
            )

        self._allow_interruptions = value

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

        self._cancel()
        return self

    async def wait_for_playout(self) -> None:
        """Waits for the entire assistant turn to complete playback.

        This method waits until the assistant has fully finished speaking,
        including any finalization steps beyond initial response generation.
        This is appropriate to call when you want to ensure the speech output
        has entirely played out, including any tool calls and response follow-ups."""

        # raise an error to avoid developer mistakes
        from .agent import _get_activity_task_info

        if task := asyncio.current_task():
            info = _get_activity_task_info(task)
            if info and info.function_call and info.speech_handle == self:
                raise RuntimeError(
                    f"cannot call `SpeechHandle.wait_for_playout()` from inside the function tool `{info.function_call.name}` that owns this SpeechHandle. "
                    "This creates a circular wait: the speech handle is waiting for the function tool to complete, "
                    "while the function tool is simultaneously waiting for the speech handle.\n"
                    "To wait for the assistant’s spoken response prior to running this tool, use `RunContext.wait_for_playout()` instead."
                )

        await asyncio.shield(self._done_fut)

    def __await__(self) -> Generator[None, None, SpeechHandle]:
        async def _await_impl() -> SpeechHandle:
            await self.wait_for_playout()
            return self

        return _await_impl().__await__()

    def add_done_callback(self, callback: Callable[[SpeechHandle], None]) -> None:
        self._done_callbacks.add(callback)

    def remove_done_callback(self, callback: Callable[[SpeechHandle], None]) -> None:
        self._done_callbacks.discard(callback)

    async def wait_if_not_interrupted(self, aw: list[asyncio.futures.Future[Any]]) -> None:
        fs: list[asyncio.Future[Any]] = [
            asyncio.gather(*aw, return_exceptions=True),
            self._interrupt_fut,
        ]
        await asyncio.wait(fs, return_when=asyncio.FIRST_COMPLETED)

    def _cancel(self) -> SpeechHandle:
        if self.done():
            return self

        with contextlib.suppress(asyncio.InvalidStateError):
            self._interrupt_fut.set_result(None)

        return self

    def _add_item_added_callback(self, callback: Callable[[llm.ChatItem], Any]) -> None:
        self._item_added_callbacks.add(callback)

    def _remove_item_added_callback(self, callback: Callable[[llm.ChatItem], Any]) -> None:
        self._item_added_callbacks.discard(callback)

    def _item_added(self, items: Sequence[llm.ChatItem]) -> None:
        for item in items:
            for cb in self._item_added_callbacks:
                cb(item)

            self._chat_items.append(item)

    def _authorize_generation(self) -> None:
        fut = asyncio.Future[None]()
        self._generations.append(fut)
        self._authorize_event.set()

    def _clear_authorization(self) -> None:
        self._authorize_event.clear()

    async def _wait_for_authorization(self) -> None:
        await self._authorize_event.wait()

    async def _wait_for_generation(self, step_idx: int = -1) -> None:
        if not self._generations:
            raise RuntimeError("cannot use wait_for_generation: no active generation is running.")

        await asyncio.shield(self._generations[step_idx])

    async def _wait_for_scheduled(self) -> None:
        await asyncio.shield(self._scheduled_fut)

    def _mark_generation_done(self) -> None:
        if not self._generations:
            raise RuntimeError("cannot use mark_generation_done: no active generation is running.")

        with contextlib.suppress(asyncio.InvalidStateError):
            self._generations[-1].set_result(None)

    def _mark_done(self) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            # will raise InvalidStateError if the future is already done (interrupted)
            self._done_fut.set_result(None)
            if self._generations:
                self._mark_generation_done()  # preemptive generation could be cancelled before being scheduled

    def _mark_scheduled(self) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            self._scheduled_fut.set_result(None)

    def _mark_interrupted_by_user(self) -> None:
        self._interrupted_by_user = True
