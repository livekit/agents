from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from opentelemetry import context as otel_context

from .. import llm, utils
from ..log import logger

INTERRUPTION_TIMEOUT = 5.0  # seconds


@dataclass
class InputDetails:
    modality: Literal["text", "audio"]


DEFAULT_INPUT_DETAILS = InputDetails(modality="audio")


class SpeechHandle:
    SPEECH_PRIORITY_LOW = 0
    """Priority for messages that should be played after all other messages in the queue"""
    SPEECH_PRIORITY_NORMAL = 5
    """Every speech generates by the VoiceAgent defaults to this priority."""
    SPEECH_PRIORITY_HIGH = 10
    """Priority for important messages that should be played before others."""

    def __init__(
        self, *, speech_id: str, allow_interruptions: bool, input_details: InputDetails
    ) -> None:
        self._id = speech_id
        self._allow_interruptions = allow_interruptions
        self._interruption_holds = 0
        self._input_details = input_details

        self._interrupt_fut = asyncio.Future[None]()
        self._done_fut = asyncio.Future[None]()
        self._scheduled_fut = asyncio.Future[None]()
        self._authorize_event = asyncio.Event()

        self._generations: list[asyncio.Future[None]] = []

        # internal tasks used by this generation
        self._tasks: list[asyncio.Task] = []
        self._chat_items: list[llm.ChatItem] = []
        self._num_steps = 1
        self._agent_turn_context: otel_context.Context | None = None

        self._interrupt_timeout_handle: asyncio.TimerHandle | None = None

        self._item_added_callbacks: set[Callable[[llm.ChatItem], None]] = set()
        self._done_callbacks: set[Callable[[SpeechHandle], None]] = set()

        def _on_done(_: asyncio.Future[None]) -> None:
            for cb in list(self._done_callbacks):
                try:
                    cb(self)
                except Exception as e:
                    logger.warning(f"error in done_callback: {cb}", exc_info=e)

        self._done_fut.add_done_callback(_on_done)
        self._maybe_run_final_output: Any = None  # kept private
        self._error: BaseException | None = None

    @staticmethod
    def create(
        allow_interruptions: bool = True,
        input_details: InputDetails = DEFAULT_INPUT_DETAILS,
    ) -> SpeechHandle:
        return SpeechHandle(
            speech_id=utils.shortuuid("speech_"),
            allow_interruptions=allow_interruptions,
            input_details=input_details,
        )

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def id(self) -> str:
        return self._id

    @property
    def input_details(self) -> InputDetails:
        return self._input_details

    @property
    def _generation_id(self) -> str:
        return f"{self._id}_{self._num_steps}"

    @property
    def _parent_generation_id(self) -> str | None:
        if self._num_steps <= 1:
            return None
        return f"{self._id}_{self._num_steps - 1}"

    @property
    def scheduled(self) -> bool:
        return self._scheduled_fut.done()

    @property
    def interrupted(self) -> bool:
        return self._interrupt_fut.done()

    @property
    def allow_interruptions(self) -> bool:
        """Whether this speech may be interrupted right now.

        The **effective** state: the value last assigned, unless one or more
        `hold_interruptions()` holders are active, in which case it is False for as long
        as they are.

        A hold and an assignment are kept separately, so neither can lose the other: an
        assignment made during a hold cannot defeat the hold, and the last release cannot
        discard an assignment made while it was held.
        """
        return self._allow_interruptions and self._interruption_holds == 0

    @allow_interruptions.setter
    def allow_interruptions(self, value: bool) -> None:
        """Allow or disallow interruptions on this SpeechHandle.

        When set to False, the SpeechHandle will no longer accept any incoming
        interruption requests until re-enabled. If the handle is already
        interrupted, clearing interruptions is not allowed.

        Assigning while a `hold_interruptions()` holder is active records the value
        without lifting the hold; it takes effect when the last holder releases.

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
    def interruptions_held(self) -> bool:
        """Whether a `hold_interruptions()` holder is keeping this speech uninterruptible.

        Narrower than `not allow_interruptions`, which is also True for a speech simply
        created or configured uninterruptible.
        """
        return self._interruption_holds > 0

    def _hold_interruptions(self) -> None:
        """Disallow interruptions until every hold taken here is released.

        Counted rather than set, because the holders of one speech are not serialised
        against each other: the inline tasks awaited from a turn's parallel tool calls run
        one at a time, and a hold released between them would let the user turns of one
        task's sub-conversation interrupt the speech the rest are still anchored to.

        The count lives beside the assigned `allow_interruptions` value rather than
        overwriting it. Overwriting it meant an assignment during a hold silently defeated
        the hold -- `allow_interruptions = True` under a held speech made it interruptible
        again -- and meant the last release overwrote whatever had been assigned since.
        """
        if self.interrupted:
            raise RuntimeError("Cannot hold interruptions, the SpeechHandle is already interrupted")

        self._interruption_holds += 1

    def _release_interruptions(self) -> None:
        # Clamped rather than asserted: this runs from a `finally`, where raising would
        # replace whatever exception is already unwinding.
        if self._interruption_holds > 0:
            self._interruption_holds -= 1

    @contextlib.contextmanager
    def hold_interruptions(self) -> Generator[SpeechHandle, None, None]:
        """Keep this speech uninterruptible for the duration of the block.

        Counted, and independent of `allow_interruptions`: overlapping holders compose,
        and the speech becomes interruptible again only once the last one releases.

        Written for wording that has to arrive whole -- a consent question, a regulatory
        disclosure, a handover announcement -- on a session whose realtime model does
        server-side turn detection, where `allow_interruptions=False` is dropped with a
        warning per `say()` and refused outright per session.

        `interrupt(force=True)` still lands, so a hold survives ordinary barge-in without
        making the speech impossible to stop.

        **Two things a hold does not do.** While one is active the caller's audio is
        replaced with silence on the paths feeding the STT and the realtime model, unless
        the session opts out with
        `turn_handling=TurnHandlingOptions(interruption={"discard_audio_if_uninterruptible": False})`
        -- so by default the user is not heard for as long as the speech plays. And a user
        turn completing while a speech is held generates no reply for that turn. A hold is
        therefore for wording, not for a question whose answer is expected mid-sentence.

        Example:
            ```python
            handle = session.say("Before we go on, do I have your consent?")
            with handle.hold_interruptions():
                await handle.wait_for_playout()
            ```

        Yields:
            SpeechHandle: this handle, so the block may be written as
                `with handle.hold_interruptions() as speech:`.

        Raises:
            RuntimeError: If the speech has already been interrupted.
        """
        self._hold_interruptions()
        try:
            yield self
        finally:
            self._release_interruptions()

    @property
    def chat_items(self) -> list[llm.ChatItem]:
        return self._chat_items

    def done(self) -> bool:
        return self._done_fut.done()

    def exception(self) -> BaseException | None:
        """Return the error that caused this speech to fail, if any.

        Awaiting a SpeechHandle never raises; call this method after the handle
        is done to check whether the generation failed (e.g. ``llm.RealtimeError``
        when a realtime reply timed out).

        Raises:
            asyncio.InvalidStateError: If the speech is not done yet.

        Returns:
            BaseException | None: The error the generation failed with, or None.
        """
        if not self._done_fut.done():
            raise asyncio.InvalidStateError("SpeechHandle is not done yet")

        return self._error

    def interrupt(self, *, force: bool = False) -> SpeechHandle:
        """Interrupt the current speech generation.

        Raises:
            RuntimeError: If this speech handle is still running and does not allow
                interruptions.

        Returns:
            SpeechHandle: The same speech handle that was interrupted.
        """
        if self.interrupted or self.done():
            # already cancelled or finished: nothing to interrupt, and protection is moot
            return self

        if not force and not self.allow_interruptions:
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
            if (
                info
                and info.function_call
                and info.speech_handle == self
                and not info.function_call.extra.get("__livekit_agents_tool_non_blocking", False)
            ):
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
        if self.done():
            asyncio.get_running_loop().call_soon(callback, self)
            return

        self._done_callbacks.add(callback)

    def remove_done_callback(self, callback: Callable[[SpeechHandle], None]) -> None:
        self._done_callbacks.discard(callback)

    async def wait_if_not_interrupted(self, aw: list[asyncio.futures.Future[Any]]) -> None:
        # wrap each future in shield so we don't cancel them when we cancel the gather future
        gather_fut = asyncio.gather(*[asyncio.shield(fut) for fut in aw], return_exceptions=True)
        fs: set[asyncio.Future[Any]] = {gather_fut, self._interrupt_fut}
        _, pending = await asyncio.wait(fs, return_when=asyncio.FIRST_COMPLETED)
        if gather_fut in pending:
            with contextlib.suppress(asyncio.CancelledError):
                gather_fut.cancel()
                await gather_fut

    def _cancel(self) -> SpeechHandle:
        if self.done():
            return self

        if not self._interrupt_fut.done():
            self._interrupt_fut.set_result(None)

            def _on_timeout() -> None:
                logger.error(
                    "speech not done in time after interruption, cancelling the speech arbitrarily.",
                    extra={"speech_id": self._id, "timeout": INTERRUPTION_TIMEOUT},
                )
                for task in self._tasks:
                    task.cancel()
                self._mark_done()

            self._interrupt_timeout_handle = asyncio.get_event_loop().call_later(
                INTERRUPTION_TIMEOUT, _on_timeout
            )

        return self

    def _add_item_added_callback(self, callback: Callable[[llm.ChatItem], Any]) -> None:
        self._item_added_callbacks.add(callback)

    def _remove_item_added_callback(self, callback: Callable[[llm.ChatItem], Any]) -> None:
        self._item_added_callbacks.discard(callback)

    def _item_added(self, items: Sequence[llm.ChatItem]) -> None:
        for item in items:
            for cb in list(self._item_added_callbacks):
                try:
                    cb(item)
                except Exception as e:
                    logger.warning(f"error in item_added_callback: {cb}", exc_info=e)

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

    def _mark_done(self, error: BaseException | None = None) -> None:
        # the error is kept out of _done_fut so awaiting the handle never raises
        # (most handles are never awaited); it is exposed via exception() instead
        if not self._done_fut.done():
            if error is not None:
                self._error = error
            self._done_fut.set_result(None)

        if self._generations:
            self._mark_generation_done()

        if self._interrupt_timeout_handle is not None:
            self._interrupt_timeout_handle.cancel()
            self._interrupt_timeout_handle = None

    def _mark_scheduled(self) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            self._scheduled_fut.set_result(None)
