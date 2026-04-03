import asyncio
import functools
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel

from ...llm.chat_context import ChatContext, ChatMessage
from ...llm.llm import LLM
from ...llm.tool_context import Tool, ToolContext, function_tool
from ...llm.utils import execute_function_call
from ...log import logger
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import EventEmitter, aio, is_given, log_exceptions

HUMAN_SPEECH_THRESHOLD = 2.5
HUMAN_SILENCE_THRESHOLD = 0.5
MACHINE_SILENCE_THRESHOLD = 1.5
NO_SPEECH_THRESHOLD = 10.0
TIMEOUT = 20.0


class AMDCategory(str, Enum):
    HUMAN = "human"
    MACHINE_IVR = "machine-ivr"
    MACHINE_VM = "machine-vm"
    MACHINE_UNAVAILABLE = "machine-unavailable"
    UNCERTAIN = "uncertain"


class AMDResult(BaseModel):
    type: Literal["amd"] = "amd"
    speech_duration: float
    category: AMDCategory
    reason: str
    transcript: str
    delay: float

    @property
    def is_human(self) -> bool:
        return self.category == AMDCategory.HUMAN

    @property
    def is_machine(self) -> bool:
        return self.category in (
            AMDCategory.MACHINE_IVR,
            AMDCategory.MACHINE_VM,
            AMDCategory.MACHINE_UNAVAILABLE,
        )


# region: prompt
AMD_PROMPT = """Task:
Classify the call greeting transcript into exactly one of these categories:

human: A person answered (e.g., "Hello?", "This is John.").
machine-ivr: A prompt to press a key (e.g., "Press 1 to continue").
machine-vm: A voicemail greeting where leaving a message IS possible.
machine-unavailable: Any greeting indicating it's NOT possible to leave message, eg because mailbox is full, not setup, etc.
uncertain: For partial transcripts that are ambiguous.

Examples:
Input: "The person you called has a voice mailbox that hasn't been set up yet. Goodbye."
Output: machine-unavailable

Input: "Thank you for calling Truly Pizza in Dana Pointe. Our hours of operation are 11AM to 8PM, Sunday through Thursday, 11AM to 9PM, Friday and Saturday, and we're closed on Tuesdays."
Output: uncertain

Input: "You for calling Truly Pizza in Dana Pointe. Our hours of operation are 11AM to 8PM, Sunday through Thursday, 11AM to 9PM, Friday and Saturday, and we're closed on Tuesdays. If you'd like to place an order, please press 1 or head to our website to order online for pickup and local delivery."
Output: machine-ivr

Input: "I'm away from my desk. If you leave a message, I will get back to you."
Output: machine-vm

Input: "Hello, this is Lisa."
Output: human"""

# endregion


def _state_guard(method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    def wrapper(self: "_AMDClassifier", *args: Any, **kwargs: Any) -> Any:
        if self.closed or not self.started:
            logger.warning(
                "AMD state is invalid: started=%s, closed=%s",
                self.started,
                self.closed,
            )
            return
        return method(self, *args, **kwargs)

    return wrapper


class _AMDClassifier(EventEmitter[Literal["amd_result"]]):
    def __init__(self, llm: LLM):
        super().__init__()
        self._human_speech_threshold = HUMAN_SPEECH_THRESHOLD
        self._human_silence_threshold = HUMAN_SILENCE_THRESHOLD
        self._machine_silence_threshold = MACHINE_SILENCE_THRESHOLD

        self._input_ch: aio.Chan[str] = aio.Chan()
        self._classify_task: asyncio.Task[None] | None = None
        self._no_speech_timer: asyncio.TimerHandle | None = None
        self._silence_timer: asyncio.TimerHandle | None = None
        self._detection_timeout_timer: asyncio.TimerHandle | None = None

        self._verdict_result: AMDResult | None = None
        self._verdict_ready = asyncio.Event()

        self._llm = llm
        self._speech_started_at: float | None = None
        self._speech_ended_at: float | None = None
        self._started = False
        self._closed = False
        self._machine_silence_reached = False
        self._emitted = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._no_speech_timer = asyncio.get_running_loop().call_later(
            NO_SPEECH_THRESHOLD,
            functools.partial(
                self._silence_timer_callback,
                category=AMDCategory.MACHINE_UNAVAILABLE,
                reason="no_speech_timeout",
            ),
        )
        self._detection_timeout_timer = asyncio.get_running_loop().call_later(
            TIMEOUT,
            functools.partial(
                self._silence_timer_callback,
                category=AMDCategory.UNCERTAIN,
                reason="detection_timeout",
            ),
        )

    @_state_guard
    def on_user_speech_started(self) -> None:
        if self._silence_timer is not None:
            self._silence_timer.cancel()
            self._silence_timer = None
        if self._no_speech_timer is not None:
            self._no_speech_timer.cancel()
            self._no_speech_timer = None
        if self._speech_started_at is None:
            self._speech_started_at = time.time()
        self._machine_silence_reached = False

    @_state_guard
    def on_user_speech_ended(self, silence_duration: float) -> None:
        if self._speech_started_at is None:
            logger.warning("on_user_speech_ended called before on_user_speech_started")
            return

        self._speech_ended_at = time.time() - silence_duration
        speech_duration = self._speech_ended_at - self._speech_started_at
        if speech_duration <= self._human_speech_threshold:
            if self._silence_timer is not None:
                self._silence_timer.cancel()
                self._silence_timer = None
            self._silence_timer = asyncio.get_running_loop().call_later(
                max(0, self._human_silence_threshold - silence_duration),
                functools.partial(
                    self._silence_timer_callback,
                    category=AMDCategory.HUMAN,
                    reason="short_greeting",
                    speech_duration=speech_duration,
                ),
            )
            return

        if self._classify_task is None:
            self._classify_task = asyncio.create_task(self._classify_user_speech())

        if self._silence_timer is not None:
            self._silence_timer.cancel()
            self._silence_timer = None
        self._silence_timer = asyncio.get_running_loop().call_later(
            max(0, self._machine_silence_threshold - silence_duration),
            functools.partial(self._silence_timer_callback, speech_duration=speech_duration),
        )

    def _set_verdict(self, result: AMDResult) -> None:
        self._verdict_result = result
        self._try_emit_result()

    def _try_emit_result(self) -> None:
        if self._verdict_result is None:
            return
        if not self._machine_silence_reached:
            return
        if self._closed or self._emitted:
            return
        self._verdict_ready.set()
        if self._detection_timeout_timer is not None:
            self._detection_timeout_timer.cancel()
            self._detection_timeout_timer = None
        self.emit("amd_result", self._verdict_result)
        self._emitted = True

    @log_exceptions(logger=logger)
    @_state_guard
    def _silence_timer_callback(
        self,
        category: NotGivenOr[AMDCategory] = NOT_GIVEN,
        reason: NotGivenOr[str] = NOT_GIVEN,
        speech_duration: float | None = None,
    ) -> None:
        if is_given(category) and is_given(reason) and self._verdict_result is None:
            self._set_verdict(
                AMDResult(
                    speech_duration=speech_duration or self.speech_duration,
                    category=category,
                    reason=reason,
                    transcript="",
                    delay=time.time() - (self._speech_ended_at or time.time()),
                )
            )

        self._machine_silence_reached = True
        self._try_emit_result()

    @_state_guard
    def push_text(self, text: str) -> None:
        """Push transcript text to the AMD classifier."""
        if self._input_ch.closed:
            logger.debug("push_text called after close")
            return
        if self._classify_task is None:
            self._classify_task = asyncio.create_task(self._classify_user_speech())
        if self._no_speech_timer is not None:
            self._no_speech_timer.cancel()
            self._no_speech_timer = None
        self._input_ch.send_nowait(text)

    def end_input(self) -> None:
        if self._input_ch.closed:
            return
        self._input_ch.close()

    @log_exceptions(logger=logger)
    async def _classify_user_speech(self) -> None:
        ctx = {"transcript": ""}
        run_atask = None

        async def save_prediction(label: AMDCategory) -> None:
            """Save the prediction to the verdict."""
            if label != AMDCategory.UNCERTAIN:
                self._set_verdict(
                    AMDResult(
                        speech_duration=self.speech_duration,
                        category=label,
                        reason="llm",
                        transcript=ctx["transcript"],
                        delay=time.time() - (self._speech_ended_at or time.time()),
                    )
                )

        tools: list[Tool] = [function_tool(save_prediction)]

        @log_exceptions(logger=logger)
        async def _run(transcript: str) -> None:
            ctx["transcript"] = transcript
            stream = self._llm.chat(
                chat_ctx=ChatContext(
                    items=[
                        ChatMessage(role="system", content=[AMD_PROMPT]),
                        ChatMessage(role="user", content=[transcript]),
                    ]
                ),
                tools=tools,
                tool_choice="required",
            )
            response = await stream.collect()
            for tool_call in response.tool_calls:
                await execute_function_call(tool_call, ToolContext(stream.tools))

        try:
            async for text in self._input_ch:
                ctx["transcript"] = (ctx["transcript"] + " " + text).lstrip()
                if run_atask is not None:
                    await aio.cancel_and_wait(run_atask)
                run_atask = asyncio.create_task(_run(ctx["transcript"]))
        finally:
            if run_atask is not None:
                await aio.cancel_and_wait(run_atask)

    async def close(self) -> None:
        if self._closed:
            return

        self._verdict_ready.set()
        if not self._input_ch.closed:
            self._input_ch.close()

        if self._no_speech_timer is not None:
            self._no_speech_timer.cancel()
            self._no_speech_timer = None
        if self._silence_timer is not None:
            self._silence_timer.cancel()
            self._silence_timer = None
        if self._detection_timeout_timer is not None:
            self._detection_timeout_timer.cancel()
            self._detection_timeout_timer = None

        if self._classify_task is not None:
            await aio.cancel_and_wait(self._classify_task)

        self._closed = True
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def speech_duration(self) -> float:
        return (
            (self._speech_ended_at or time.time()) - self._speech_started_at
            if self._speech_started_at is not None
            else 0.0
        )
