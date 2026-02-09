import asyncio
import contextlib
import time
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Literal, TypeAlias, get_args

from ...llm.chat_context import ChatMessage
from ...llm.llm import LLM, ChatContext
from ...log import logger
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import EventEmitter, aio, is_given, log_exceptions

HUMAN_SPEECH_THRESHOLD = 2.5
HUMAN_SILENCE_THRESHOLD = 0.5
MACHINE_SILENCE_THRESHOLD = 1.5
NO_SPEECH_THRESHOLD = 20.0


class AMDPhase(Enum):
    SHORT_SPEECH = 1
    LONG_SPEECH = 2


AMDCategory: TypeAlias = Literal["human", "machine-dtf", "machine-vm", "machine-nvm"]


@dataclass
class AMDResult:
    phase: AMDPhase
    category: AMDCategory
    reason: str
    transcript: str
    delay: float


AMD_PROMPT = """Task:
Classify the call greeting transcript into exactly one of these categories:

human: A person answered (e.g., "Hello?", "This is John.").
machine-dtf: A prompt to press a key (e.g., "Press 1 to continue").
machine-vm: A voicemail greeting where leaving a message IS possible.
machine-nvm: Any greeting indicating it's NOT possible to leave message, eg because mailbox is full, not setup, etc.
uncertain: For partial transcripts that are ambiguous.

Examples:
Input: "The person you called has a voice mailbox that hasn't been set up yet. Goodbye."
Output: machine-nvm

Input: "Thank you for calling Truly Pizza in Dana Pointe. Our hours of operation are 11AM to 8PM, Sunday through Thursday, 11AM to 9PM, Friday and Saturday, and we're closed on Tuesdays."
Output: uncertain

Input: "You for calling Truly Pizza in Dana Pointe. Our hours of operation are 11AM to 8PM, Sunday through Thursday, 11AM to 9PM, Friday and Saturday, and we're closed on Tuesdays. If you'd like to place an order, please press 1 or head to our website to order online for pickup and local delivery."
Output: machine-dtf

Input: "I'm away from my desk. If you leave a message, I will get back to you."
Output: machine-vm

Input: "Hello, this is Lisa."
Output: human"""


class AMD(EventEmitter[Literal["amd_result"]]):
    def __init__(self, llm: LLM):
        super().__init__()
        self._human_speech_threshold = HUMAN_SPEECH_THRESHOLD
        self._human_silence_threshold = HUMAN_SILENCE_THRESHOLD
        self._machine_silence_threshold = MACHINE_SILENCE_THRESHOLD

        self._phase: AMDPhase = AMDPhase.SHORT_SPEECH
        self._silence_timer: asyncio.TimerHandle | None = None
        self._input_ch: aio.Chan[str] = aio.Chan()
        self._classify_task: asyncio.Task[None] | None = None
        self._no_speech_timer: asyncio.TimerHandle = asyncio.get_running_loop().call_later(
            NO_SPEECH_THRESHOLD,
            partial(
                self._silence_timer_callback,
                category="machine-nvm",
                reason="no_speech_timeout",
            ),
        )

        self._verdit: AMDResult | None = None
        self._ready_event = asyncio.Event()
        self._llm = llm
        self._speech_started_at: float | None = None
        self._speech_ended_at: float | None = None
        self._closed = False

    def on_user_speech_started(self) -> None:
        if self._silence_timer is not None:
            self._silence_timer.cancel()
        if self._speech_started_at is None:
            self._speech_started_at = time.time()

    def on_user_speech_ended(self, silence_duration: float):
        if self._closed or self._ready_event.is_set():
            return

        speech_duration = time.time() - self._speech_started_at
        self._speech_ended_at = time.time()
        if speech_duration <= self._human_speech_threshold and self._phase == AMDPhase.SHORT_SPEECH:
            wait_for_silence = max(0, self._human_silence_threshold - silence_duration)
            if self._silence_timer is not None:
                self._silence_timer.cancel()
                self._silence_timer = None
            self._silence_timer = asyncio.get_event_loop().call_later(
                wait_for_silence,
                partial(
                    self._silence_timer_callback,
                    category="human",
                    reason="short_greeting",
                ),
            )
            return

        if speech_duration > self._human_speech_threshold:
            if self._phase == AMDPhase.SHORT_SPEECH:
                self._phase = AMDPhase.LONG_SPEECH
            wait_for_silence = max(0, self._machine_silence_threshold - silence_duration)
            if self._classify_task is None or self._classify_task.done():
                self._classify_task = asyncio.create_task(self._classify_user_speech())
            if self._silence_timer is not None:
                self._silence_timer.cancel()
                self._silence_timer = None

            self._silence_timer = asyncio.get_event_loop().call_later(
                wait_for_silence,
                self._silence_timer_callback,
            )

    @log_exceptions(logger=logger)
    def _silence_timer_callback(
        self,
        category: NotGivenOr[AMDCategory] = NOT_GIVEN,
        reason: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if self._closed:
            return

        if is_given(category) and is_given(reason):
            with contextlib.suppress(asyncio.InvalidStateError):
                self._verdit = AMDResult(
                    phase=self._phase,
                    category=category,
                    reason=reason,
                    transcript="",
                    delay=time.time() - self._speech_ended_at,
                )
                if self._no_speech_timer is not None:
                    self._no_speech_timer.cancel()
                    self._no_speech_timer = None

        if self._verdit is not None:
            self._ready_event.set()
            self.emit("amd_result", self._verdit)
            self.close()

    def push_text(self, text: str) -> None:
        if self._input_ch.closed or self._closed:
            return
        self._input_ch.send_nowait(text)

    def end_input(self) -> None:
        self._input_ch.close()

    @log_exceptions(logger=logger)
    async def _classify_user_speech(self) -> None:
        transcript = ""
        run_atask = None

        async def _run(transcript: str) -> None:
            stream = self._llm.chat(
                chat_ctx=ChatContext(
                    items=[
                        ChatMessage(role="system", content=[AMD_PROMPT]),
                        ChatMessage(role="user", content=[transcript]),
                    ]
                )
            )
            response = ""
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    response += chunk.delta.content
            response = response.strip().lower()
            if response in set(get_args(AMDCategory)):
                self._verdit = AMDResult(
                    phase=self._phase,
                    category=response,
                    reason="llm",
                    transcript=transcript,
                    delay=time.time() - self._speech_ended_at,
                )

            if self._no_speech_timer is not None:
                self._no_speech_timer.cancel()
                self._no_speech_timer = None

        async for text in self._input_ch:
            transcript += text
            if run_atask is not None and not run_atask.done():
                run_atask.cancel()
                run_atask = None
            run_atask = asyncio.create_task(_run(transcript))

    def close(self) -> None:
        if self._no_speech_timer is not None:
            self._no_speech_timer.cancel()
            self._no_speech_timer = None
        if self._silence_timer is not None:
            self._silence_timer.cancel()
            self._silence_timer = None
        if self._classify_task is not None and not self._classify_task.done():
            self._classify_task.cancel()
            self._classify_task = None
        if not self._input_ch.closed:
            self._input_ch.close()

        self._closed = True
