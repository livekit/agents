import asyncio
from collections.abc import Generator
from typing import Any, Literal

from ...llm.llm import LLM, ChatContext, ChatMessage
from ...utils import aio

HUMAN_SPEECH_THRESHOLD = 2.5
HUMAN_SILENCE_THRESHOLD = 0.5
MACHINE_SILENCE_THRESHOLD = 1.0


class AMD:
    def __init__(self, llm: LLM):
        self._human_speech_threshold = HUMAN_SPEECH_THRESHOLD
        self._human_silence_threshold = HUMAN_SILENCE_THRESHOLD
        self._machine_silence_threshold = MACHINE_SILENCE_THRESHOLD
        self._phase: Literal[1, 2] = 1
        self._silence_timer: asyncio.TimerHandle | None = None
        self._input_ch: aio.Chan[str] = aio.Chan()
        self._classify_task: asyncio.Task[None] | None = None

        self._verdit = asyncio.Future[
            Literal["human", "machine-dtf", "machine-vm", "machine-nvm"]
        ]()
        self._ready_event = asyncio.Event()
        self._llm = llm

    def __await__(
        self,
    ) -> Generator[Any, None, Literal["human", "machine-dtf", "machine-vm", "machine-nvm"]]:
        async def _await_impl() -> Literal["human", "machine-dtf", "machine-vm", "machine-nvm"]:
            await self._ready_event.wait()
            return self._verdit.result()

        return _await_impl().__await__()

    def on_user_speech_started(self) -> None:
        if self._silence_timer is not None:
            self._silence_timer.cancel()

    def on_user_speech_ended(self, speech_duration: float, silence_duration: float) -> bool:
        if speech_duration <= self._human_speech_threshold and self._phase == 1:
            wait_for_silence = max(0, self._human_silence_threshold - silence_duration)
            self._silence_timer = asyncio.get_event_loop().call_later(
                wait_for_silence, self._silence_timer_callback, label="human"
            )
        elif speech_duration > self._human_speech_threshold and self._phase == 2:
            wait_for_silence = max(0, self._machine_silence_threshold - silence_duration)
            if self._classify_task is None or self._classify_task.done():
                self._classify_task = asyncio.create_task(self._classify_user_speech())
            self._silence_timer = asyncio.get_event_loop().call_later(
                wait_for_silence, self._silence_timer_callback, label="machine"
            )

    def _silence_timer_callback(self, label: Literal["human", "machine"]) -> None:
        self._ready_event.set()

    def end_input(self) -> None:
        self._input_ch.close()

    async def _classify_user_speech(self) -> None:
        async for text in self._input_ch:
            response = await self._llm.chat(
                chat_ctx=ChatContext(messages=[ChatMessage(role="user", content=text)])
            )
            self._verdit.set_result(response.choices[0].message.content)
