from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable
from typing import Literal

from livekit import rtc
from livekit.agents import NOT_GIVEN, NotGivenOr
from livekit.agents.llm import (
    ChatContext,
    GenerationCreatedEvent,
    RealtimeCapabilities,
    RealtimeModel,
    RealtimeModelError,
    RealtimeSession,
)
from livekit.agents.llm.tool_context import Tool, ToolChoice, ToolContext


def fake_capabilities(**overrides: bool) -> RealtimeCapabilities:
    """Build a RealtimeCapabilities with everything enabled, overriding individual flags."""
    defaults: dict[str, bool] = {
        "message_truncation": True,
        "turn_detection": True,
        "user_transcription": True,
        "auto_tool_reply_generation": True,
        "audio_output": True,
        "manual_function_calls": True,
        "mutable_chat_context": True,
        "mutable_instructions": True,
        "mutable_tools": True,
        "per_response_tool_choice": True,
        "supports_say": True,
    }
    defaults.update(overrides)
    return RealtimeCapabilities(**defaults)


class FakeRealtimeSession(RealtimeSession):
    """A hermetic RealtimeSession that records calls and can be scripted to fail."""

    def __init__(self, model: FakeRealtimeModel, *, turn_detection_disabled: bool = False) -> None:
        super().__init__(model)
        self.turn_detection_disabled = turn_detection_disabled
        self._chat_ctx = ChatContext.empty()
        self._tools = ToolContext.empty()
        self.closed = False
        self.interrupted = False
        self.committed = False
        self.audio_cleared = False
        self.pushed_audio: list[rtc.AudioFrame] = []
        self.generate_reply_calls = 0
        self.updated_instructions: str | None = None
        self.tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN
        self.say_calls: list[str | AsyncIterable[str]] = []
        self.user_activity_started = False
        self._reply_futs: list[asyncio.Future[GenerationCreatedEvent]] = []
        # test hooks to pause or fail aclose() mid-swap
        self.aclose_entered = asyncio.Event()
        self.block_aclose: asyncio.Event | None = None
        self.aclose_error: Exception | None = None
        # test hook to fail bring-up (raised from update_chat_ctx during _update_session)
        self.update_error: Exception | None = None

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> ToolContext:
        return self._tools

    async def update_instructions(self, instructions: str) -> None:
        self.updated_instructions = instructions

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        if self.update_error is not None:
            raise self.update_error
        self._chat_ctx = chat_ctx

    async def update_tools(self, tools: list[Tool]) -> None:
        self._tools = ToolContext(tools)

    def update_options(self, *, tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN) -> None:
        self.tool_choice = tool_choice

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self.pushed_audio.append(frame)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        self.generate_reply_calls += 1
        fut: asyncio.Future[GenerationCreatedEvent] = asyncio.get_event_loop().create_future()
        self._reply_futs.append(fut)
        return fut

    def commit_audio(self) -> None:
        self.committed = True

    def clear_audio(self) -> None:
        self.audio_cleared = True

    def interrupt(self) -> None:
        self.interrupted = True

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        pass

    def start_user_activity(self) -> None:
        self.user_activity_started = True

    def say(self, text: str | AsyncIterable[str]) -> asyncio.Future[GenerationCreatedEvent]:
        self.say_calls.append(text)
        return asyncio.get_event_loop().create_future()

    async def aclose(self) -> None:
        self.aclose_entered.set()
        if self.block_aclose is not None:
            await self.block_aclose.wait()
        if self.aclose_error is not None:
            raise self.aclose_error
        self.closed = True

    # --- test helpers -------------------------------------------------------

    def emit_error(self, *, recoverable: bool) -> None:
        """Simulate the provider emitting an error event."""
        self.emit(
            "error",
            RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model.label,
                error=RuntimeError("simulated provider error"),
                recoverable=recoverable,
            ),
        )


class FakeRealtimeModel(RealtimeModel):
    """A RealtimeModel whose sessions are FakeRealtimeSession, tracking every session created."""

    def __init__(
        self,
        *,
        capabilities: RealtimeCapabilities | None = None,
        label: str = "fake.realtime.RealtimeModel",
    ) -> None:
        super().__init__(capabilities=capabilities or fake_capabilities())
        self._label = label
        self.created_sessions: list[FakeRealtimeSession] = []
        self.closed = False
        # when set, every session this model creates fails to bring up
        self.bring_up_error: Exception | None = None

    @property
    def active_session(self) -> FakeRealtimeSession:
        return self.created_sessions[-1]

    def session(self, *, turn_detection_disabled: bool = False) -> FakeRealtimeSession:
        sess = FakeRealtimeSession(self, turn_detection_disabled=turn_detection_disabled)
        sess.update_error = self.bring_up_error
        self.created_sessions.append(sess)
        return sess

    async def aclose(self) -> None:
        self.closed = True
