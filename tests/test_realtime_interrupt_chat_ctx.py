"""Regression test for https://github.com/livekit/agents/issues/6424

When an interrupted realtime response produced a message the user never heard
(the server created it, but the agent was interrupted before pulling/playing
it), that message must be removed from the server-side chat context. The
cleanup used to be guarded on `any_skipped`, which only scans messages that
made it into `message_outputs` - the very list the abandoned messages are left
out of - so it never ran for them.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Literal

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, llm
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.voice import io

pytestmark = pytest.mark.unit

MessageStreamFactory = Callable[[], AsyncIterator[llm.MessageGeneration]]


async def _items(items):
    for item in items:
        yield item


def _modalities(*values):
    future = asyncio.get_running_loop().create_future()
    future.set_result(list(values))
    return future


def _audio_frame(duration: float = 0.02) -> rtc.AudioFrame:
    sample_rate = 16000
    samples = int(sample_rate * duration)
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


class _HermeticRealtimeSession(llm.RealtimeSession):
    def __init__(
        self,
        model: llm.RealtimeModel,
        message_stream_factory: MessageStreamFactory,
        server_created_message_ids: Sequence[str] = (),
    ) -> None:
        super().__init__(model)
        self._chat_context = llm.ChatContext.empty()
        self._tool_context = llm.ToolContext.empty()
        self._message_stream_factory = message_stream_factory
        self._server_created_message_ids = tuple(server_created_message_ids)

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_context

    @property
    def tools(self) -> llm.ToolContext:
        return self._tool_context

    async def update_instructions(self, instructions: str) -> None:
        pass

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self._chat_context = chat_ctx.copy()

    async def update_tools(self, tools) -> None:
        self._tool_context = llm.ToolContext(tools)

    def update_options(self, *, tool_choice=NOT_GIVEN) -> None:
        pass

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        pass

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools=NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        # A realtime server commits a conversation item as soon as it creates a
        # message for the response, well before the agent consumes it.
        for message_id in self._server_created_message_ids:
            self._chat_context.add_message(
                role="assistant",
                content=f"server content for {message_id}",
                id=message_id,
            )

        event = llm.GenerationCreatedEvent(
            message_stream=self._message_stream_factory(),
            function_stream=_items([]),
            user_initiated=True,
            response_id="response-1",
        )
        future = asyncio.get_running_loop().create_future()
        future.set_result(event)
        return future

    def commit_audio(self) -> None:
        pass

    def clear_audio(self) -> None:
        pass

    def interrupt(self) -> None:
        pass

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        pass

    async def aclose(self) -> None:
        pass


class _HermeticRealtimeModel(llm.RealtimeModel):
    def __init__(
        self,
        message_stream_factory: MessageStreamFactory,
        server_created_message_ids: Sequence[str] = (),
    ) -> None:
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                turn_detection=False,
                user_transcription=False,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
                mutable_chat_context=True,
                mutable_instructions=True,
                mutable_tools=True,
                per_response_tool_choice=True,
            )
        )
        self._message_stream_factory = message_stream_factory
        self._server_created_message_ids = tuple(server_created_message_ids)
        self.last_session: _HermeticRealtimeSession | None = None

    def session(self) -> llm.RealtimeSession:
        self.last_session = _HermeticRealtimeSession(
            self,
            self._message_stream_factory,
            server_created_message_ids=self._server_created_message_ids,
        )
        return self.last_session

    async def aclose(self) -> None:
        pass


class _AutoFinishAudioOutput(io.AudioOutput):
    def __init__(self) -> None:
        super().__init__(
            label="AutoFinishAudioOutput",
            capabilities=io.AudioOutputCapabilities(pause=True),
        )
        self._playback_duration = 0.0
        self._playback_active = False

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self._playback_duration += frame.duration
        if not self._playback_active:
            self._playback_active = True
            self.on_playback_started(created_at=time.time())

    def flush(self) -> None:
        super().flush()
        if not self._playback_active:
            return
        playback_position = self._playback_duration
        self._playback_duration = 0.0
        self._playback_active = False
        self.on_playback_finished(playback_position=playback_position, interrupted=False)

    def clear_buffer(self) -> None:
        if not self._playback_active:
            return
        self._playback_active = False
        self.on_playback_finished(playback_position=self._playback_duration, interrupted=True)


async def test_interrupt_removes_unplayed_message_from_server_ctx() -> None:
    played_out = asyncio.Event()
    interrupt_landed = asyncio.Event()

    async def message_stream() -> AsyncIterator[llm.MessageGeneration]:
        yield llm.MessageGeneration(
            message_id="played",
            text_stream=_items(["First answer."]),
            audio_stream=_items([_audio_frame()]),
            modalities=_modalities("text", "audio"),
        )
        # We are only resumed when the agent pulls the next message, which
        # happens after "played" fully finished its playout. That is exactly
        # the window where the interrupt has to land.
        played_out.set()
        await interrupt_landed.wait()
        yield llm.MessageGeneration(
            message_id="unplayed",
            text_stream=_items(["Second answer."]),
            audio_stream=_items([_audio_frame()]),
            modalities=_modalities("text", "audio"),
        )

    model = _HermeticRealtimeModel(
        message_stream, server_created_message_ids=("played", "unplayed")
    )
    audio_output = _AutoFinishAudioOutput()

    async with AgentSession(llm=model) as session:
        session.output.audio = audio_output
        await session.start(Agent(instructions="Answer briefly."), record=False)

        run_task = asyncio.ensure_future(session.run(user_input="Hello"))
        await asyncio.wait_for(played_out.wait(), timeout=2.0)

        speech = session.current_speech
        assert speech is not None

        # interrupt() only returns once the speech is done, and the speech
        # cannot finish while the message stream is parked on interrupt_landed,
        # so drive the interrupt as a task and release "unplayed" once the
        # interrupted flag has landed.
        interrupt_task = asyncio.ensure_future(session.interrupt(force=True))
        while not speech.interrupted:
            await asyncio.sleep(0)
        interrupt_landed.set()

        await asyncio.wait_for(interrupt_task, timeout=2.0)
        await asyncio.wait_for(run_task, timeout=2.0)

        local_ids = [m.id for m in session.history.messages() if m.role == "assistant"]

    assert model.last_session is not None
    server_ids = [
        item.id
        for item in model.last_session.chat_ctx.items
        if getattr(item, "role", None) == "assistant"
    ]

    # The user never heard "unplayed", so it must not survive server-side.
    assert local_ids == ["played"]
    assert server_ids == local_ids
