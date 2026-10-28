from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable, Callable
from typing import TYPE_CHECKING, Literal

from livekit import rtc
from livekit.agents import NOT_GIVEN, Agent, LiveSession, NotGivenOr, utils
from livekit.agents.llm import (
    ChatContext,
    FunctionCall,
    GenerationCreatedEvent,
    InputSpeechStartedEvent,
    InputSpeechStoppedEvent,
    InputTranscriptionCompleted,
    MessageGeneration,
    RealtimeCapabilities,
    RealtimeModel,
    RealtimeModelError,
    RealtimeSession,
)
from livekit.agents.llm.tool_context import Tool, ToolChoice, ToolContext
from livekit.agents.metrics import RealtimeModelMetrics

from .fake_io import FakeAudioOutput

if TYPE_CHECKING:
    from .a2a_driver import Reply


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
        # a different interruption model rather than a richer one, so it stays off
        "supports_overlapping_speech": False,
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
        self.say_futs: list[asyncio.Future[GenerationCreatedEvent]] = []
        # test hooks to pause or fail aclose() mid-swap
        self.aclose_entered = asyncio.Event()
        self.block_aclose: asyncio.Event | None = None
        self.aclose_error: Exception | None = None
        # a provider that only learns its usage as the connection drains reports it here
        self.closing_metrics: RealtimeModelMetrics | None = None
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
        fut: asyncio.Future[GenerationCreatedEvent] = asyncio.get_event_loop().create_future()
        self.say_futs.append(fut)
        return fut

    async def aclose(self) -> None:
        self.aclose_entered.set()
        if self.block_aclose is not None:
            await self.block_aclose.wait()
        if self.aclose_error is not None:
            raise self.aclose_error
        if self.closing_metrics is not None:
            self.emit("metrics_collected", self.closing_metrics)
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


REPLY_DELAY = 0.05
"""How long after the person's turn the provider starts its own reply."""


class ScriptedRealtimeSession(FakeRealtimeSession):
    """Answers each ``generate_reply`` and ``say`` at once, and each turn ``hear`` plays, with
    the model's next scripted reply; it keeps what was said in its context, as a provider does."""

    _realtime_model: ScriptedRealtimeModel

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        self._chat_ctx = chat_ctx.copy()

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        fut = super().generate_reply(instructions=instructions, tool_choice=tool_choice)
        said = instructions if isinstance(instructions, str) else ""
        fut.set_result(self._answer(said, user_initiated=True))
        return fut

    def say(self, text: str | AsyncIterable[str]) -> asyncio.Future[GenerationCreatedEvent]:
        fut = super().say(text)
        assert isinstance(text, str)
        self._chat_ctx.add_message(role="assistant", content=text)
        fut.set_result(_generation(text, [], user_initiated=True, hold=None))
        return fut

    def hear(self, transcript: str) -> None:
        """The person says ``transcript``: the provider transcribes it and, a moment later,
        replies on its own."""
        self.emit("input_speech_started", InputSpeechStartedEvent())
        self.emit("input_speech_stopped", InputSpeechStoppedEvent(user_transcription_enabled=True))
        item_id = utils.shortuuid("item_")
        self._chat_ctx.add_message(role="user", content=transcript, id=item_id)
        self.emit(
            "input_audio_transcription_completed",
            InputTranscriptionCompleted(item_id=item_id, transcript=transcript, is_final=True),
        )
        self._reply_later()

    def noise(self) -> None:
        """A sound on the line the provider takes for speech: no transcript, and a moment later
        a reply of its own."""
        self.emit("input_speech_started", InputSpeechStartedEvent())
        self.emit("input_speech_stopped", InputSpeechStoppedEvent(user_transcription_enabled=False))
        self._reply_later()

    def _reply_later(self) -> None:
        def reply() -> None:
            self.emit("generation_created", self._answer("", user_initiated=False))

        asyncio.get_running_loop().call_later(REPLY_DELAY, reply)

    def _answer(self, instructions: str, *, user_initiated: bool) -> GenerationCreatedEvent:
        model = self._realtime_model
        ctx = self._chat_ctx.copy()
        if instructions:
            ctx.add_message(role="developer", content=instructions)
        model.requests.append(ctx)
        reply = next((r for r in model.replies if r.when is None or r.when(ctx)), None)
        if reply is not None:
            model.replies.remove(reply)
        text = reply.content if reply is not None else ""
        calls = [
            FunctionCall(call_id=c.call_id, name=c.name, arguments=c.arguments)
            for c in (reply.tool_calls if reply is not None else [])
        ]
        if text:
            self._chat_ctx.add_message(role="assistant", content=text)
        self._chat_ctx.insert(calls)
        hold = reply.hold if reply is not None else None
        return _generation(text, calls, user_initiated=user_initiated, hold=hold)


class ScriptedRealtimeModel(FakeRealtimeModel):
    """A realtime model that says what ``replies`` script (``a2a_driver.Reply``, taken as
    ``ScriptedLLM`` takes them) and keeps, in ``requests``, the context of each reply with its
    instructions as a last ``developer`` message. It leaves the reply to a tool's output to
    the agent, and with ``say=False`` it has no voice of its own."""

    def __init__(self, *replies: Reply, say: bool = True) -> None:
        super().__init__(
            capabilities=fake_capabilities(auto_tool_reply_generation=False, supports_say=say)
        )
        self.replies = list(replies)
        self.requests: list[ChatContext] = []

    @property
    def active_session(self) -> ScriptedRealtimeSession:
        session = self.created_sessions[-1]
        assert isinstance(session, ScriptedRealtimeSession)
        return session

    def session(self, *, turn_detection_disabled: bool = False) -> ScriptedRealtimeSession:
        sess = ScriptedRealtimeSession(self, turn_detection_disabled=turn_detection_disabled)
        self.created_sessions.append(sess)
        return sess


_held: set[asyncio.Task[None]] = set()


def _generation(
    text: str, calls: list[FunctionCall], *, user_initiated: bool, hold: asyncio.Event | None
) -> GenerationCreatedEvent:
    """A generation of ``text`` with a short audio of it, then ``calls``; complete at once, or
    under way until ``hold`` is set."""
    message_ch = utils.aio.Chan[MessageGeneration]()
    function_ch = utils.aio.Chan[FunctionCall]()
    text_ch = utils.aio.Chan[str]()
    audio_ch = utils.aio.Chan[rtc.AudioFrame]()
    modalities: asyncio.Future[list[Literal["text", "audio"]]] = asyncio.Future()
    modalities.set_result(["audio", "text"])
    if text:
        message_ch.send_nowait(
            MessageGeneration(
                message_id=utils.shortuuid("msg_"),
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        text_ch.send_nowait(text)
        audio_ch.send_nowait(_audio_frame(0.1))
    for call in calls:
        function_ch.send_nowait(call)

    def end() -> None:
        text_ch.close()
        audio_ch.close()
        message_ch.close()
        function_ch.close()

    if hold is None:
        end()
    else:
        _held.add(task := asyncio.create_task(hold.wait()))
        task.add_done_callback(lambda t: (_held.discard(t), end()))
    return GenerationCreatedEvent(
        message_stream=message_ch,
        function_stream=function_ch,
        user_initiated=user_initiated,
        response_id=utils.shortuuid("resp_"),
    )


_SAMPLE_RATE = 24000


def _audio_frame(duration: float) -> rtc.AudioFrame:
    samples = int(_SAMPLE_RATE * duration)
    return rtc.AudioFrame(
        data=b"\x00\x01" * samples,
        sample_rate=_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=samples,
    )


async def run_realtime_tool_turn(
    agent: Agent,
    *,
    tool_executed: asyncio.Event,
    interrupt: bool = False,
    on_session: Callable[[LiveSession], None] | None = None,
) -> tuple[LiveSession, FakeRealtimeModel]:
    """Drive one realtime turn that speaks a second of audio and calls the agent's first tool.

    With ``interrupt`` the turn is cut short as soon as the tool returns; otherwise it plays out.
    """
    model = FakeRealtimeModel(capabilities=fake_capabilities())

    async with LiveSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        if on_session is not None:
            on_session(session)
        await session.start(agent)

        speech_handle = session.generate_reply()
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)

        message_ch = utils.aio.Chan[MessageGeneration]()
        function_ch = utils.aio.Chan[FunctionCall]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["audio", "text"])

        message_ch.send_nowait(
            MessageGeneration(
                message_id="message-id",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        text_ch.send_nowait("let me check")
        text_ch.close()
        # a full second of audio, so the turn is still playing when the tool returns
        audio_ch.send_nowait(_audio_frame(1.0))
        audio_ch.close()
        function_ch.send_nowait(
            FunctionCall(call_id="1", name=agent.tools[0].info.name, arguments="{}")
        )
        function_ch.close()

        model.active_session._reply_futs[0].set_result(
            GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="response-id",
            )
        )

        await asyncio.wait_for(tool_executed.wait(), timeout=5)
        if interrupt:
            session.interrupt()
        await asyncio.wait_for(speech_handle.wait_for_playout(), timeout=5)

    return session, model
