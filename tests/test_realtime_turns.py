"""A realtime turn driven through AgentSession: chat context sync, tool calls and agent state."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    ToolResult,
    TurnHandlingOptions,
    function_tool,
    inference,
    llm,
    utils,
    vad,
)
from livekit.agents.voice import ModelSettings
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
from .fake_realtime import (
    FakeRealtimeModel,
    FakeRealtimeSession,
    _audio_frame,
    fake_capabilities,
    run_realtime_tool_turn,
)
from .fake_stt import FakeSTT

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


def _generation(
    text: str | None = None,
    *,
    audio: float = 0.0,
    calls: Sequence[llm.FunctionCall] = (),
    response_id: str = "response-id",
) -> llm.GenerationCreatedEvent:
    """A finished generation: one message when `text` is given, then `calls`."""
    messages = utils.aio.Chan[llm.MessageGeneration]()
    functions = utils.aio.Chan[llm.FunctionCall]()
    if text is not None:
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["audio", "text"] if audio else ["text"])
        text_ch.send_nowait(text)
        text_ch.close()
        if audio:
            audio_ch.send_nowait(_audio_frame(audio))
        audio_ch.close()
        messages.send_nowait(
            llm.MessageGeneration(
                message_id=f"{response_id}-message",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
    messages.close()
    for call in calls:
        functions.send_nowait(call)
    functions.close()
    return llm.GenerationCreatedEvent(
        message_stream=messages,
        function_stream=functions,
        user_initiated=True,
        response_id=response_id,
    )


async def _reply_future(
    rt_session: FakeRealtimeSession, index: int = 0
) -> asyncio.Future[llm.GenerationCreatedEvent]:
    for _ in range(500):
        if len(rt_session._reply_futs) > index:
            return rt_session._reply_futs[index]
        await asyncio.sleep(0)
    raise AssertionError(f"the session never asked the model for reply #{index + 1}")


async def _wait_for_agent_state(session: AgentSession, state: str) -> None:
    for _ in range(500):
        if session.agent_state == state:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"agent_state stuck at {session.agent_state!r}, expected {state!r}")


# --- agent state -------------------------------------------------------------


@pytest.mark.parametrize("audio_output", ["absent", "disabled"])
async def test_text_reply_returns_to_listening(audio_output: str) -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=audio_output != "absent"))
    async with AgentSession(llm=model) as session:
        if audio_output == "disabled":
            session.output.audio = FakeAudioOutput(can_pause=True)
            session.output.set_audio_enabled(False)
        session.output.transcription = FakeTextOutput()
        states: list[str] = []
        session.on("agent_state_changed", lambda ev: states.append(ev.new_state))
        await session.start(Agent(instructions="test"))

        reply = session.generate_reply()
        (await _reply_future(model.active_session)).set_result(_generation("Hello"))
        await reply.wait_for_playout()

        assert reply.exception() is None
        assert states == ["listening", "speaking", "listening"]


class _SpeechStartVAD(vad.VAD):
    """Report speech as soon as the test supplies audio, without a wall-clock schedule."""

    def __init__(self) -> None:
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.032))

    def stream(self) -> vad.VADStream:
        return _SpeechStartVADStream(self)


class _SpeechStartVADStream(vad.VADStream):
    async def _main_task(self) -> None:
        await self._input_ch.recv()
        self._event_ch.send_nowait(
            vad.VADEvent(
                type=vad.VADEventType.START_OF_SPEECH,
                samples_index=0,
                timestamp=0.0,
                speech_duration=0.0,
                silence_duration=0.0,
                frames=[],
            )
        )
        self._event_ch.send_nowait(
            vad.VADEvent(
                type=vad.VADEventType.INFERENCE_DONE,
                samples_index=512,
                timestamp=0.032,
                speech_duration=0.032,
                silence_duration=0.0,
                speaking=True,
            )
        )
        async for _ in self._input_ch:
            pass


@asynccontextmanager
async def _after_adaptive_text_reply(
    monkeypatch: pytest.MonkeyPatch, audio_output: str | None, stt: FakeSTT
) -> AsyncIterator[tuple[AgentSession, FakeRealtimeModel, FakeAudioInput]]:
    """A client-side turn-taking session that has just played out a text-only reply."""
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret")
    # keep the remote classifier pending: neither behavior may need its verdict
    verdicts = utils.aio.Chan[inference.OverlappingSpeechEvent]()
    stream = MagicMock()
    stream.__aiter__.side_effect = verdicts.__aiter__
    stream.aclose = AsyncMock()
    monkeypatch.setattr(inference.AdaptiveInterruptionDetector, "stream", lambda self: stream)

    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=False, user_transcription=False, audio_output=audio_output is not None
        )
    )
    async with AgentSession(
        llm=model,
        stt=stt,
        vad=_SpeechStartVAD(),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            interruption={"mode": "adaptive", "min_duration": 0.0, "backchannel_boundary": 0.0},
        ),
    ) as session:
        audio_input = FakeAudioInput()
        session.input.audio = audio_input
        if audio_output is not None:
            session.output.audio = FakeAudioOutput(can_pause=True)
            session.output.set_audio_enabled(False)
        session.output.transcription = FakeTextOutput()
        await session.start(Agent(instructions="test"))

        reply = session.generate_reply()
        (await _reply_future(model.active_session)).set_result(_generation("Hello"))
        await reply.wait_for_playout()
        yield session, model, audio_input


@pytest.mark.parametrize("audio_output", [None, "disabled"])
async def test_user_transcript_is_delivered_after_text_reply(
    monkeypatch: pytest.MonkeyPatch, audio_output: str | None
) -> None:
    stt = FakeSTT()
    async with _after_adaptive_text_reply(monkeypatch, audio_output, stt) as (
        session,
        _,
        audio_input,
    ):
        user_states = utils.aio.Chan[str]()
        transcripts = utils.aio.Chan[str]()
        session.on("user_state_changed", lambda ev: user_states.send_nowait(ev.new_state))
        session.on("user_input_transcribed", lambda ev: transcripts.send_nowait(ev.transcript))
        audio_input.push(0.032)
        assert await user_states.recv() == "speaking"
        (await stt.stream_ch.recv()).send_fake_transcript("Next user turn")

        assert await transcripts.recv() == "Next user turn"


@pytest.mark.parametrize("audio_output", [None, "disabled"])
async def test_vad_interrupts_pending_reply_after_text_reply(
    monkeypatch: pytest.MonkeyPatch, audio_output: str | None
) -> None:
    async with _after_adaptive_text_reply(monkeypatch, audio_output, FakeSTT()) as (
        session,
        model,
        audio_input,
    ):
        pending_reply = session.generate_reply()
        await _reply_future(model.active_session, 1)
        audio_input.push(0.032)
        await pending_reply.wait_for_playout()

        assert pending_reply.interrupted


async def test_say_during_tool_keeps_agent_thinking() -> None:
    # ctx.with_filler() speaks through session.say(); its end must not drop the state to
    # "listening" while the tool still runs, which would arm the user-away timer
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    class SlowToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup_order(self) -> str:
            """Look up an order, slowly."""
            tool_started.set()
            await release_tool.wait()
            return "order 42 shipped"

    async with AgentSession(llm=model, user_away_timeout=2.0) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(SlowToolAgent())
        user_states: list[str] = []
        session.on("user_state_changed", lambda ev: user_states.append(ev.new_state))

        reply = session.generate_reply()
        call = llm.FunctionCall(call_id="1", name="lookup_order", arguments="{}")
        (await _reply_future(model.active_session)).set_result(
            _generation("let me check", audio=1.0, calls=[call])
        )
        await tool_started.wait()

        try:
            await _wait_for_agent_state(session, "thinking")
            filler = session.say("just a moment")
            for _ in range(500):
                if model.active_session.say_futs:
                    break
                await asyncio.sleep(0)
            model.active_session.say_futs[0].set_result(
                _generation("just a moment", audio=0.5, response_id="filler")
            )
            await filler.wait_for_playout()
            assert session.agent_state == "thinking"

            await asyncio.sleep(4.0)  # longer than user_away_timeout
            assert session.agent_state == "thinking"
            assert "away" not in user_states
        finally:
            release_tool.set()

        await reply.wait_for_playout()


async def test_tool_without_a_reply_returns_the_agent_to_listening() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    class SilentToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def silent_lookup(self) -> None:
            """Look something up and return nothing, so no reply follows."""
            tool_started.set()
            await release_tool.wait()

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(SilentToolAgent())

        reply = session.generate_reply()
        call = llm.FunctionCall(call_id="1", name="silent_lookup", arguments="{}")
        (await _reply_future(model.active_session)).set_result(
            _generation("let me check", audio=1.0, calls=[call])
        )
        await tool_started.wait()
        await _wait_for_agent_state(session, "thinking")

        release_tool.set()
        await reply.wait_for_playout()
        await _wait_for_agent_state(session, "listening")


async def test_tool_reply_from_the_server_keeps_the_agent_thinking() -> None:
    # with auto_tool_reply_generation the server composes the reply, so no local task keeps
    # the agent busy while it does
    model = FakeRealtimeModel(capabilities=fake_capabilities(auto_tool_reply_generation=True))

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup_order(self) -> str:
            """Look up an order, so a reply is required."""
            await asyncio.sleep(1.0)
            return "order 42 shipped"

    async with AgentSession(llm=model, user_away_timeout=2.0) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(ToolAgent())
        user_states: list[str] = []
        session.on("user_state_changed", lambda ev: user_states.append(ev.new_state))

        reply = session.generate_reply()
        call = llm.FunctionCall(call_id="1", name="lookup_order", arguments="{}")
        (await _reply_future(model.active_session)).set_result(
            _generation("let me check", audio=1.0, calls=[call])
        )
        await reply.wait_for_playout()

        assert session.agent_state == "thinking"
        await asyncio.sleep(4.0)  # longer than user_away_timeout
        assert session.agent_state == "thinking"
        assert "away" not in user_states


# --- tool calls ----------------------------------------------------------------


async def test_tool_call_created_at_is_stamped_at_execution_start() -> None:
    executed = asyncio.Event()

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup(self) -> str:
            executed.set()
            return "ok"

    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    async with AgentSession(llm=model) as session:
        await session.start(ToolAgent())
        speech_handle = session.generate_reply()
        reply_fut = await _reply_future(model.active_session)

        # a call the model emitted well before this generation was authorized to speak
        call = llm.FunctionCall(
            call_id="call_1", name="lookup", arguments="{}", created_at=time.time() - 60
        )
        authorized_at = time.time()
        reply_fut.set_result(_generation("Let me look that up.", calls=[call]))
        await executed.wait()
        await speech_handle.wait_for_playout()

    assert call.created_at >= authorized_at
    # the call is recorded when it starts and again with its output, so it must not double up
    assert [i.call_id for i in session.history.items if i.type == "function_call"] == ["call_1"]


async def test_records_the_call_of_a_tool_that_never_ran() -> None:
    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup(self) -> str:
            return "ok"

    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    async with AgentSession(llm=model) as session:
        await session.start(ToolAgent())
        speech_handle = session.generate_reply()
        # a non-object arguments payload that json_repair cannot turn into one
        call = llm.FunctionCall(call_id="call_1", name="lookup", arguments="[1, 2, 3]")
        (await _reply_future(model.active_session)).set_result(
            _generation("Let me look that up.", calls=[call])
        )
        await speech_handle.wait_for_playout()
        chat_ctx = session.current_agent.chat_ctx

    for ctx in (chat_ctx, session.history):
        calls = [item for item in ctx.items if item.type == "function_call"]
        outputs = [item for item in ctx.items if item.type == "function_call_output"]
        assert [item.call_id for item in calls] == ["call_1"]
        assert [item.call_id for item in outputs] == ["call_1"]
        assert outputs[0].is_error

    # the pair survives the filter the agent applies to its own history
    copied = chat_ctx.copy(tools=session.current_agent.tools)
    assert [item.type for item in copied.items if item.type.startswith("function_call")] == [
        "function_call",
        "function_call_output",
    ]


@pytest.mark.parametrize("wrapped", [True, False])
async def test_cancelled_tool_reply_marks_the_synced_output(wrapped: bool) -> None:
    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")
            self.tool_executed = asyncio.Event()

        @function_tool
        async def lookup(self) -> str | ToolResult:
            """Look something up."""
            self.tool_executed.set()
            return ToolResult("ok", reply_required=True) if wrapped else "ok"

    agent = ToolAgent()
    _, model = await run_realtime_tool_turn(
        agent,
        tool_executed=agent.tool_executed,
        on_session=lambda s: s.on("function_tools_executed", lambda ev: ev.cancel_tool_reply()),
    )

    synced = [i for i in model.active_session.chat_ctx.items if i.type == "function_call_output"]
    assert len(synced) == 1
    assert synced[0].call_id == "1"
    assert not synced[0].reply_required


@pytest.mark.parametrize("reply_required", [True, False])
@pytest.mark.parametrize("interrupt", [True, False])
async def test_tool_result_reply_required_reaches_realtime_session(
    reply_required: bool, interrupt: bool
) -> None:
    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")
            self.tool_executed = asyncio.Event()

        @function_tool
        async def send_dtmf(self) -> ToolResult:
            """Send DTMF events."""
            self.tool_executed.set()
            return ToolResult("Successfully sent DTMF events: 1", reply_required=reply_required)

    agent = ToolAgent()
    session, model = await run_realtime_tool_turn(
        agent, tool_executed=agent.tool_executed, interrupt=interrupt
    )

    for items in (agent.chat_ctx.items, session.history.items, model.active_session.chat_ctx.items):
        outputs = [item for item in items if item.type == "function_call_output"]
        assert len(outputs) == 1
        assert outputs[0].call_id == "1"
        assert outputs[0].output == "Successfully sent DTMF events: 1"
        assert not outputs[0].is_error
        assert outputs[0].reply_required is (reply_required and not interrupt)
    assert model.active_session.generate_reply_calls == 1


# --- chat context sync ---------------------------------------------------------


class _AckedSession(FakeRealtimeSession):
    """Mirror a provider that commits context updates only when acknowledged."""

    def __init__(self, model: FakeRealtimeModel) -> None:
        super().__init__(model)
        self.tool_push_started = asyncio.Event()
        self.release_tool_ack = asyncio.Event()
        self.deleted_items: list[llm.ChatItem] = []
        self._provider_lock = asyncio.Lock()

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        async with self._provider_lock:
            if any(item.type == "function_call_output" for item in chat_ctx.items):
                self.tool_push_started.set()
                await self.release_tool_ack.wait()
            ids = {item.id for item in chat_ctx.items}
            self.deleted_items.extend(item for item in self._chat_ctx.items if item.id not in ids)
            self._chat_ctx = chat_ctx.copy()


class _AckedModel(FakeRealtimeModel):
    def __init__(self) -> None:
        super().__init__(
            capabilities=fake_capabilities(audio_output=False, auto_tool_reply_generation=False)
        )

    def session(self, *, turn_detection_disabled: bool = False) -> _AckedSession:
        session = _AckedSession(self)
        self.created_sessions.append(session)
        return session


@pytest.mark.parametrize("interrupt", [False, True])
async def test_text_turn_preserves_pending_tool_output(interrupt: bool) -> None:
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup(self) -> ToolResult:
            """Look up a result without requesting another model response."""
            tool_started.set()
            await release_tool.wait()
            return ToolResult("order 42 shipped", reply_required=False)

    model = _AckedModel()
    async with AgentSession(llm=model) as session:
        agent = ToolAgent()
        await session.start(agent)
        remote = model.created_sessions[-1]
        assert isinstance(remote, _AckedSession)
        try:
            tool_turn = session.generate_reply()
            reply_fut = await _reply_future(remote)
            call = llm.FunctionCall(call_id="call_1", name="lookup", arguments="{}")
            remote._chat_ctx.items.append(call)
            reply_fut.set_result(_generation(calls=[call]))
            await tool_started.wait()
            if interrupt:
                tool_turn.interrupt()
            release_tool.set()
            await remote.tool_push_started.wait()

            user_turn = session.generate_reply(user_input="What happened to my order?")
            authorized = asyncio.Event()
            wait_for_authorization = user_turn.wait_if_not_interrupted

            async def mark_authorized(futures: list[asyncio.Future[Any]]) -> None:
                await wait_for_authorization(futures)
                authorized.set()

            user_turn.wait_if_not_interrupted = mark_authorized
            # once authorized, the user turn's context update is queued behind the held tool ack
            await authorized.wait()
            remote.release_tool_ack.set()
            (await _reply_future(remote, 1)).set_result(_generation())
            await user_turn.wait_for_playout()
            await tool_turn.wait_for_playout()

            local_outputs = [i for i in agent.chat_ctx.items if i.type == "function_call_output"]
            remote_outputs = [i for i in remote.chat_ctx.items if i.type == "function_call_output"]
            assert len(local_outputs) == 1
            assert remote_outputs == local_outputs
            assert not any(i.type == "function_call_output" for i in remote.deleted_items)
            assert any(
                i.type == "message" and i.text_content == "What happened to my order?"
                for i in remote.chat_ctx.items
            )
        finally:
            release_tool.set()
            remote.release_tool_ack.set()
            for future in remote._reply_futs:
                if not future.done():
                    future.set_result(_generation())


async def test_explicit_context_replacement_can_remove_tool_outputs() -> None:
    model = _AckedModel()
    async with AgentSession(llm=model) as session:
        agent = Agent(instructions="test")
        await session.start(agent)
        remote = model.created_sessions[-1]
        assert isinstance(remote, _AckedSession)
        remote.release_tool_ack.set()
        context = llm.ChatContext(
            [
                llm.FunctionCall(call_id="call_1", name="lookup", arguments="{}"),
                llm.FunctionCallOutput(call_id="call_1", output="done", is_error=False),
            ]
        )
        await agent.update_chat_ctx(context, exclude_invalid_function_calls=False)
        await agent.update_chat_ctx(llm.ChatContext.empty())
        assert remote.chat_ctx.items == []
        assert any(i.type == "function_call_output" for i in remote.deleted_items)


class _ReplyTaskActivity(SimpleNamespace):
    """The attribute surface AgentActivity._realtime_reply_task touches, around a fake session."""

    def __init__(self, rt_session: FakeRealtimeSession) -> None:
        authorization_allowed = asyncio.Event()
        authorization_allowed.set()
        user_silence = asyncio.Event()
        user_silence.set()
        generation_calls: list[dict[str, Any]] = []

        async def _realtime_generation_task(**kwargs: Any) -> None:
            generation_calls.append(kwargs)

        super().__init__(
            _rt_session=rt_session,
            _rt_chat_ctx_lock=asyncio.Lock(),
            _authorization_allowed=authorization_allowed,
            _user_silence_event=user_silence,
            # mirrors the real method with hold_reply_while_user_speaking at its default:
            # an interruptible speech waits for the caller to stop before it is authorized
            _waits_for_user_silence=lambda speech_handle: speech_handle.allow_interruptions,
            _rt_overlapping_speech_enabled=False,
            tools=[],
            _on_enter_ignored_tools=lambda tool_ctx: [],
            _tool_choice=None,
            _rt_tool_choice_overridden=False,
            _restore_rt_tool_choice=lambda: None,
            _agent=SimpleNamespace(_chat_ctx=llm.ChatContext.empty()),
            _session=SimpleNamespace(
                _conversation_item_added=lambda msg: None,
                _update_agent_state=lambda state: None,
            ),
            _realtime_generation_task=_realtime_generation_task,
            generation_calls=generation_calls,
        )


def _run_reply_task(activity: _ReplyTaskActivity, handle: SpeechHandle) -> asyncio.Task[None]:
    handle._authorize_generation()
    return asyncio.create_task(
        AgentActivity._realtime_reply_task(
            cast(AgentActivity, activity),
            speech_handle=handle,
            model_settings=ModelSettings(),
            user_input="hello",
        )
    )


@pytest.mark.parametrize(
    "update_error",
    # the plugin raises this after its ack timeout, when the items were already sent
    [None, llm.RealtimeError("update_chat_ctx timed out.")],
    ids=["ok", "realtime-error"],
)
async def test_reply_is_generated_despite_a_realtime_chat_ctx_error(
    update_error: Exception | None,
) -> None:
    rt_session = FakeRealtimeModel().session()
    rt_session.update_error = update_error
    activity = _ReplyTaskActivity(rt_session)
    handle = SpeechHandle.create()
    task = _run_reply_task(activity, handle)

    for _ in range(500):
        if rt_session._reply_futs:
            break
        if task.done():
            task.result()  # surfaces why the task stopped short
            raise AssertionError("the reply task finished without asking for a reply")
        await asyncio.sleep(0)
    rt_session._reply_futs[-1].set_result(cast(llm.GenerationCreatedEvent, object()))
    await task

    assert rt_session.generate_reply_calls == 1
    assert len(activity.generation_calls) == 1
    assert not handle.done()
    assert any(
        item.type == "message" and item.role == "user" for item in activity._agent._chat_ctx.items
    )


async def test_unexpected_chat_ctx_error_fails_the_speech_handle() -> None:
    rt_session = FakeRealtimeModel().session()
    rt_session.update_error = RuntimeError("boom")
    activity = _ReplyTaskActivity(rt_session)
    handle = SpeechHandle.create()

    await _run_reply_task(activity, handle)

    assert rt_session.generate_reply_calls == 0
    assert activity.generation_calls == []
    assert isinstance(handle.exception(), RuntimeError)


@pytest.mark.parametrize("mutable_chat_context", [True, False])
async def test_say_adds_spoken_message_to_realtime_chat_ctx(mutable_chat_context: bool) -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            supports_say=False, mutable_chat_context=mutable_chat_context
        )
    )
    agent = Agent(instructions="test")
    async with AgentSession(llm=model) as session:
        await session.start(agent)
        await session.say("Thanks for joining! Are you ready?")

        local_msgs = [item for item in agent.chat_ctx.items if item.type == "message"]
        remote_msgs = [
            item for item in model.active_session.chat_ctx.items if item.type == "message"
        ]

    assert [msg.text_content for msg in local_msgs] == ["Thanks for joining! Are you ready?"]
    if mutable_chat_context:
        assert [msg.id for msg in remote_msgs] == [local_msgs[0].id]
    else:
        assert remote_msgs == []


async def test_say_lets_next_speech_start_while_chat_ctx_update_is_pending() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(supports_say=False))
    agent = Agent(instructions="test")
    async with AgentSession(llm=model) as session:
        await session.start(agent)
        rt_session = model.active_session
        release_update = asyncio.Event()
        apply_update = rt_session.update_chat_ctx

        async def unacknowledged_update(chat_ctx: llm.ChatContext) -> None:
            await release_update.wait()
            await apply_update(chat_ctx)

        rt_session.update_chat_ctx = unacknowledged_update  # type: ignore[method-assign]

        first = session.say("first")
        second = session.say("second")
        for _ in range(500):
            if any(
                isinstance(item, llm.ChatMessage) and item.text_content == "second"
                for item in agent.chat_ctx.items
            ):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("the second speech never started")
        assert not first.done()

        release_update.set()
        await first
        await second


async def test_say_places_spoken_message_before_later_remote_items() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(supports_say=False))
    agent = Agent(instructions="test")
    async with AgentSession(llm=model) as session:
        await session.start(agent)
        # a user turn the server added while the message was still playing
        remote_ctx = model.active_session.chat_ctx.copy()
        user_msg = remote_ctx.add_message(role="user", content="yes", created_at=time.time() + 60)
        await model.active_session.update_chat_ctx(remote_ctx)

        await session.say("Are you ready?")

        remote_ids = [
            item.id for item in model.active_session.chat_ctx.items if item.type == "message"
        ]
        spoken = next(
            item
            for item in agent.chat_ctx.items
            if isinstance(item, llm.ChatMessage) and item.text_content == "Are you ready?"
        )

    assert remote_ids == [spoken.id, user_msg.id]
