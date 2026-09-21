"""Agent state around a realtime tool call.

A generation that ends while a tool is still running must stay "thinking":
`ctx.with_filler()` speaks through `session.say()`, and a realtime model that advertises
`supports_say` serves that from `_realtime_generation_task` instead of the TTS path, where
the reset to "listening" armed the user-away timer mid-tool (#6904).

The turn also has to hand the state back once the tool ends, which the realtime path never
did for a tool that asks for no reply.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, LatencyBudgetEvent, function_tool, llm, utils
from livekit.agents.llm import (
    FunctionCall,
    GenerationCreatedEvent,
    InputSpeechStartedEvent,
    InputSpeechStoppedEvent,
    MessageGeneration,
)

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, fake_capabilities

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

_SAMPLE_RATE = 24000


def _generation(
    *,
    response_id: str,
    text: str,
    audio_duration: float,
    function_calls: Sequence[FunctionCall] = (),
) -> GenerationCreatedEvent:
    """A single-message generation carrying `audio_duration` seconds of silence."""
    message_ch = utils.aio.Chan[MessageGeneration]()
    function_ch = utils.aio.Chan[FunctionCall]()
    text_ch = utils.aio.Chan[str]()
    audio_ch = utils.aio.Chan[rtc.AudioFrame]()
    modalities = asyncio.Future[list[str]]()
    modalities.set_result(["audio", "text"])

    message_ch.send_nowait(
        MessageGeneration(
            message_id=f"{response_id}-message",
            text_stream=text_ch,
            audio_stream=audio_ch,
            modalities=modalities,
        )
    )
    message_ch.close()
    text_ch.send_nowait(text)
    text_ch.close()
    samples = int(_SAMPLE_RATE * audio_duration)
    audio_ch.send_nowait(
        rtc.AudioFrame(
            data=b"\x00\x01" * samples,
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=samples,
        )
    )
    audio_ch.close()
    for fnc_call in function_calls:
        function_ch.send_nowait(fnc_call)
    function_ch.close()

    return GenerationCreatedEvent(
        message_stream=message_ch,
        function_stream=function_ch,
        user_initiated=True,
        response_id=response_id,
    )


async def _wait_for_agent_state(session: AgentSession, state: str) -> None:
    for _ in range(500):
        if session.agent_state == state:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"agent_state stuck at {session.agent_state!r}, expected {state!r}")


async def test_realtime_say_during_tool_keeps_agent_thinking() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    class SlowToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

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
        for _ in range(500):
            if model.active_session._reply_futs:
                break
            await asyncio.sleep(0)
        model.active_session._reply_futs[0].set_result(
            _generation(
                response_id="reply",
                text="let me check",
                audio_duration=1.0,
                function_calls=[FunctionCall(call_id="1", name="lookup_order", arguments="{}")],
            )
        )
        await asyncio.wait_for(tool_started.wait(), timeout=5)

        try:
            # the reply plays out while the tool runs; only the tool keeps the turn open after
            await _wait_for_agent_state(session, "thinking")

            filler = session.say("just a moment")
            for _ in range(500):
                if model.active_session.say_futs:
                    break
                await asyncio.sleep(0)
            model.active_session.say_futs[0].set_result(
                _generation(response_id="filler", text="just a moment", audio_duration=0.5)
            )
            await asyncio.wait_for(filler.wait_for_playout(), timeout=5)

            # the filler ended, the tool did not: "listening" here would arm the away timer
            assert session.agent_state == "thinking"

            await asyncio.sleep(4.0)  # longer than user_away_timeout
            assert session.agent_state == "thinking"
            assert "away" not in user_states
        finally:
            # a held tool would otherwise block the session teardown on any failure
            release_tool.set()

        await asyncio.wait_for(reply.wait_for_playout(), timeout=10)


async def test_realtime_tool_without_a_reply_returns_the_agent_to_listening() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    class SilentToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        @function_tool
        async def silent_lookup(self) -> None:
            """Look something up and return nothing, so no reply follows."""
            tool_started.set()
            await release_tool.wait()

    async with AgentSession(llm=model, user_away_timeout=2.0) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(SilentToolAgent())

        reply = session.generate_reply()
        for _ in range(500):
            if model.active_session._reply_futs:
                break
            await asyncio.sleep(0)
        model.active_session._reply_futs[0].set_result(
            _generation(
                response_id="reply",
                text="let me check",
                audio_duration=1.0,
                function_calls=[FunctionCall(call_id="1", name="silent_lookup", arguments="{}")],
            )
        )
        await asyncio.wait_for(tool_started.wait(), timeout=5)
        await _wait_for_agent_state(session, "thinking")

        release_tool.set()
        await asyncio.wait_for(reply.wait_for_playout(), timeout=10)

        # nothing speaks after the tool, so this turn is what has to hand the state back
        await _wait_for_agent_state(session, "listening")


async def test_realtime_tool_reply_from_the_server_keeps_the_agent_thinking() -> None:
    """With ``auto_tool_reply_generation`` the reply is server-side, so no local task marks
    the agent busy — the turn must hand back "thinking", not "listening"."""
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    assert model.capabilities.auto_tool_reply_generation

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

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
        for _ in range(500):
            if model.active_session._reply_futs:
                break
            await asyncio.sleep(0)
        model.active_session._reply_futs[0].set_result(
            _generation(
                response_id="reply",
                text="let me check",
                audio_duration=1.0,
                function_calls=[FunctionCall(call_id="1", name="lookup_order", arguments="{}")],
            )
        )
        await asyncio.wait_for(reply.wait_for_playout(), timeout=10)

        # the server is still composing the reply; dropping to "listening" here would kill
        # the client's busy indicator and arm the user-away timer over that reply
        assert session.agent_state == "thinking"
        await asyncio.sleep(4.0)  # longer than user_away_timeout
        assert session.agent_state == "thinking"
        assert "away" not in user_states


async def test_manual_tool_reply_keeps_realtime_latency_budget_turn() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(auto_tool_reply_generation=False))
    tool_called = asyncio.Event()

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup_weather(self) -> str:
            """Return the current weather."""
            tool_called.set()
            return "sunny"

    events: list[LatencyBudgetEvent] = []
    speeches = []
    async with AgentSession(llm=model, latency_budget={"budget": 0.01}) as session:
        session.output.audio = FakeAudioOutput()
        session.on("latency_budget", events.append)
        session.on("speech_created", speeches.append)
        await session.start(ToolAgent())

        rt_session = model.active_session
        rt_session.emit(
            "input_speech_stopped", InputSpeechStoppedEvent(user_transcription_enabled=False)
        )
        message_ch = utils.aio.Chan[MessageGeneration]()
        function_ch = utils.aio.Chan[FunctionCall]()
        message_ch.close()
        function_ch.send_nowait(
            FunctionCall(call_id="weather-1", name="lookup_weather", arguments="{}")
        )
        function_ch.close()
        rt_session.emit(
            "generation_created",
            GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=False,
            ),
        )

        await asyncio.wait_for(tool_called.wait(), timeout=5)
        await asyncio.sleep(0.05)
        for _ in range(500):
            if rt_session._reply_futs:
                break
            await asyncio.sleep(0)
        assert rt_session._reply_futs
        rt_session._reply_futs[0].set_result(
            _generation(response_id="manual-tool-reply", text="It is sunny", audio_duration=0.2)
        )
        assert speeches
        await asyncio.wait_for(speeches[0].speech_handle.wait_for_playout(), timeout=5)

    assert len(events) == 1
    assert events[0].level == "exceeded"
    assert events[0].speech_id is not None


@pytest.mark.parametrize("barge_in", [False, True])
async def test_google_auto_tool_reply_does_not_restart_latency_budget(barge_in: bool) -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities())

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup_weather(self) -> str:
            """Return the current weather."""
            return "sunny"

    events: list[LatencyBudgetEvent] = []
    speeches = []
    async with AgentSession(llm=model, latency_budget={"budget": 0.01}) as session:
        session.output.audio = FakeAudioOutput()
        session.on("latency_budget", events.append)
        session.on("speech_created", speeches.append)
        await session.start(ToolAgent())

        rt_session = model.active_session
        rt_session.emit(
            "input_speech_stopped", InputSpeechStoppedEvent(user_transcription_enabled=False)
        )
        await asyncio.sleep(0.02)
        first_generation = _generation(
            response_id="first",
            text="Let me check",
            audio_duration=0.02,
            function_calls=[
                FunctionCall(call_id="weather-1", name="lookup_weather", arguments="{}")
            ],
        )
        first_generation.user_initiated = False
        rt_session.emit("generation_created", first_generation)
        await asyncio.wait_for(speeches[0].speech_handle.wait_for_playout(), timeout=5)
        assert len(events) == 1

        for _ in range(500):
            if session._activity and session._activity._realtime_auto_tool_reply_pending:
                break
            await asyncio.sleep(0.01)
        assert session._activity and session._activity._realtime_auto_tool_reply_pending

        if barge_in:
            # A real barge-in cancels the old tool continuation and starts a new turn.
            rt_session.emit("input_speech_started", InputSpeechStartedEvent())
            rt_session.emit(
                "input_speech_stopped", InputSpeechStoppedEvent(user_transcription_enabled=False)
            )
        else:
            # Google emits a synthetic speech-start before an automatic tool generation.
            rt_session.emit("input_speech_started", InputSpeechStartedEvent(is_synthetic=True))
        tool_message_ch = utils.aio.Chan[MessageGeneration]()
        tool_function_ch = utils.aio.Chan[FunctionCall]()
        tool_message_ch.close()
        tool_function_ch.close()
        rt_session.emit(
            "generation_created",
            GenerationCreatedEvent(
                message_stream=tool_message_ch,
                function_stream=tool_function_ch,
                user_initiated=False,
            ),
        )
        rt_session.emit(
            "input_speech_stopped",
            InputSpeechStoppedEvent(user_transcription_enabled=False, is_synthetic=True),
        )
        await asyncio.sleep(0.05)
        assert len(events) == (2 if barge_in else 1)

        # The next actual user turn must still be able to start its own watch.
        rt_session.emit("input_speech_started", InputSpeechStartedEvent())
        rt_session.emit(
            "input_speech_stopped", InputSpeechStoppedEvent(user_transcription_enabled=False)
        )
        await asyncio.sleep(0.02)
        assert len(events) == (3 if barge_in else 2)


@pytest.mark.parametrize("failure", ["timeout", "update_error"])
async def test_missing_auto_tool_reply_does_not_poison_later_turns(failure: str) -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities())

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup_weather(self) -> str:
            """Return the current weather."""
            return "sunny"

    events: list[LatencyBudgetEvent] = []
    async with AgentSession(llm=model, latency_budget={"budget": 0.01}) as session:
        session.output.audio = FakeAudioOutput()
        session.on("latency_budget", events.append)
        await session.start(ToolAgent())
        rt_session = model.active_session
        if failure == "update_error":
            rt_session.update_error = llm.RealtimeError("update failed")

        rt_session.emit(
            "input_speech_stopped", InputSpeechStoppedEvent(user_transcription_enabled=False)
        )
        message_ch = utils.aio.Chan[MessageGeneration]()
        function_ch = utils.aio.Chan[FunctionCall]()
        message_ch.close()
        function_ch.send_nowait(
            FunctionCall(call_id="weather-1", name="lookup_weather", arguments="{}")
        )
        function_ch.close()
        rt_session.emit(
            "generation_created",
            GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=False,
            ),
        )

        for _ in range(500):
            if session._activity and session._activity._realtime_auto_tool_reply_pending:
                break
            await asyncio.sleep(0.01)
        if failure == "timeout":
            assert session._activity and session._activity._realtime_auto_tool_reply_pending
            await asyncio.sleep(5.1)
        else:
            await asyncio.sleep(0.02)

        assert session._activity is not None
        assert not session._activity._realtime_auto_tool_reply_pending
        assert session._activity._pending_auto_tool_reply_fut is None

        previous_events = len(events)
        rt_session.emit("input_speech_started", InputSpeechStartedEvent())
        rt_session.emit(
            "input_speech_stopped", InputSpeechStoppedEvent(user_transcription_enabled=False)
        )
        await asyncio.sleep(0.02)
        assert len(events) == previous_events + 1


async def test_old_auto_tool_reply_cleanup_keeps_new_expectation() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    async with AgentSession(llm=model) as session:
        await session.start(Agent(instructions="test"))
        assert session._activity is not None
        activity = session._activity
        old_reply = asyncio.get_running_loop().create_future()
        new_reply = asyncio.get_running_loop().create_future()
        activity._pending_auto_tool_reply_fut = new_reply
        activity._realtime_auto_tool_reply_pending = True

        activity._clear_realtime_auto_tool_reply(old_reply)
        assert activity._pending_auto_tool_reply_fut is new_reply
        assert activity._realtime_auto_tool_reply_pending

        activity._clear_realtime_auto_tool_reply(new_reply)
        assert activity._pending_auto_tool_reply_fut is None
        assert not activity._realtime_auto_tool_reply_pending
