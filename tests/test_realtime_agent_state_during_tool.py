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
from livekit.agents import Agent, AgentSession, function_tool, utils
from livekit.agents.llm import FunctionCall, GenerationCreatedEvent, MessageGeneration
from livekit.agents.voice.speech_handle import SpeechHandle

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


@pytest.mark.parametrize("outcome", ["reply", "silent", "interrupted"])
async def test_realtime_tool_reply_waiting_behind_speech_keeps_agent_thinking(outcome: str) -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(auto_tool_reply_generation=False))
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    class ToolAgent(Agent):
        @function_tool
        async def lookup(self) -> str:
            """Look up an order."""
            tool_started.set()
            await release_tool.wait()
            return "order 42 shipped"

    async with AgentSession(llm=model, user_away_timeout=0.1) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(ToolAgent(instructions="Answer briefly."))
        reply = session.generate_reply()
        for _ in range(500):
            if model.active_session._reply_futs:
                break
            await asyncio.sleep(0)
        model.active_session._reply_futs[0].set_result(
            _generation(
                response_id="lookup",
                text="checking",
                audio_duration=0.1,
                function_calls=[FunctionCall(call_id="1", name="lookup", arguments="{}")],
            )
        )
        await asyncio.wait_for(tool_started.wait(), timeout=5)
        try:
            await _wait_for_agent_state(session, "thinking")
            filler = session.say("a story first")
            for _ in range(500):
                if model.active_session.say_futs:
                    break
                await asyncio.sleep(0)
            model.active_session.say_futs[0].set_result(
                _generation(response_id="story", text="a story first", audio_duration=0.5)
            )
            await _wait_for_agent_state(session, "speaking")
            states: list[str] = []
            user_states: list[str] = []
            session.on("agent_state_changed", lambda ev: states.append(ev.new_state))
            session.on("user_state_changed", lambda ev: user_states.append(ev.new_state))
            release_tool.set()
            await asyncio.wait_for(filler.wait_for_playout(), timeout=5)

            for _ in range(500):
                if len(model.active_session._reply_futs) == 2:
                    break
                await asyncio.sleep(0)
            assert len(model.active_session._reply_futs) == 2
            assert session.agent_state == "thinking"
            assert "listening" not in states
            await asyncio.sleep(0.2)
            assert "away" not in user_states

            if outcome == "interrupted":
                reply.interrupt()
            else:
                model.active_session._reply_futs[1].set_result(
                    _generation(
                        response_id="answer",
                        text="order 42 shipped" if outcome == "reply" else "",
                        audio_duration=0.1,
                    )
                )
            await asyncio.wait_for(reply.wait_for_playout(), timeout=5)
            await _wait_for_agent_state(session, "listening")
            await asyncio.sleep(0.2)
            assert "away" in user_states
        finally:
            release_tool.set()


@pytest.mark.parametrize("run_mode", ["none", "active", "timed_out"])
@pytest.mark.parametrize("outcome", ["reply", "interrupt", "close"])
async def test_on_enter_completion_preserves_pending_server_reply(
    run_mode: str, outcome: str
) -> None:
    """A hook finishing must not drop busy state, even after the RunResult wait expires."""
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    entered = asyncio.Event()
    release_enter = asyncio.Event()

    class ToolAgent(Agent):
        async def on_enter(self) -> None:
            entered.set()
            await release_enter.wait()

        @function_tool
        async def lookup(self) -> str:
            """Look up an order."""
            return "order 42 shipped"

    async with AgentSession(llm=model, user_away_timeout=0.1) as session:
        session.output.audio = FakeAudioOutput()
        speeches: list[SpeechHandle] = []
        session.on("speech_created", lambda ev: speeches.append(ev.speech_handle))
        await session.start(ToolAgent(instructions="Answer briefly."))
        await asyncio.wait_for(entered.wait(), timeout=5)
        run = session.run(user_input="go") if run_mode != "none" else None
        if run is None:
            session.generate_reply()
        try:
            for _ in range(500):
                if model.active_session._reply_futs:
                    break
                await asyncio.sleep(0)
            model.active_session._reply_futs[0].set_result(
                _generation(
                    response_id="lookup",
                    text="checking",
                    audio_duration=0.1,
                    function_calls=[FunctionCall(call_id="1", name="lookup", arguments="{}")],
                )
            )
            await asyncio.wait_for(speeches[0].wait_for_playout(), timeout=5)
            assert session.agent_state == "thinking"
            activity = session._activity
            assert activity is not None
            if run_mode == "timed_out":
                await asyncio.sleep(5.1)
                assert activity._pending_auto_tool_reply_fut is None
            elif run is not None:
                assert activity._pending_auto_tool_reply_fut is not None
                assert not run.done()

            states: list[str] = []
            user_states: list[str] = []
            session.on("agent_state_changed", lambda ev: states.append(ev.new_state))
            session.on("user_state_changed", lambda ev: user_states.append(ev.new_state))
            release_enter.set()
            assert activity._on_enter_task is not None
            await activity._on_enter_task
            await asyncio.sleep(0.2)
            assert session.agent_state == "thinking"
            assert "listening" not in states
            assert "away" not in user_states

            if outcome == "close":
                await session.aclose()
                assert not activity._awaiting_auto_tool_reply
                assert activity._pending_auto_tool_reply_fut is None
                if run is not None:
                    await asyncio.wait_for(run, timeout=5)
                return
            if outcome == "interrupt":
                await session.interrupt()
            else:
                answer = _generation(
                    response_id="answer", text="order 42 shipped", audio_duration=0.1
                )
                answer.user_initiated = False
                model.active_session.emit("generation_created", answer)
                await asyncio.wait_for(speeches[-1].wait_for_playout(), timeout=5)
            await _wait_for_agent_state(session, "listening")
            if run is not None:
                await asyncio.wait_for(run, timeout=5)
            await asyncio.sleep(0.2)
            assert "away" in user_states
        finally:
            release_enter.set()
