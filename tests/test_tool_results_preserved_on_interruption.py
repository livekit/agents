"""Regression tests for https://github.com/livekit/agents/issues/3702

Completed tool calls/outputs must survive interruption, or the next inference
re-issues the call and duplicates side effects.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, function_tool, utils
from livekit.agents.llm import (
    FunctionCall,
    FunctionToolCall,
    GenerationCreatedEvent,
    MessageGeneration,
)
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.events import FunctionToolsExecutedEvent
from livekit.agents.voice.speech_handle import INTERRUPTION_TIMEOUT, SpeechHandle

from .fake_io import FakeAudioOutput
from .fake_realtime import (
    FakeRealtimeModel,
    _audio_frame,
    fake_capabilities,
    run_realtime_tool_turn,
)
from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60.0


class WeatherAgent(Agent):
    def __init__(self, *, tool_delay: float = 0.0) -> None:
        super().__init__(instructions="You are a helpful assistant.")
        self.tool_delay = tool_delay
        self.tool_executed = asyncio.Event()

    @function_tool
    async def get_weather(self, location: str) -> str:
        """
        Called when the user asks about the weather.

        Args:
            location: The location to get the weather for
        """
        if self.tool_delay > 0.0:
            await asyncio.sleep(self.tool_delay)
        self.tool_executed.set()
        return f"The weather in {location} is sunny today."


def _assert_weather_tool_preserved(agent: Agent, session: AgentSession) -> None:
    for label, items in (
        ("agent chat_ctx", agent.chat_ctx.items),
        ("session history", session.history.items),
    ):
        calls = [i for i in items if i.type == "function_call"]
        outs = [i for i in items if i.type == "function_call_output"]
        assert len(calls) == 1, f"{label}: the tool call must be preserved exactly once"
        assert calls[0].name == "get_weather"
        assert len(outs) == 1, f"{label}: the tool output must be preserved exactly once"
        assert outs[0].output == "The weather in Tokyo is sunny today."
        assert items.index(calls[0]) < items.index(outs[0])


def _weather_tool_turn(actions: FakeActions, *, tts_duration: float) -> None:
    actions.add_user_speech(0.5, 2.5, "What's the weather in Tokyo?")
    actions.add_llm(
        content="Let me check the weather for you.",
        tool_calls=[
            FunctionToolCall(name="get_weather", arguments='{"location": "Tokyo"}', call_id="1")
        ],
    )
    actions.add_tts(tts_duration)


async def test_tool_results_preserved_when_tool_reply_turn_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interruption lands right after the tool reply turn is scheduled: the reply
    returns at an early interruption gate, tool messages must already be committed."""
    actions = FakeActions()
    _weather_tool_turn(actions, tts_duration=1.0)  # playout ~3.5s -> 4.5s
    # tool reply turn, interrupted before it generates anything
    actions.add_llm(
        content="It's sunny in Tokyo!",
        input="The weather in Tokyo is sunny today.",
        ttft=2.0,
        duration=2.5,
    )
    actions.add_tts(2.0)

    session = create_session(actions)
    agent = WeatherAgent()

    # the tool reply turn is the only speech scheduled with force=True;
    # interrupt synchronously right after that scheduling decision
    forced_schedules: list[SpeechHandle] = []
    orig_schedule = AgentActivity._schedule_speech

    def _interrupt_after_forced_schedule(
        self: AgentActivity, speech: SpeechHandle, priority: int, force: bool = False
    ) -> None:
        orig_schedule(self, speech, priority, force=force)
        if force:
            forced_schedules.append(speech)
            speech.interrupt()

    monkeypatch.setattr(AgentActivity, "_schedule_speech", _interrupt_after_forced_schedule)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert forced_schedules, "the tool reply turn was never scheduled; the test needs updating"
    _assert_weather_tool_preserved(agent, session)

    # the interrupted tool reply turn never spoke, its message must not appear
    assistant_texts = [
        i.text_content
        for i in agent.chat_ctx.items
        if i.type == "message" and i.role == "assistant"
    ]
    assert "It's sunny in Tokyo!" not in assistant_texts


async def test_tool_results_preserved_when_interrupted_during_playout() -> None:
    """Completed tool results stay observable when their parent speech is interrupted."""
    actions = FakeActions()
    _weather_tool_turn(actions, tts_duration=10.0)  # playout 3.5s -> 13.5s
    actions.add_user_speech(5.0, 6.0, "Stop!", stt_delay=0.2)  # interrupts at 5.5s
    actions.add_llm(content="Okay, stopping.")
    actions.add_tts(1.0)
    actions.add_tts(1.0, input="The weather in Tokyo is sunny today.")

    session = create_session(actions)
    agent = WeatherAgent()  # the tool completes at ~3.4s, before the interruption
    tool_executed_events: list[FunctionToolsExecutedEvent] = []
    tool_speeches: list[SpeechHandle] = []

    def on_function_tools_executed(event: FunctionToolsExecutedEvent) -> None:
        tool_executed_events.append(event)
        output = event.function_call_outputs[0]
        if output is not None:
            tool_speeches.append(session.say(output.output, allow_interruptions=False))

    session.on("function_tools_executed", on_function_tools_executed)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    _assert_weather_tool_preserved(agent, session)
    assert len(tool_executed_events) == 1
    event = tool_executed_events[0]
    assert event.function_calls[0].name == "get_weather"
    output = event.function_call_outputs[0]
    assert output is not None
    assert output.output == "The weather in Tokyo is sunny today."
    assert len(tool_speeches) == 1
    assert tool_speeches[0].done()
    assert not tool_speeches[0].interrupted
    assert not tool_speeches[0].allow_interruptions


async def test_tool_results_preserved_when_tool_in_flight_at_interruption() -> None:
    """Interruption fires while the tool is still executing: in-flight tools run
    to completion and their results must be committed."""
    actions = FakeActions()
    _weather_tool_turn(actions, tts_duration=10.0)  # playout 3.5s -> 13.5s
    actions.add_user_speech(5.0, 6.0, "Stop!", stt_delay=0.2)  # interrupts at 5.5s
    actions.add_llm(content="Okay, stopping.")
    actions.add_tts(1.0)

    session = create_session(actions)
    # the tool starts at ~3.4s and completes at ~7.4s, well after the interruption
    agent = WeatherAgent(tool_delay=4.0)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    _assert_weather_tool_preserved(agent, session)


async def test_handoff_tool_reports_its_cancellation_when_interrupted() -> None:
    """An interrupted handoff is recorded as failed, and the agent does not switch.

    The tool ran, so dropping its call would let the next inference run it again.
    """

    class TransferAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        @function_tool
        async def transfer_to_billing(self) -> Agent:
            """Transfer the user to the billing department."""
            return Agent(instructions="You are the billing agent.")

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "I have a billing question.")
    actions.add_llm(
        content="Transferring you to billing now.",
        tool_calls=[FunctionToolCall(name="transfer_to_billing", arguments="{}", call_id="1")],
    )
    actions.add_tts(10.0)  # playout 3.5s -> 13.5s
    actions.add_user_speech(5.0, 6.0, "Stop!", stt_delay=0.2)  # interrupts at 5.5s
    actions.add_llm(content="Okay, stopping.")
    actions.add_tts(1.0)

    session = create_session(actions)
    agent = TransferAgent()
    tool_executed_events: list[FunctionToolsExecutedEvent] = []
    session.on("function_tools_executed", tool_executed_events.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert session.current_agent is agent, "the handoff must not be applied"

    assert len(tool_executed_events) == 1
    assert tool_executed_events[0].function_calls[0].name == "transfer_to_billing"
    assert tool_executed_events[0].function_call_outputs[0].is_error

    for label, items in (
        ("agent chat_ctx", agent.chat_ctx.items),
        ("session history", session.history.items),
    ):
        calls = [i for i in items if i.type == "function_call"]
        outs = [i for i in items if i.type == "function_call_output"]
        assert len(calls) == 1, f"{label}: the attempted transfer must be recorded"
        assert len(outs) == 1, f"{label}: the interrupted handoff must be answered once"
        assert outs[0].call_id == calls[0].call_id
        assert outs[0].is_error
        assert not outs[0].reply_required


# --- realtime models -------------------------------------------------------------------
# A realtime model also holds the tool call open server-side. Gemini blocks the session until
# it is answered and offers no way to cancel, so the preserved result is synced to the session
# as well as to the local context (issue #6569).


async def test_realtime_tool_results_preserved_and_synced_when_interrupted() -> None:
    """The result reaches both the local context and the realtime session, wanting no reply."""

    class RealtimeWeatherAgent(WeatherAgent):
        @function_tool
        async def get_weather(self) -> str:
            """Called when the user asks about the weather."""
            self.tool_executed.set()
            return "The weather in Tokyo is sunny today."

    agent = RealtimeWeatherAgent()
    tool_executed_events: list[FunctionToolsExecutedEvent] = []
    session, model = await run_realtime_tool_turn(
        agent,
        tool_executed=agent.tool_executed,
        interrupt=True,
        on_session=lambda s: s.on("function_tools_executed", tool_executed_events.append),
    )

    _assert_weather_tool_preserved(agent, session)
    assert len(tool_executed_events) == 1
    assert tool_executed_events[0].function_calls[0].name == "get_weather"
    assert (
        tool_executed_events[0].function_call_outputs[0].output
        == "The weather in Tokyo is sunny today."
    )

    synced = [i for i in model.active_session.chat_ctx.items if i.type == "function_call_output"]
    assert len(synced) == 1, "the tool output was never synced to the realtime session"
    assert synced[0].call_id == "1"
    assert not synced[0].reply_required


async def test_realtime_handoff_tool_reports_its_cancellation_when_interrupted() -> None:
    """An interrupted handoff is answered as failed, and the agent does not switch.

    The session holds the call open until it is answered, and an empty success would claim a
    transfer that never happened.
    """

    class RealtimeTransferAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")
            self.tool_executed = asyncio.Event()

        @function_tool
        async def transfer_to_billing(self) -> Agent:
            """Transfer the user to the billing department."""
            self.tool_executed.set()
            return Agent(instructions="You are the billing agent.")

    agent = RealtimeTransferAgent()
    session, model = await run_realtime_tool_turn(
        agent, tool_executed=agent.tool_executed, interrupt=True
    )

    assert session.current_agent is agent, "the handoff must not be applied"

    for label, items in (
        ("agent chat_ctx", agent.chat_ctx.items),
        ("session history", session.history.items),
        ("realtime session", model.active_session.chat_ctx.items),
    ):
        outs = [i for i in items if i.type == "function_call_output"]
        assert len(outs) == 1, f"{label}: the interrupted handoff must be answered once"
        assert outs[0].call_id == "1"
        assert outs[0].is_error
        assert not outs[0].reply_required


_INTERRUPTED_INFLIGHT_OUTPUT = "the tool call was interrupted before it finished"


class _SlowRealtimeAgent(Agent):
    def __init__(self, *, tool_sleep: float) -> None:
        super().__init__(instructions="You are a helpful assistant.")
        self.tool_sleep = tool_sleep
        self.tool_started = asyncio.Event()
        self.tool_finished = asyncio.Event()

    @function_tool
    async def get_weather(self) -> str:
        """Called when the user asks about the weather."""
        self.tool_started.set()
        await asyncio.sleep(self.tool_sleep)
        self.tool_finished.set()
        return "The weather in Tokyo is sunny today."


def _fnc_outputs(items: Sequence) -> list:
    return [item for item in items if item.type == "function_call_output"]


@asynccontextmanager
async def _interrupt_while_realtime_tool_runs(
    *, tool_sleep: float
) -> AsyncIterator[tuple[_SlowRealtimeAgent, AgentSession, FakeRealtimeModel]]:
    """Play a realtime turn, then interrupt it once its tool has started."""
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    agent = _SlowRealtimeAgent(tool_sleep=tool_sleep)

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(agent)

        speech = session.generate_reply()
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)

        message_ch = utils.aio.Chan[MessageGeneration]()
        function_ch = utils.aio.Chan[FunctionCall]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities: asyncio.Future[list[str]] = asyncio.Future()
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
        audio_ch.send_nowait(_audio_frame(1.0))
        audio_ch.close()
        function_ch.send_nowait(FunctionCall(call_id="1", name="get_weather", arguments="{}"))
        function_ch.close()
        model.active_session._reply_futs[0].set_result(
            GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="response-id",
            )
        )

        await asyncio.wait_for(agent.tool_started.wait(), timeout=SESSION_TIMEOUT)
        session.interrupt()
        await asyncio.wait_for(speech.wait_for_playout(), timeout=SESSION_TIMEOUT)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + tool_sleep + INTERRUPTION_TIMEOUT
        while loop.time() < deadline and not _fnc_outputs(agent.chat_ctx.items):
            await asyncio.sleep(0.05)
        yield agent, session, model


async def test_realtime_inflight_tool_keeps_its_result_when_it_finishes_in_time() -> None:
    """A tool still running at interrupt time, but done before the interruption timeout,
    keeps the result it returned."""
    async with _interrupt_while_realtime_tool_runs(tool_sleep=1.0) as (agent, session, model):
        assert agent.tool_finished.is_set()
        for label, items in (
            ("agent chat_ctx", agent.chat_ctx.items),
            ("session history", session.history.items),
            ("realtime session", model.active_session.chat_ctx.items),
        ):
            outs = _fnc_outputs(items)
            assert len(outs) == 1, label
            assert outs[0].call_id == "1"
            assert outs[0].output == "The weather in Tokyo is sunny today."
            assert not outs[0].is_error
            assert not outs[0].reply_required


async def test_realtime_tool_running_past_interruption_timeout_is_answered() -> None:
    """A tool still running when INTERRUPTION_TIMEOUT cancels the generation is answered
    with one error output, and a later result from that call is not recorded again.

    GPT Live leaves the call in ``_backend_open_calls`` until that output is synced, and
    refuses every later response while it is open (#7679).
    """
    tool_sleep = INTERRUPTION_TIMEOUT + 10.0
    async with _interrupt_while_realtime_tool_runs(tool_sleep=tool_sleep) as (
        agent,
        session,
        model,
    ):
        assert agent.tool_started.is_set()
        assert not agent.tool_finished.is_set()

        for label, items in (
            ("agent chat_ctx", agent.chat_ctx.items),
            ("session history", session.history.items),
        ):
            calls = [item for item in items if item.type == "function_call"]
            outs = _fnc_outputs(items)
            assert len(calls) == 1, label
            assert calls[0].call_id == "1"
            assert len(outs) == 1, f"{label}: the in-flight call must be answered once"
            assert outs[0].call_id == calls[0].call_id
            assert outs[0].is_error
            assert outs[0].output == _INTERRUPTED_INFLIGHT_OUTPUT
            assert not outs[0].reply_required
            assert items.index(calls[0]) < items.index(outs[0])

        # the fake session stores whatever update_chat_ctx was given; that output is what
        # GPT Live appends to close the call in _backend_open_calls
        synced = _fnc_outputs(model.active_session.chat_ctx.items)
        assert len(synced) == 1, "the error output was never synced to the realtime session"
        assert synced[0].call_id == "1"
        assert synced[0].is_error
        assert synced[0].output == _INTERRUPTED_INFLIGHT_OUTPUT
        assert not synced[0].reply_required

        # the tool's own return must not become a second output
        await asyncio.sleep(tool_sleep)
        assert agent.tool_finished.is_set()
        for label, items in (
            ("agent chat_ctx", agent.chat_ctx.items),
            ("session history", session.history.items),
            ("realtime session", model.active_session.chat_ctx.items),
        ):
            outs = _fnc_outputs(items)
            assert len(outs) == 1, f"{label}: the late result must not be recorded again"
            assert outs[0].output == _INTERRUPTED_INFLIGHT_OUTPUT

        # the activity can still ask the model to speak
        before = model.active_session.generate_reply_calls
        follow_up = session.generate_reply(user_input="Thanks")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + INTERRUPTION_TIMEOUT
        while model.active_session.generate_reply_calls == before and loop.time() < deadline:
            await asyncio.sleep(0.05)
        assert model.active_session.generate_reply_calls == before + 1

        message_ch = utils.aio.Chan[MessageGeneration]()
        function_ch = utils.aio.Chan[FunctionCall]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities: asyncio.Future[list[str]] = asyncio.Future()
        modalities.set_result(["audio", "text"])
        message_ch.send_nowait(
            MessageGeneration(
                message_id="follow-up",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        text_ch.send_nowait("You're welcome.")
        text_ch.close()
        audio_ch.send_nowait(_audio_frame(0.2))
        audio_ch.close()
        function_ch.close()
        model.active_session._reply_futs[-1].set_result(
            GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="follow-up",
            )
        )
        await asyncio.wait_for(follow_up.wait_for_playout(), timeout=SESSION_TIMEOUT)
        assert any(
            item.type == "message"
            and item.role == "assistant"
            and "welcome" in (item.text_content or "")
            for item in agent.chat_ctx.items
        )
