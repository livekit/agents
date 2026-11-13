"""Tool calls and their replies inside a session: interruption, timeouts and deferred replies."""

from __future__ import annotations

import asyncio
import logging

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool, llm, utils
from livekit.agents.llm import FunctionToolCall
from livekit.agents.llm.async_toolset import AsyncToolset
from livekit.agents.llm.chat_context import ChatContext, FunctionCall, FunctionCallOutput
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.events import FunctionToolsExecutedEvent, SpeechCreatedEvent
from livekit.agents.voice.generation import (
    _RUNNING_PLACEHOLDER_KEY,
    _inject_running_tool_calls,
    _strip_running_tool_calls,
)
from livekit.agents.voice.speech_handle import INTERRUPTION_TIMEOUT, SpeechHandle

from .fake_io import FakeAudioOutput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_realtime import (
    FakeRealtimeModel,
    _audio_frame,
    fake_capabilities,
    run_realtime_tool_turn,
)
from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60.0
TOOL_DURATION = INTERRUPTION_TIMEOUT + 3
WEATHER = "The weather in Tokyo is sunny today."


def _outputs(chat_ctx: llm.ChatContext, call_id: str) -> list[llm.FunctionCallOutput]:
    return [
        item
        for item in chat_ctx.items
        if item.type == "function_call_output" and item.call_id == call_id
    ]


# --- completed tool results survive an interruption ------------------------------------


class WeatherAgent(Agent):
    def __init__(self, *, tool_delay: float = 0.0) -> None:
        super().__init__(instructions="You are a helpful assistant.")
        self.tool_delay = tool_delay
        self.tool_executed = asyncio.Event()

    @function_tool
    async def get_weather(self, location: str) -> str:
        """Called when the user asks about the weather.

        Args:
            location: The location to get the weather for
        """
        if self.tool_delay:
            await asyncio.sleep(self.tool_delay)
        self.tool_executed.set()
        return WEATHER


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
        assert outs[0].output == WEATHER
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
    """Tool messages are committed before the tool reply turn hits its interruption gate."""
    actions = FakeActions()
    _weather_tool_turn(actions, tts_duration=1.0)
    actions.add_llm(content="It's sunny in Tokyo!", input=WEATHER, ttft=2.0, duration=2.5)
    actions.add_tts(2.0)

    session = create_session(actions)
    agent = WeatherAgent()

    # the tool reply turn is the only speech scheduled with force=True
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

    assert forced_schedules, "the tool reply turn was never scheduled"
    _assert_weather_tool_preserved(agent, session)
    assistant_texts = [
        i.text_content
        for i in agent.chat_ctx.items
        if i.type == "message" and i.role == "assistant"
    ]
    assert "It's sunny in Tokyo!" not in assistant_texts


async def test_tool_results_preserved_when_interrupted_during_playout() -> None:
    actions = FakeActions()
    _weather_tool_turn(actions, tts_duration=10.0)  # playout 3.5s -> 13.5s
    actions.add_user_speech(5.0, 6.0, "Stop!", stt_delay=0.2)  # interrupts at 5.5s
    actions.add_llm(content="Okay, stopping.")
    actions.add_tts(1.0)
    actions.add_tts(1.0, input=WEATHER)

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
    [event] = tool_executed_events
    assert event.function_calls[0].name == "get_weather"
    assert event.function_call_outputs[0] is not None
    assert event.function_call_outputs[0].output == WEATHER
    [speech] = tool_speeches
    assert speech.done()
    assert not speech.interrupted


async def test_tool_results_preserved_when_tool_in_flight_at_interruption() -> None:
    actions = FakeActions()
    _weather_tool_turn(actions, tts_duration=10.0)
    actions.add_user_speech(5.0, 6.0, "Stop!", stt_delay=0.2)
    actions.add_llm(content="Okay, stopping.")
    actions.add_tts(1.0)

    session = create_session(actions)
    # the tool completes at ~7.4s, after the interruption
    agent = WeatherAgent(tool_delay=4.0)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    _assert_weather_tool_preserved(agent, session)


async def test_handoff_tool_reports_its_cancellation_when_interrupted() -> None:
    """An interrupted handoff is recorded as failed, and the agent does not switch."""

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
    actions.add_tts(10.0)
    actions.add_user_speech(5.0, 6.0, "Stop!", stt_delay=0.2)
    actions.add_llm(content="Okay, stopping.")
    actions.add_tts(1.0)

    session = create_session(actions)
    agent = TransferAgent()
    tool_executed_events: list[FunctionToolsExecutedEvent] = []
    session.on("function_tools_executed", tool_executed_events.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert session.current_agent is agent
    [event] = tool_executed_events
    assert event.function_calls[0].name == "transfer_to_billing"
    assert event.function_call_outputs[0].is_error

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


# a realtime model holds the call open server-side (Gemini blocks until it is answered), so the
# preserved result is synced to the realtime session as well as to the local context


async def test_realtime_tool_results_preserved_and_synced_when_interrupted() -> None:
    class RealtimeWeatherAgent(WeatherAgent):
        @function_tool
        async def get_weather(self) -> str:
            """Called when the user asks about the weather."""
            self.tool_executed.set()
            return WEATHER

    agent = RealtimeWeatherAgent()
    tool_executed_events: list[FunctionToolsExecutedEvent] = []
    session, model = await run_realtime_tool_turn(
        agent,
        tool_executed=agent.tool_executed,
        interrupt=True,
        on_session=lambda s: s.on("function_tools_executed", tool_executed_events.append),
    )

    _assert_weather_tool_preserved(agent, session)
    [event] = tool_executed_events
    assert event.function_call_outputs[0].output == WEATHER

    synced = [i for i in model.active_session.chat_ctx.items if i.type == "function_call_output"]
    assert len(synced) == 1, "the tool output was never synced to the realtime session"
    assert synced[0].call_id == "1"
    assert not synced[0].reply_required


async def test_realtime_handoff_tool_reports_its_cancellation_when_interrupted() -> None:
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

    assert session.current_agent is agent
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


# --- the interruption timeout releases a tool still running --------------------------


def _start_realtime_tool_reply(rt_session: object, *, with_message: bool) -> None:
    message_ch = utils.aio.Chan[llm.MessageGeneration]()
    function_ch = utils.aio.Chan[llm.FunctionCall]()
    if with_message:
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["audio", "text"])
        message_ch.send_nowait(
            llm.MessageGeneration(
                message_id="message-id",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        text_ch.send_nowait("let me check")
        text_ch.close()
        audio_ch.send_nowait(_audio_frame(1.0))
        audio_ch.close()
    message_ch.close()
    function_ch.send_nowait(llm.FunctionCall(call_id="1", name="lookup", arguments="{}"))
    function_ch.close()
    rt_session._reply_futs[0].set_result(  # type: ignore[attr-defined]
        llm.GenerationCreatedEvent(
            message_stream=message_ch,
            function_stream=function_ch,
            user_initiated=True,
            response_id="response-id",
        )
    )


async def test_interrupt_timeout_releases_the_running_tool() -> None:
    """The timeout answers the open call with the running placeholder; the result follows later."""
    tool_started = asyncio.Event()

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup(self) -> str:
            """Look something up."""
            tool_started.set()
            await asyncio.sleep(TOOL_DURATION)
            return "the fee is 10 dollars"

    model = FakeRealtimeModel(capabilities=fake_capabilities())

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput(can_pause=True)
        await session.start(ToolAgent())
        rt_session = model.active_session

        handle = session.generate_reply()
        while not rt_session._reply_futs:
            await asyncio.sleep(0)
        _start_realtime_tool_reply(rt_session, with_message=True)

        await asyncio.wait_for(tool_started.wait(), timeout=5)
        handle.interrupt(force=True)
        await asyncio.sleep(INTERRUPTION_TIMEOUT + 0.5)

        assert handle.done()
        [output] = _outputs(rt_session.chat_ctx, "1")
        assert not output.is_error
        assert "still in progress" in output.output

        await asyncio.sleep(TOOL_DURATION)
        [final] = _outputs(rt_session.chat_ctx, "1_final")
        assert "the fee is 10 dollars" in final.output


@pytest.mark.parametrize("interrupt_during_wait", [False, True])
async def test_interrupt_timeout_keeps_the_result_of_a_returned_tool(
    interrupt_during_wait: bool,
) -> None:
    """An interrupted speech sends a returned tool's result without waiting for the floor."""

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup(self) -> str:
            """Look something up."""
            await asyncio.sleep(2.0)
            return "the fee is 10 dollars"

    model = FakeRealtimeModel(capabilities=fake_capabilities())

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(ToolAgent())
        rt_session = model.active_session

        handle = session.generate_reply()
        while not rt_session._reply_futs:
            await asyncio.sleep(0)
        _start_realtime_tool_reply(rt_session, with_message=False)

        # the turn has played out and waits for the tool in the background
        await asyncio.sleep(0.5)
        if not interrupt_during_wait:
            session.interrupt()
        # another speech is current when the tool returns, and still is at the timeout
        session.generate_reply()
        if interrupt_during_wait:
            # the tool has returned and the speech already waits for the floor
            await asyncio.sleep(2.0)
            handle.interrupt()
        await asyncio.sleep(INTERRUPTION_TIMEOUT + 0.5)

        assert handle.done()
        [output] = _outputs(rt_session.chat_ctx, "1")
        assert "the fee is 10 dollars" in output.output


async def test_interrupt_timeout_releases_the_running_pipeline_tool() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Look up my order")
    actions.add_llm(
        content="Let me check.",
        tool_calls=[FunctionToolCall(name="lookup_order", arguments="{}", call_id="1")],
    )
    actions.add_tts(4.0)

    class SlowToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup_order(self) -> str:
            """Look up the order."""
            await asyncio.sleep(TOOL_DURATION)
            return "order 42 shipped"

    session = create_session(actions)
    agent = SlowToolAgent()

    def _interrupt_while_speaking(ev: object) -> None:
        # only the first reply, so the deferred reply that carries the result plays out
        if getattr(ev, "new_state", None) == "speaking":
            session.off("agent_state_changed", _interrupt_while_speaking)
            asyncio.get_running_loop().call_later(0.5, lambda: session.interrupt(force=True))

    session.on("agent_state_changed", _interrupt_while_speaking)
    await asyncio.wait_for(run_session(session, agent, drain_delay=TOOL_DURATION + 5), timeout=60)

    # the placeholder answers the call in history, so the model does not issue it again
    [output] = _outputs(agent.chat_ctx, "1")
    assert "still in progress" in output.output
    [final] = _outputs(agent.chat_ctx, "1_final")
    assert "order 42 shipped" in final.output


# --- running tool placeholders in the LLM context -----------------------------------


def test_inject_adds_flagged_pair_for_inflight_call() -> None:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="book me a flight")

    running = FunctionCall(call_id="call_1", name="book_flight", arguments="{}", created_at=1.0)
    _inject_running_tool_calls(chat_ctx, [running])

    [call] = [i for i in chat_ctx.items if i.type == "function_call"]
    [out] = [i for i in chat_ctx.items if i.type == "function_call_output"]
    assert call.extra.get(_RUNNING_PLACEHOLDER_KEY) is True
    assert out.output and out.is_error is False
    # ids of their own, the same on every turn the call is still running
    assert (call.id, out.id) == (f"{running.id}_running", f"{running.id}_running_output")
    assert chat_ctx.items.index(call) < chat_ctx.items.index(out)
    # the flag goes on a copy; the live FunctionCall stays clean
    assert _RUNNING_PLACEHOLDER_KEY not in running.extra


def test_inject_skips_call_already_present() -> None:
    chat_ctx = ChatContext.empty()
    chat_ctx.insert(FunctionCall(call_id="call_1", name="book_flight", arguments="{}"))
    chat_ctx.insert(
        FunctionCallOutput(call_id="call_1", name="book_flight", output="booked", is_error=False)
    )

    _inject_running_tool_calls(
        chat_ctx, [FunctionCall(call_id="call_1", name="book_flight", arguments="{}")]
    )

    [out] = [i for i in chat_ctx.items if i.type == "function_call_output"]
    assert out.output == "booked"


def test_strip_removes_injected_pairs_only() -> None:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")
    chat_ctx.insert(FunctionCall(call_id="real", name="t", arguments="{}"))
    chat_ctx.insert(FunctionCallOutput(call_id="real", name="t", output="done", is_error=False))

    _inject_running_tool_calls(chat_ctx, [FunctionCall(call_id="run", name="t2", arguments="{}")])
    # llm_node may add items between the inject and the strip
    chat_ctx.add_message(role="system", content="extra context from llm_node")
    _strip_running_tool_calls(chat_ctx)

    assert [getattr(i, "call_id", i.type) for i in chat_ctx.items] == [
        "message",
        "real",
        "real",
        "message",
    ]


# --- async tool replies that land while the agent changes ----------------------------

# how long the async tool keeps working after its update; its result lands while the
# agent that called it is still switching away
TOOL_WORK = 0.5


@function_tool
async def send_link(ctx: RunContext) -> str:
    """Send a link."""
    await ctx.update("sending the link")
    await asyncio.sleep(TOOL_WORK)
    return "link sent"


class _Second(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="second agent")


class _First(Agent):
    def __init__(self, *, with_send_link: bool) -> None:
        super().__init__(instructions="first agent", tools=[send_link] if with_send_link else [])

    @function_tool
    async def handoff(self) -> Agent:
        """Hand off to the second agent."""
        await asyncio.sleep(0.05)  # finish after send_link's update, so the handoff is kept
        return _Second()


class _WaitTask(AgentTask[None]):
    def __init__(self) -> None:
        super().__init__(instructions="wait task")

    async def on_enter(self) -> None:
        await asyncio.sleep(TOOL_WORK * 2)
        self.complete(None)


class _Parent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="parent agent", tools=[send_link])

    @function_tool
    async def collect(self) -> str:
        """Run a task."""
        await _WaitTask()
        return "collected"


class _AskTask(AgentTask[None]):
    def __init__(self) -> None:
        super().__init__(instructions="ask task")

    async def on_enter(self) -> None:
        await asyncio.sleep(TOOL_WORK * 2)  # keeps the first run open past send_link's work

    @function_tool
    async def finish(self) -> None:
        """Finish the task."""
        self.complete(None)


class _Asker(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="asker agent", tools=[send_link])

    @function_tool
    async def ask(self) -> str:
        """Ask the user."""
        await _AskTask()
        return "asked"


def _tool_outputs(agent: Agent) -> list[str]:
    return [i.output for i in agent.chat_ctx.items if i.type == "function_call_output"]


def _send_link_with(other_tool: str) -> list[FunctionToolCall]:
    return [
        FunctionToolCall(name=name, arguments="{}", call_id=f"call_{name}")
        for name in ("send_link", other_tool)
    ]


def _track_speakers(sess: AgentSession) -> list[str]:
    speakers: list[str] = []

    def _on_speech_created(ev: SpeechCreatedEvent) -> None:
        if ev.source == "generate_reply":
            speakers.append(type(sess.current_agent).__name__)

    sess.on("speech_created", _on_speech_created)
    return speakers


async def _run(
    agent: Agent,
    other_tool: str,
    caplog: pytest.LogCaptureFixture,
    *,
    session_tools: list[AsyncToolset] | None = None,
) -> tuple[list[str], Agent]:
    """Run one turn calling ``send_link`` alongside ``other_tool``; return speakers and the final agent."""
    fake_llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="one moment",
                ttft=0.1,
                duration=0.1,
                tool_calls=_send_link_with(other_tool),
            ),
            # the draining agent's tool reply outlasts the async tool's work
            FakeLLMResponse(input="", content="checking", ttft=0.1, duration=TOOL_WORK * 2),
        ]
    )
    async with AgentSession(llm=fake_llm, tools=session_tools or []) as sess:
        speakers = _track_speakers(sess)
        await sess.start(agent)

        with caplog.at_level(logging.DEBUG, logger="livekit.agents"):
            await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)
            await asyncio.sleep(TOOL_WORK * 4)

        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert not errors, errors
        return speakers, sess.current_agent


async def test_reply_during_handoff_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    """An activity-scoped async tool that finishes while its agent drains must not raise."""
    first = _First(with_send_link=True)
    speakers, current = await _run(first, "handoff", caplog)
    assert speakers == ["_First"]  # only the user turn
    assert isinstance(current, _Second)
    assert "link sent" in _tool_outputs(first)
    assert any("owning activity closed" in r.getMessage() for r in caplog.records)


async def test_reply_during_agent_task_waits_for_resume(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A reply that lands under an AgentTask is spoken by its own agent once the task returns."""
    parent = _Parent()
    speakers, current = await _run(parent, "collect", caplog)
    assert speakers == ["_Parent", "_Parent"]  # the user turn, then the deferred reply
    assert current is parent
    assert "link sent" in _tool_outputs(parent)


async def test_reply_under_agent_task_does_not_block_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A reply waiting on an AgentTask that needs another user turn does not hold the run open."""
    finish = [FunctionToolCall(name="finish", arguments="{}", call_id="call_finish")]
    fake_llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="one moment",
                ttft=0.1,
                duration=0.1,
                tool_calls=_send_link_with("ask"),
            ),
            FakeLLMResponse(input="done", content="", ttft=0.1, duration=0.1, tool_calls=finish),
        ]
    )
    asker = _Asker()
    async with AgentSession(llm=fake_llm) as sess:
        speakers = _track_speakers(sess)
        await sess.start(asker)

        with caplog.at_level(logging.DEBUG, logger="livekit.agents"):
            await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)
            assert isinstance(sess.current_agent, _AskTask)
            await asyncio.wait_for(sess.run(user_input="done"), timeout=5.0)

        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert not errors, errors
        assert sess.current_agent is asker
        assert speakers[-1] == "_Asker"  # the deferred reply, after the task returns
        assert "link sent" in _tool_outputs(asker)


async def test_session_scoped_reply_follows_handoff(caplog: pytest.LogCaptureFixture) -> None:
    """A session-scoped async tool that finishes during a handoff is spoken by the next agent."""
    first = _First(with_send_link=False)
    toolset = AsyncToolset(id="links", tools=[send_link])
    speakers, current = await _run(first, "handoff", caplog, session_tools=[toolset])
    assert isinstance(current, _Second)
    assert speakers == ["_First", "_Second"]
    assert "link sent" in _tool_outputs(current)


async def test_session_scoped_reply_is_dropped_on_close() -> None:
    """A session-scoped reply still pending at close is dropped, and the close completes."""
    calls = [FunctionToolCall(name="send_link", arguments="{}", call_id="call_send_link")]
    fake_llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go", content="one moment", ttft=0.1, duration=0.1, tool_calls=calls
            ),
            # keeps the agent busy past the tool's work, so the reply waits for idle
            FakeLLMResponse(input="busy", content="still going", ttft=0.1, duration=10.0),
        ]
    )
    toolset = AsyncToolset(id="links", tools=[send_link])
    sess = AgentSession(llm=fake_llm, tools=[toolset])
    await sess.start(_First(with_send_link=False))
    await sess.run(user_input="go")
    sess.generate_reply(user_input="busy")
    await asyncio.sleep(TOOL_WORK * 2)
    assert toolset._executor._reply_task is not None
    assert not toolset._executor._reply_task.done()

    await asyncio.wait_for(sess.aclose(), timeout=5.0)
