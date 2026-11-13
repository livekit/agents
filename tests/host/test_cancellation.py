"""Cancellation across the native boundary, both ways: a cancelled Python caller cancels the
native run and the Python work it drives, the core cancelling or abandoning Python work closes
it, and both sides settle before the caller resumes."""

from __future__ import annotations

import asyncio
import gc
from collections.abc import AsyncIterator

import pytest
from opentelemetry import trace
from pydantic import BaseModel

from livekit.agents import (
    Agent,
    APIConnectOptions,
    APIError,
    ModelSettings,
    RunContext,
    TextSession,
    VoiceSession,
    function_tool,
    llm,
)
from livekit.agents.a2a import RemoteAgent
from livekit.agents.llm import ChatContext, Toolset
from livekit.agents.types import FlushSentinel
from livekit.agents.utils import aio
from livekit.agents.voice.run_result import StreamEvent
from tests.fakes import FakeAudioOutput, FakeLLM, FakeTTS, Hang, audio_frame, tool_call, until

from .a2a_peer import Peer

# -- a cancelled Python caller cancels the native run


async def test_a_run_cancelled_in_its_instructions_never_runs_the_tool() -> None:
    """Cancelling `agent.run` while an instructions callback awaits runs that callback's
    cleanup and starts nothing after it."""
    entered, cleaned = asyncio.Event(), asyncio.Event()
    ran: list[str] = []
    agent = Agent(instructions="Help.", llm=FakeLLM([tool_call("effect")]))

    @agent.instructions
    async def prepare(ctx: RunContext) -> str:
        entered.set()
        try:
            await asyncio.Future[None]()
        finally:
            cleaned.set()
        return "unreachable"

    @agent.tool
    async def effect(ctx: RunContext) -> str:
        """Perform an effect."""
        ran.append("effect")
        return "done"

    run = asyncio.create_task(agent.run("Act"))
    await entered.wait()
    run.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run
    assert cleaned.is_set()
    assert ran == []


async def _cancelled_during_tool(*, return_late_result: bool) -> list[StreamEvent]:
    """Every event a `run_stream` streamed when cancelled while its tool body waits."""
    entered, cleaned = asyncio.Event(), asyncio.Event()
    # a reply is scripted after the tool so that a run that wrongly continues streams it
    agent = Agent(instructions="Help.", llm=FakeLLM([tool_call("wait", call_id="c1"), "late"]))

    @agent.tool
    async def wait(ctx: RunContext) -> str:
        """Wait for the service."""
        entered.set()
        try:
            await asyncio.Future[None]()
        except asyncio.CancelledError:
            if return_late_result:
                return "late result"
            raise
        finally:
            cleaned.set()
        return "unreachable"

    seen: list[StreamEvent] = []
    async with agent.run_stream("Wait") as stream:
        events = stream.stream_events()
        seen.append(await anext(events))
        await entered.wait()
        stream.cancel()
        with pytest.raises(asyncio.CancelledError):
            async for event in events:
                seen.append(event)
        with pytest.raises(asyncio.CancelledError):
            await stream.result()
    assert cleaned.is_set()
    return seen


@pytest.mark.parametrize("return_late_result", [False, True])
async def test_a_stream_cancelled_during_a_tool_settles_it_and_replies_no_more(
    return_late_result: bool,
) -> None:
    """Cancelling during a tool settles its body, and no step follows the call, even when the
    body returns a result after its cancellation."""
    seen = await _cancelled_during_tool(return_late_result=return_late_result)

    assert [(event.type, getattr(event, "call_id", None)) for event in seen] == [
        ("tool_call", "c1")
    ]


@pytest.mark.xfail(
    strict=True,
    reason="docs/agents.md: a cancelled call records a tool_result with outcome cancelled; none is emitted",
)
@pytest.mark.parametrize("return_late_result", [False, True])
async def test_a_tool_cancelled_mid_call_records_a_cancelled_outcome(
    return_late_result: bool,
) -> None:
    """An unfinished cancelled call streams a cancelled tool_result (docs/agents.md)."""
    seen = await _cancelled_during_tool(return_late_result=return_late_result)

    assert [(e.type, getattr(e, "call_id", None), getattr(e, "outcome", None)) for e in seen] == [
        ("tool_call", "c1", None),
        ("tool_result", "c1", "cancelled"),
    ]


async def test_leaving_run_stream_with_its_iterator_unread_cancels_the_run() -> None:
    """Exiting the `run_stream` block mid-run cancels the tool body; the run's result and its
    abandoned iterator both report the cancellation."""
    entered, cancelled = asyncio.Event(), asyncio.Event()
    agent = Agent(instructions="Help.", llm=FakeLLM([tool_call("wait"), "Done"]))

    @agent.tool
    async def wait(ctx: RunContext) -> str:
        """Wait for the application."""
        entered.set()
        try:
            await asyncio.Future[None]()
        finally:
            cancelled.set()
        return "unreachable"

    async with agent.run_stream("start") as stream:
        events = stream.stream_events()
        await anext(events)
        await entered.wait()

    assert cancelled.is_set()
    with pytest.raises(asyncio.CancelledError):
        await stream.result()
    with pytest.raises(asyncio.CancelledError):
        await anext(events)


async def test_a_second_concurrent_reader_is_refused_and_a_cancelled_read_does_not_cancel_the_run() -> (
    None
):
    """A stream has one reader: a second concurrent read is refused, and cancelling a pending
    read leaves the run going for the next read."""
    entered, release = asyncio.Event(), asyncio.Event()
    agent = Agent(instructions="Help.", llm=FakeLLM([tool_call("wait"), "Done"]))

    @agent.tool
    async def wait(ctx: RunContext) -> str:
        """Wait for the application."""
        entered.set()
        await release.wait()
        return "ready"

    async with agent.run_stream("start") as stream:
        events = stream.stream_events()
        await anext(events)
        await entered.wait()
        pending = asyncio.ensure_future(anext(events))
        await asyncio.sleep(0)  # the first read is waiting in the native stream

        with pytest.raises(RuntimeError, match="another consumer"):
            await anext(events)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release.set()
        async for _ in events:
            pass
        result = await stream.result()

    assert result.status == "completed" and result.output == "Done"


async def test_cancelling_a_parent_run_closes_its_childs_model_stream() -> None:
    """Cancelling a run cancels the child agent its tool delegated to, down to the child's
    Python LLM stream."""
    hang = Hang()

    class Booking(BaseModel):
        reference: str

    child = Agent(name="booking", instructions="Book.", output_type=Booking, llm=FakeLLM([hang]))
    parent = Agent(
        instructions="Delegate bookings.",
        tools=[child.as_tool()],
        llm=FakeLLM([tool_call("booking")]),
    )

    run = asyncio.create_task(parent.run("One night"))
    await hang.started.wait()
    run.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run
    assert hang.ended.is_set()


async def test_cancelling_a_delegating_run_cancels_the_remote_task(peer: Peer) -> None:
    """Cancelling a run whose tool waits on a remote A2A agent sends that agent `CancelTask`."""
    peer.hold = True
    remote = RemoteAgent(
        peer.url,
        name="coverage_expert",
        description="Checks account coverage.",
        conn_options=APIConnectOptions(max_retry=0),
    )
    agent = Agent(
        instructions="Ask coverage_expert.",
        tools=[remote.as_tool()],
        llm=FakeLLM([tool_call("coverage_expert", request="Check")]),
    )

    run = asyncio.create_task(agent.run("Check"))
    await peer.received.wait()
    await asyncio.sleep(0.05)  # the peer's working status reached the client
    run.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run
    await until(lambda: peer.cancelled, "the remote task was never cancelled")
    assert peer.cancelled == ["remote-task"]


async def test_cancelling_a_toolset_setup_waits_for_its_cleanup() -> None:
    """A run cancelled during a toolset's setup returns only after that setup's `finally` has
    finished."""
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    finished: list[bool] = []

    class Resource(Toolset):
        async def setup(self) -> Resource:
            entered.set()
            try:
                await asyncio.Future[None]()
            finally:
                cleaning.set()
                await release.wait()
                finished.append(True)
            return self

    agent = Agent(instructions="Help.", tools=[Resource(id="resource")], llm=FakeLLM(["Hi"]))
    run = asyncio.create_task(agent.run("Hello"))
    await entered.wait()
    run.cancel()
    await cleaning.wait()

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(run), 0.05)  # held by the authored cleanup
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert finished == [True]


async def test_a_cancelled_session_start_can_be_joined_by_the_next_start() -> None:
    """Cancelling a `start()` caller leaves the startup running: the next `start()` joins it
    instead of setting the session up twice."""
    entered, release = asyncio.Event(), asyncio.Event()
    setups: list[str] = []

    class Slow(Toolset):
        async def setup(self) -> Slow:
            setups.append(self.id)
            entered.set()
            await release.wait()
            return await super().setup()

    session = VoiceSession(vad=None, tools=[Slow(id="slow")])
    agent = Agent(instructions="Welcome the caller.")
    first = asyncio.create_task(session.start(agent))
    await entered.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    second = asyncio.create_task(session.start(agent))
    release.set()
    await second
    await session.aclose()

    assert session.current_agent is agent
    assert setups == ["slow"]


async def test_a_failed_turn_leaves_the_saved_state_usable() -> None:
    """A `TextSession` turn that fails does not spoil the state it started from: the next turn
    continues the conversation from it."""
    model = FakeLLM(["Ready.", APIError("scripted failure", retryable=False), "Again."])
    agent = Agent(name="desk", instructions="Help.", llm=model)
    first = await TextSession(agent=agent).run("Hi")

    with pytest.raises(APIError):
        await TextSession(agent=agent).run("Fail", state=first.state)
    retried = await TextSession(agent=agent).run("Retry", state=first.state)

    assert retried.result.status == "completed" and retried.result.output == "Again."
    said = [item.text_content for item in model.requests[-1].items if item.type == "message"]
    assert said[-3:] == ["Hi", "Ready.", "Retry"]


async def test_the_callers_span_is_restored_after_a_cancelled_run() -> None:
    """Callbacks run in their own OpenTelemetry context: cancelling a run mid-tool leaves the
    caller's current span unchanged."""
    entered = asyncio.Event()
    agent = Agent(instructions="Help.", llm=FakeLLM([tool_call("lookup")]))

    @agent.tool
    async def lookup(ctx: RunContext) -> str:
        """Look up the customer's record."""
        entered.set()
        await asyncio.Future[None]()
        return "unreachable"

    caller = trace.NonRecordingSpan(
        trace.SpanContext(303, 33, is_remote=False, trace_flags=trace.TraceFlags(1))
    )
    with trace.use_span(caller):
        async with agent.run_stream("begin") as stream:
            await entered.wait()
            stream.cancel()

        assert trace.get_current_span() is caller


# -- the toolsets a run opens


async def test_concurrent_runs_share_one_toolset_setup_and_close() -> None:
    """A toolset is set up when the first run needs it and closed after the last one ends, even
    a run left early (tools.md "Toolsets")."""
    calls: list[str] = []
    lifecycle: list[str] = []
    release = asyncio.Event()

    class Counting(Toolset):
        async def setup(self) -> Counting:
            lifecycle.append("setup")
            return await super().setup()

        async def aclose(self) -> None:
            lifecycle.append("close")
            await super().aclose()

        @function_tool
        async def wait(self, ctx: RunContext) -> str:
            """Wait until released."""
            calls.append("wait")
            await release.wait()
            return "done"

    model = FakeLLM({"go": tool_call("wait"), "done": "finished"})
    agent = Agent(instructions="Help.", llm=model, tools=[Counting(id="counting")])

    runs = [asyncio.create_task(agent.run("go")) for _ in range(2)]
    await until(lambda: len(calls) == 2, "both runs called the tool")
    release.set()
    results = await asyncio.gather(*runs)

    assert all(r.status == "completed" and r.output == "finished" for r in results)
    assert lifecycle == ["setup", "close"]


@pytest.mark.xfail(
    strict=True,
    reason="the agent's toolsets close in a task the session close does not join "
    "(lk-agents-runtime voice/session/host_drain.rs close_toolsets_later)",
)
async def test_closing_a_session_waits_for_its_agents_nested_toolset_to_close() -> None:
    """`VoiceSession.aclose()` returns only after the active agent's toolsets, nested ones
    included, have finished closing (voice-session.md: "end the call's session and wait for it")."""
    closed: list[str] = []

    class Inner(Toolset):
        async def aclose(self) -> None:
            await asyncio.sleep(0.05)  # a slow disconnect
            closed.append(self.id)
            await super().aclose()

    agent = Agent(instructions="Help.", tools=[Toolset(id="outer", tools=[Inner(id="inner")])])
    session = VoiceSession(vad=None)
    await session.start(agent)

    await session.aclose()

    assert closed == ["inner"]


# -- the core cancels or abandons Python work it drives


async def test_closing_a_python_llm_stream_cancels_its_provider_and_ends_the_reader() -> None:
    """`aclose()` on a Python LLM's stream cancels the provider's work, and a reader waiting on
    the stream sees its end."""
    hang = Hang()
    stream = FakeLLM([hang]).chat(chat_ctx=ChatContext.empty())
    reader = asyncio.create_task(anext(stream))
    await hang.started.wait()

    await stream.aclose()

    assert hang.ended.is_set()
    with pytest.raises(StopAsyncIteration):
        await reader


async def test_an_interruption_cancels_the_python_llm_and_closes_the_node_generator() -> None:
    """Interrupting a reply cancels the Python LLM it is waiting on and closes the generator
    the session's `llm_node` returned."""
    hang = Hang()
    closed = asyncio.Event()
    session = VoiceSession(vad=None, llm=FakeLLM([["Let me check.", hang]]), tts=FakeTTS())
    session.output.audio = FakeAudioOutput()

    @session.llm_node
    async def think(
        ctx: RunContext,
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterator[llm.ChatChunk | str | FlushSentinel]:
        try:
            async for chunk in session.default.llm_node(ctx, chat_ctx, tools, model_settings):
                yield chunk
        finally:
            closed.set()

    async with session:
        await session.start(Agent(instructions="Help."))
        speech = session.generate_reply(user_input="hello")
        await hang.started.wait()
        await session.interrupt()
        await speech

    assert speech.interrupted
    assert hang.ended.is_set()
    assert closed.is_set()


async def test_closing_the_default_llm_node_returns_once_the_python_llm_ended() -> None:
    """`aclose()` on the chunks `session.default.llm_node` returned ends the Python LLM's stream
    before it returns, so the node's `finally` leaves no request running."""
    hang = Hang()
    ended_at_close: list[bool] = []
    session = VoiceSession(vad=None, llm=FakeLLM([["Let me check.", hang]]), tts=FakeTTS())
    session.output.audio = FakeAudioOutput()

    @session.llm_node
    async def think(
        ctx: RunContext,
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterator[llm.ChatChunk | str | FlushSentinel]:
        chunks = session.default.llm_node(ctx, chat_ctx, tools, model_settings)
        try:
            async for chunk in chunks:
                yield chunk
        finally:
            await chunks.aclose()  # type: ignore[attr-defined]
            ended_at_close.append(hang.ended.is_set())

    async with session:
        await session.start(Agent(instructions="Help."))
        speech = session.generate_reply(user_input="hello")
        await hang.started.wait()
        await session.interrupt()
        await speech

    assert ended_at_close == [True]


async def test_a_session_close_cancels_the_reply_and_runs_its_done_callbacks() -> None:
    """Closing the session mid-reply cancels the Python LLM and still tells the host the
    speech is done."""
    hang = Hang()
    done: list[bool] = []

    async with VoiceSession(llm=FakeLLM([hang])) as session:
        await session.start(Agent(instructions="Help."))
        speech = session.generate_reply(user_input="hello")
        speech.add_done_callback(lambda handle: done.append(handle.done()))
        await hang.started.wait()
    await until(lambda: done, "the close never reported the speech done")

    assert done == [True]
    assert hang.ended.is_set()


async def test_cancelling_a_playout_waiter_does_not_cancel_the_playout() -> None:
    """A cancelled `wait_for_playout()` leaves the segment playing; the next waiter gets its
    end."""
    output = FakeAudioOutput()
    await output.capture_frame(audio_frame(0.1))
    output.flush()
    waiter = asyncio.create_task(output.wait_for_playout())
    await asyncio.sleep(0)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    finished = await output.wait_for_playout()

    assert not finished.interrupted
    assert finished.playback_position == pytest.approx(0.1)


async def test_a_playout_wait_with_nothing_left_to_play_returns_the_last_finish() -> None:
    """`wait_for_playout()` once every captured segment finished returns the last finish at
    once instead of waiting for a segment that never comes."""
    output = FakeAudioOutput()
    await output.capture_frame(audio_frame(0.1))
    output.flush()
    await output.wait_for_playout()

    async with asyncio.timeout(5):
        finished = await output.wait_for_playout()

    assert finished.playback_position == pytest.approx(0.1)


# -- the cleanup plugins run on their own tasks


@pytest.mark.parametrize("fails", ["before_the_cancel", "instead_of_cancelling"])
async def test_cancel_and_wait_retrieves_the_failure_of_a_task_it_closes(
    fails: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task that failed before `cancel_and_wait`, or fails instead of being cancelled, has its
    failure retrieved: the loop never reports it as an exception nobody retrieved."""
    reported: list[str] = []
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "call_exception_handler", lambda c: reported.append(c["message"]))

    async def provider_task() -> None:
        try:
            if fails == "instead_of_cancelling":
                await asyncio.Future[None]()
        finally:
            raise RuntimeError("provider task failed while closing")

    task = asyncio.create_task(provider_task())
    await asyncio.sleep(0)
    await aio.cancel_and_wait(task)
    del task
    gc.collect()

    assert reported == []
