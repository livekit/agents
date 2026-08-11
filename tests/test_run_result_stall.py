"""A run that never completes must say what is holding it open.

`RunResult` completes only once every watched handle is done. When one never
completes, the run hangs and the caller times out with nothing naming the
culprit — the transcript looks complete, because the reply itself did land, so
the failure reads as spurious. These tests pin the diagnostic that names it.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from livekit.agents import function_tool
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice import Agent, AgentSession, run_result as run_result_module

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


async def _start_session() -> tuple[AgentSession, Agent]:
    session = AgentSession()
    agent = Agent(
        instructions="test agent",
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(input="hello", content="hi there", ttft=0.0, duration=0.0)
            ]
        ),
    )
    await session.start(agent=agent)
    return session, agent


@pytest.mark.asyncio
async def test_pending_handles_names_the_blocking_handle():
    session, _ = await _start_session()

    result = session.run(user_input="hello")
    await asyncio.wait_for(result, timeout=10.0)

    # a completed run is waiting on nothing
    assert result._pending_handles() == []

    # an unresolved handle watched by the run is reported under its label
    blocker: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    result._done_fut = asyncio.Future()  # re-open so _watch_handle takes the handle
    result._watch_handle(blocker, label="foreground_hold")
    assert result._pending_handles() == ["foreground_hold"]

    blocker.set_result(None)
    assert result._pending_handles() == []

    await session.aclose()


@pytest.mark.asyncio
async def test_stall_watchdog_reports_pending_handles(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """A run held open past the threshold logs the handles it is waiting on."""
    monkeypatch.setattr(run_result_module, "STALL_WARN_AFTER", 0.05)

    session, _ = await _start_session()

    result = session.run(user_input="hello")
    await asyncio.wait_for(result, timeout=10.0)

    # hold the completed run open the way a never-resolving guard would
    result._done_fut = asyncio.Future()
    blocker: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    result._watch_handle(blocker, label="foreground_hold")

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        result._stall_watchdog_atask = None  # allow a second arming for the test
        result._start_stall_watchdog()
        await asyncio.sleep(0.2)

    stalls = [r for r in caplog.records if "still waiting on" in r.message]
    assert stalls, "expected a stall warning naming the pending handles"
    assert stalls[0].pending_handles == ["foreground_hold"]
    assert stalls[0].user_input == "hello"

    blocker.set_result(None)
    result._done_fut.set_result(None)
    await session.aclose()


@pytest.mark.asyncio
async def test_stall_watchdog_is_silent_for_a_normal_turn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """A turn that completes must not warn, however low the threshold."""
    monkeypatch.setattr(run_result_module, "STALL_WARN_AFTER", 0.05)

    session, _ = await _start_session()

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        result = session.run(user_input="hello")
        await asyncio.wait_for(result, timeout=10.0)
        await asyncio.sleep(0.2)

    assert [r for r in caplog.records if "still waiting on" in r.message] == []
    assert result._stall_watchdog_atask is None

    await session.aclose()


@pytest.mark.asyncio
async def test_stall_watchdog_reports_a_real_stalled_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """The 6661 shape: a run that genuinely never completes names its handle.

    A tool that never returns holds the reply's speech handle open, so the run
    stays pending forever. The warning has to point at the speech handle that
    `generate_reply` registered, rather than leaving the caller to guess.
    """
    monkeypatch.setattr(run_result_module, "STALL_WARN_AFTER", 0.05)

    tool_entered = asyncio.Event()
    release_tool = asyncio.Event()

    class StallingAgent(Agent):
        @function_tool
        async def never_returns(self) -> str:
            """A tool that hangs until the test releases it."""
            tool_entered.set()
            await release_tool.wait()
            return "released"

    session = AgentSession()
    agent = StallingAgent(
        instructions="test agent",
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="hello",
                    content="",
                    ttft=0.0,
                    duration=0.0,
                    tool_calls=[
                        FunctionToolCall(name="never_returns", arguments="{}", call_id="1")
                    ],
                )
            ]
        ),
    )
    await session.start(agent=agent)

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        result = session.run(user_input="hello")
        await asyncio.wait_for(tool_entered.wait(), timeout=5.0)
        await asyncio.sleep(0.2)

    assert not result.done(), "the hanging tool should have kept the run pending"

    stalls = [r for r in caplog.records if "still waiting on" in r.message]
    assert stalls, "a run stuck on a hanging tool must be reported"
    assert stalls[0].pending_handles == [
        p for p in stalls[0].pending_handles if p.startswith("generate_reply(speech_id=")
    ], f"expected the reply handle to be named, got {stalls[0].pending_handles}"
    assert stalls[0].pending_handles

    release_tool.set()
    await asyncio.wait_for(session.aclose(), timeout=10.0)
