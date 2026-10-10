from __future__ import annotations

import asyncio
import logging

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall
from livekit.agents.llm.async_toolset import AsyncToolset
from livekit.agents.voice.events import SpeechCreatedEvent

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

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


def _outputs(agent: Agent) -> list[str]:
    return [i.output for i in agent.chat_ctx.items if i.type == "function_call_output"]


async def _run(
    agent: Agent,
    other_tool: str,
    caplog: pytest.LogCaptureFixture,
    *,
    session_tools: list[AsyncToolset] | None = None,
) -> tuple[list[str], Agent]:
    """Run one turn whose LLM step calls ``send_link`` alongside ``other_tool``. Returns the
    agent current at each ``generate_reply`` speech, and the agent current at the end."""
    calls = [
        FunctionToolCall(name=name, arguments="{}", call_id=f"call_{name}")
        for name in ("send_link", other_tool)
    ]
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go", content="one moment", ttft=0.1, duration=0.1, tool_calls=calls
            ),
            # the draining agent's tool reply outlasts the async tool's work
            FakeLLMResponse(input="", content="checking", ttft=0.1, duration=TOOL_WORK * 2),
        ]
    )
    async with AgentSession(llm=llm, tools=session_tools or []) as sess:
        speakers: list[str] = []

        def _on_speech_created(ev: SpeechCreatedEvent) -> None:
            if ev.source == "generate_reply":
                speakers.append(type(sess.current_agent).__name__)

        sess.on("speech_created", _on_speech_created)
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
    assert "link sent" in _outputs(first)
    assert any("owning activity closed" in r.getMessage() for r in caplog.records)


async def test_reply_during_agent_task_waits_for_resume(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An async tool that finishes under an AgentTask is announced by its own agent once
    the task returns, never by the task."""
    parent = _Parent()
    speakers, current = await _run(parent, "collect", caplog)
    assert speakers == ["_Parent", "_Parent"]  # the user turn, then the deferred reply
    assert current is parent
    assert "link sent" in _outputs(parent)


async def test_session_scoped_reply_follows_handoff(caplog: pytest.LogCaptureFixture) -> None:
    """A session-scoped async tool that finishes during a handoff is announced by the
    next agent."""
    first = _First(with_send_link=False)
    toolset = AsyncToolset(id="links", tools=[send_link])
    speakers, current = await _run(first, "handoff", caplog, session_tools=[toolset])
    assert isinstance(current, _Second)
    assert speakers == ["_First", "_Second"]


async def test_session_scoped_reply_is_dropped_on_close() -> None:
    """A session-scoped reply still pending when the session closes is dropped, and the
    close completes."""
    calls = [FunctionToolCall(name="send_link", arguments="{}", call_id="call_send_link")]
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go", content="one moment", ttft=0.1, duration=0.1, tool_calls=calls
            ),
            # keeps the agent busy past the tool's work, so the reply waits for idle
            FakeLLMResponse(input="busy", content="still going", ttft=0.1, duration=10.0),
        ]
    )
    toolset = AsyncToolset(id="links", tools=[send_link])
    sess = AgentSession(llm=llm, tools=[toolset])
    await sess.start(_First(with_send_link=False))
    await sess.run(user_input="go")
    sess.generate_reply(user_input="busy")
    await asyncio.sleep(TOOL_WORK * 2)  # the tool has finished; its reply waits for idle
    assert toolset._executor._reply_task is not None
    assert not toolset._executor._reply_task.done()

    await asyncio.wait_for(sess.aclose(), timeout=5.0)
