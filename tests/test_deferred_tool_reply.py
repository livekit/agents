from __future__ import annotations

import asyncio
import logging

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.events import SpeechCreatedEvent

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

# how long the async tool keeps working after its update; its result lands while the
# agent that called it is still switching away
TOOL_WORK = 0.5


async def _send_link(ctx: RunContext) -> str:
    await ctx.update("sending the link")
    await asyncio.sleep(TOOL_WORK)
    return "link sent"


class _Second(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="second agent")


class _First(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="first agent")

    @function_tool
    async def send_link(self, ctx: RunContext) -> str:
        """Send a link."""
        return await _send_link(ctx)

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
        super().__init__(instructions="parent agent")

    @function_tool
    async def send_link(self, ctx: RunContext) -> str:
        """Send a link."""
        return await _send_link(ctx)

    @function_tool
    async def collect(self) -> str:
        """Run a task."""
        await _WaitTask()
        return "collected"


def _outputs(agent: Agent) -> list[str]:
    return [i.output for i in agent.chat_ctx.items if i.type == "function_call_output"]


async def _run(
    agent: Agent, *tool_names: str, caplog: pytest.LogCaptureFixture
) -> tuple[list[str], Agent]:
    """Run one turn whose LLM step calls ``send_link`` alongside ``tool_names``. Returns the
    agent current at each ``generate_reply`` speech, and the agent current at the end."""
    calls = [
        FunctionToolCall(name=name, arguments="{}", call_id=f"call_{name}")
        for name in ("send_link", *tool_names)
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
    async with AgentSession(llm=llm) as sess:
        speakers: list[str] = []

        def _on_speech_created(ev: SpeechCreatedEvent) -> None:
            if ev.source == "generate_reply":
                speakers.append(type(sess.current_agent).__name__)

        sess.on("speech_created", _on_speech_created)
        await sess.start(agent)
        assert agent._activity is not None
        executor = agent._activity._tool_executor

        with caplog.at_level(logging.DEBUG, logger="livekit.agents"):
            await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)
            await asyncio.sleep(TOOL_WORK * 4)

        assert executor._reply_task is not None and executor._reply_task.done()
        assert executor._reply_task.exception() is None
        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert not errors, errors
        # the result landed while the agent was paused, not after it closed
        assert any("owning activity is paused" in r.getMessage() for r in caplog.records)
        return speakers, sess.current_agent


async def test_deferred_reply_during_handoff_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    """An async tool that finishes while its agent drains for a handoff must not raise."""
    first = _First()
    speakers, current = await _run(first, "handoff", caplog=caplog)
    assert speakers == ["_First"]  # only the user turn
    assert isinstance(current, _Second)
    assert "link sent" in _outputs(first)


async def test_deferred_reply_during_agent_task_is_not_spoken_by_the_task(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An async tool that finishes while its agent is paused under an AgentTask must not
    have its reply spoken by the task. The result stays in the paused agent's context."""
    parent = _Parent()
    speakers, current = await _run(parent, "collect", caplog=caplog)
    assert speakers == ["_Parent"]  # only the user turn
    assert current is parent
    assert "link sent" in _outputs(parent)
