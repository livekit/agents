"""Text-session execution identity follows the enclosing transport task."""

import pytest

from livekit.agents import Agent, RunContext, TextSession
from livekit.agents.voice.agent_run import RunHooks, reset_run_hooks, set_run_hooks

from .fake_llm import FakeLLM
from .test_agent_run import respond

pytestmark = pytest.mark.unit


async def test_text_session_stores_the_a2a_task_as_its_execution_identity() -> None:
    seen: list[str] = []
    agent = Agent(
        instructions="Collect details.",
        llm=FakeLLM(fake_responses=[respond("hi", "Which date?"), respond("monday", "Thanks.")]),
        output_type=int,
    )

    @agent.instructions
    def identity(ctx: RunContext) -> str:
        seen.append(ctx.execution_id)
        return "Ask for more information."

    token = set_run_hooks(RunHooks(task_id="a2a-task"))
    try:
        first = await TextSession(agent=agent).run("hi")
        assert first.state.execution_id == "a2a-task"
        restored = type(first.state).from_json(first.state.to_json())
        second = await TextSession(agent=agent).run("monday", state=restored)
    finally:
        reset_run_hooks(token)
    assert second.state.execution_id == "a2a-task"
    assert seen == ["a2a-task", "a2a-task"]
