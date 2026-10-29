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
        restored = type(first.state)._from_json(first.state._to_json())
        second = await TextSession(agent=agent).run("monday", state=restored)
    finally:
        reset_run_hooks(token)
    assert second.state.execution_id == "a2a-task"
    assert seen == ["a2a-task", "a2a-task"]


@pytest.mark.parametrize("transport", [False, True])
async def test_completed_text_session_starts_a_new_execution_with_retained_history(
    transport: bool,
) -> None:
    seen: list[str] = []
    agent = Agent(
        instructions="Help.",
        llm=FakeLLM(fake_responses=[respond("hi", "Hello."), respond("again", "Welcome back.")]),
    )

    @agent.instructions
    def identity(ctx: RunContext) -> str:
        seen.append(ctx.execution_id)
        return "Continue."

    token = set_run_hooks(RunHooks(task_id="first-task" if transport else None))
    try:
        first = await TextSession(agent=agent).run("hi")
    finally:
        reset_run_hooks(token)
    assert first.result.status == "completed"
    token = set_run_hooks(RunHooks(task_id="second-task" if transport else None))
    try:
        second = await TextSession(agent=agent).run("again", state=first.state)
    finally:
        reset_run_hooks(token)
    assert seen[0] != seen[1]
    assert second.state.execution_id == seen[1]
    if transport:
        assert seen == ["first-task", "second-task"]


async def test_direct_resume_uses_the_execution_saved_by_the_text_session() -> None:
    seen: list[str] = []
    agent = Agent(
        instructions="Collect details.",
        output_type=int,
        llm=FakeLLM(fake_responses=[respond("hi", "Which date?"), respond("monday", "Thanks.")]),
    )

    @agent.instructions
    def identity(ctx: RunContext) -> str:
        seen.append(ctx.execution_id)
        return "Continue collecting."

    token = set_run_hooks(RunHooks(task_id="a2a-task"))
    try:
        first = await TextSession(agent=agent).run("hi")
    finally:
        reset_run_hooks(token)
    assert first.result.status == "input_required"
    await agent.run("monday", message_history=first.result.all_messages())
    assert seen == ["a2a-task", "a2a-task"]


async def test_pending_text_session_refuses_a_different_a2a_task_identity() -> None:
    agent = Agent(
        instructions="Collect details.",
        output_type=int,
        llm=FakeLLM(fake_responses=[respond("hi", "Which number?")]),
    )
    token = set_run_hooks(RunHooks(task_id="pending-task"))
    try:
        first = await TextSession(agent=agent).run("hi")
    finally:
        reset_run_hooks(token)
    token = set_run_hooks(RunHooks(task_id="unrelated-task"))
    try:
        with pytest.raises(ValueError, match="execution identity changed while awaiting input"):
            await TextSession(agent=agent).run("two", state=first.state)
    finally:
        reset_run_hooks(token)
