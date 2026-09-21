from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall, ToolError
from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice.events import ToolCallEnded, ToolExecutionUpdatedEvent

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


def _response(*calls: FunctionToolCall) -> FakeLLMResponse:
    return FakeLLMResponse(
        input="book",
        content="",
        ttft=0,
        duration=0,
        tool_calls=list(calls),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("session_policy", "agent_policy", "should_run"),
    [
        ("run", None, True),
        ("run", {"async_options": {}}, True),
        ("skip", {}, False),
        ("run", {"on_dependency_error": "skip"}, False),
        ("skip", {"on_dependency_error": "run"}, True),
    ],
)
async def test_dependency_policy_inheritance_and_agent_override(
    session_policy: str,
    agent_policy: dict[str, object] | None,
    should_run: bool,
) -> None:
    dependent_started = asyncio.Event()

    @function_tool(name="root")
    async def root(ctx: RunContext) -> str:
        raise ToolError("root failed")

    @function_tool(name="dependent", after=("root",))
    async def dependent(ctx: RunContext) -> str:
        dependent_started.set()
        return "dependent ran"

    session = AgentSession(
        llm=FakeLLM(
            fake_responses=[
                _response(
                    FunctionToolCall(name="dependent", arguments="{}", call_id="dependent"),
                    FunctionToolCall(name="root", arguments="{}", call_id="root"),
                )
            ]
        ),
        stt=None,
        vad=None,
        tts=None,
        turn_handling={"turn_detection": None},
        tool_handling={"on_dependency_error": session_policy},
    )
    agent_kwargs: dict[str, object] = {"instructions": "workflow", "tools": [root, dependent]}
    if agent_policy is not None:
        agent_kwargs["tool_handling"] = agent_policy
    agent = Agent(**agent_kwargs)  # type: ignore[arg-type]
    dependent_terminal = asyncio.Event()

    def on_tool_update(event: ToolExecutionUpdatedEvent) -> None:
        if isinstance(event.update, ToolCallEnded) and event.update.call_id == "dependent":
            dependent_terminal.set()

    session.on("tool_execution_updated", on_tool_update)
    await session.start(agent)
    try:
        session.generate_reply(user_input="book")
        await asyncio.wait_for(dependent_terminal.wait(), timeout=5)
        assert dependent_started.is_set() is should_run
    finally:
        await asyncio.wait_for(session.aclose(), timeout=5)


def test_agent_async_options_only_inherits_dependency_policy() -> None:
    session = AgentSession(tool_handling={"on_dependency_error": "run"})
    agent = Agent(instructions="workflow", tool_handling={"async_options": {}})

    assert session._dependency_error_policy == "run"
    assert agent._dependency_error_policy is NOT_GIVEN


@pytest.mark.parametrize("invalid", ["cancel", "", None, 1])
def test_invalid_dependency_policy_is_rejected(invalid: object) -> None:
    with pytest.raises(ValueError):
        AgentSession(tool_handling={"on_dependency_error": invalid})  # type: ignore[typeddict-item]
    with pytest.raises(ValueError):
        Agent(
            instructions="workflow",
            tool_handling={"on_dependency_error": invalid},  # type: ignore[typeddict-item]
        )


def test_dependency_names_reject_a_single_string() -> None:
    with pytest.raises(TypeError, match="sequence of tool names"):

        @function_tool(after="root")  # type: ignore[arg-type]
        async def dependent(ctx: RunContext) -> str:
            return "dependent"
