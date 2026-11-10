from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, function_tool
from livekit.agents.beta.workflows import TaskGroup
from livekit.agents.llm import FunctionToolCall, ToolError

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


async def test_parallel_completing_tool_call_reaches_next_task_ctx() -> None:
    """A tool call that completes an AgentTask, returned with a parallel call,
    must appear (with its output) in the next task's chat context.
    Repro for https://github.com/livekit/agents/issues/8307
    """
    captured_calls: list[str] = []
    captured_outputs: list[str] = []
    done = asyncio.Event()

    class CollectTask(AgentTask[str]):
        def __init__(self) -> None:
            super().__init__(instructions="Collect the value.")

        @function_tool
        async def record_value(self, value: str) -> str:
            """Record the value and finish the task."""
            self.complete(value)
            return "recorded"

        @function_tool
        async def lookup(self, query: str) -> str:
            """Look up extra information."""
            return f"result for {query}"

    class VerifyTask(AgentTask[None]):
        def __init__(self) -> None:
            super().__init__(instructions="Verify the context.")

        async def on_enter(self) -> None:
            for item in self.chat_ctx.items:
                if item.type == "function_call":
                    captured_calls.append(item.name or "")
                elif item.type == "function_call_output":
                    captured_outputs.append(item.name or "")
            self.complete(None)

    class Parent(Agent):
        async def on_enter(self) -> None:
            group = TaskGroup(summarize_chat_ctx=False)
            group.add(CollectTask, id="collect", description="Collect the value.")
            group.add(VerifyTask, id="verify", description="Verify the context.")
            await group
            done.set()

    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(
                        name="record_value", arguments='{"value": "x"}', call_id="call_record"
                    ),
                    FunctionToolCall(
                        name="lookup", arguments='{"query": "y"}', call_id="call_lookup"
                    ),
                ],
            ),
        ]
    )
    session = AgentSession(llm=llm)
    await session.start(Parent(instructions="Synthetic conversation."), session_host=False)
    await asyncio.sleep(0.5)
    try:
        await asyncio.wait_for(session.run(user_input="go"), timeout=3)
        await asyncio.wait_for(done.wait(), timeout=3)
    finally:
        await session.aclose()

    assert "lookup" in captured_calls
    assert "record_value" in captured_calls
    assert "lookup" in captured_outputs
    assert "record_value" in captured_outputs


async def test_cancel_supersedes_deferred_completion() -> None:
    """Cancelling a task whose completion was deferred mid tool batch must cancel
    cleanly instead of raising RuntimeError (#8307).
    """

    class SomeTask(AgentTask[None]):
        def __init__(self) -> None:
            super().__init__(instructions="Do something.")

    task = SomeTask()
    # simulate a completion deferred while its tool batch was committing
    task._AgentTask__pending_complete = None  # type: ignore[attr-defined]
    task.cancel()  # must not raise
    assert task.done()
    exc = task._AgentTask__fut.exception()  # type: ignore[attr-defined]
    assert isinstance(exc, ToolError)
