"""Regression tests for RunResult when a run watches multiple speech handles.

``session.run()`` automatically watches every speech handle created while it is
active, e.g. a ``session.say()`` fired from a function tool. The final output
and error must be read from the handle that actually carries them, not merely
from the last watched handle that happened to complete.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall, RealtimeError
from livekit.agents.voice.run_result import RunResult
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_llm import FakeLLM, FakeLLMResponse, FakeLLMStream

pytestmark = pytest.mark.unit


class _HandoffFakeLLM(FakeLLM):
    """FakeLLM that skips trailing ``agent_config_update`` items.

    Completing an AgentTask hands the activity back to the previous agent and
    records an ``agent_config_update`` on the chat context. The stock FakeLLM
    only reads the last item, which may be that update instead of the tool
    output that triggered the follow-up turn.
    """

    def chat(self, **kwargs: Any) -> FakeLLMStream:
        return _HandoffFakeLLMStream(
            self,
            chat_ctx=kwargs["chat_ctx"],
            tools=kwargs.get("tools") or [],
            conn_options=kwargs["conn_options"],
        )


class _HandoffFakeLLMStream(FakeLLMStream):
    def _get_index_text(self) -> str:
        for item in reversed(self.chat_ctx.items):
            if (
                item.type == "message"
                and item.role in ("user", "system")
                and (text := item.text_content)
            ):
                return text
            if item.type == "function_call_output":
                return item.output
        raise ValueError("No input text found")


async def _run_done_callbacks() -> None:
    # SpeechHandle is a future: its done callbacks are scheduled on the loop.
    await asyncio.sleep(0)


async def test_output_from_earlier_handle_is_kept() -> None:
    """The AgentTask result lives on the first handle; a later handle without
    output used to replace it, leaving the run with no final output."""
    run = RunResult[str](output_type=str)
    with_output = SpeechHandle.create()
    with_output._maybe_run_final_output = "task result"
    without_output = SpeechHandle.create()

    run._watch_handle(with_output)
    run._watch_handle(without_output)

    with_output._mark_done()
    await _run_done_callbacks()
    without_output._mark_done()
    await _run_done_callbacks()

    await asyncio.wait_for(run, timeout=1.0)
    assert run.final_output == "task result"


async def test_earlier_handle_error_is_not_swallowed() -> None:
    """The failing handle completes first; a later clean handle without output
    used to let the run resolve successfully and hide the failure."""
    run = RunResult[None](output_type=None)
    failing = SpeechHandle.create()
    clean = SpeechHandle.create()

    run._watch_handle(failing)
    run._watch_handle(clean)

    failing._mark_done(error=RealtimeError("provider exploded"))
    await _run_done_callbacks()
    clean._mark_done()
    await _run_done_callbacks()

    with pytest.raises(RealtimeError, match="provider exploded"):
        await asyncio.wait_for(run, timeout=1.0)


@dataclass
class TaskOutput:
    value: str


class ResultTask(AgentTask[TaskOutput]):
    """Completes with a result on its own speech handle."""

    def __init__(self) -> None:
        super().__init__(instructions="result task")

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="finish_now")

    @function_tool
    async def finish(self, ctx: RunContext) -> str:
        """Finish the task."""
        self.complete(TaskOutput(value="task-result"))
        return "done"


class RootAgent(Agent):
    """Awaits the task, then speaks through session.say() in the same tool."""

    def __init__(self) -> None:
        super().__init__(instructions="root agent")

    @function_tool
    async def start(self, ctx: RunContext) -> str:
        """Start the result task."""
        result = await ResultTask()
        # this speech handle has no output of its own and outlives the task's
        # speech handle; it must not replace the task result
        self.session.say("task done")
        return result.value


async def test_run_returns_task_output_when_tool_says_too() -> None:
    """A session.say() from the tool that awaits an AgentTask must not hide
    the task result from ``run(output_type=...)``."""
    llm = _HandoffFakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="start", arguments="{}", call_id="call_1")],
            ),
            FakeLLMResponse(
                input="finish_now",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="finish", arguments="{}", call_id="call_2")],
            ),
            FakeLLMResponse(input="done", content="", ttft=0, duration=0),
            FakeLLMResponse(input="task-result", content="outer reply", ttft=0, duration=0),
        ]
    )

    async with AgentSession(llm=llm) as sess:
        await sess.start(RootAgent())

        result = await asyncio.wait_for(
            sess.run(
                user_input="go",
                output_type=TaskOutput,
                output_options={"max_retries": 0},
            ),
            timeout=5.0,
        )
        assert result.final_output.value == "task-result"
