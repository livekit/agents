"""Regression tests for error precedence in ``RunResult``.

``session.run()`` watches every speech handle created while it is active, e.g.
a ``session.say()`` fired from a function tool. A handle that carries the task
output may complete before another watched speech fails: returning that output
would hide the failure, so the error must still be propagated.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from livekit.agents import (
    Agent,
    AgentSession,
    AgentTask,
    APIConnectionError,
    APIConnectOptions,
    RunContext,
    function_tool,
)
from livekit.agents.llm import FunctionToolCall, RealtimeError
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.agents.voice.run_result import RunResult
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_llm import FakeLLM, FakeLLMResponse, FakeLLMStream

pytestmark = pytest.mark.unit


class _FailingReplyLLM(FakeLLM):
    """FakeLLM that raises for one input, like a provider failure."""

    def __init__(self, *, failing_input: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._failing_input = failing_input

    def chat(self, **kwargs: Any) -> FakeLLMStream:
        return _FailingReplyStream(
            self,
            failing_input=self._failing_input,
            chat_ctx=kwargs["chat_ctx"],
            tools=kwargs.get("tools") or [],
            conn_options=kwargs["conn_options"],
        )


class _FailingReplyStream(FakeLLMStream):
    def __init__(self, llm: FakeLLM, *, failing_input: str, **kwargs: Any) -> None:
        super().__init__(llm, **kwargs)
        self._failing_input = failing_input

    def _get_index_text(self) -> str:
        # completing an AgentTask records an ``agent_config_update`` after the
        # tool output, so the lookup key is the last user/system message or the
        # last function call output
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

    async def _run(self) -> None:
        if self._get_index_text() == self._failing_input:
            raise APIConnectionError("provider exploded")
        await super()._run()


async def _settle_done_callbacks() -> None:
    # SpeechHandle is a future: its done callbacks are scheduled on the loop.
    await asyncio.sleep(0)


async def test_later_handle_error_is_not_hidden_by_output() -> None:
    """The task result completes first; a watched speech that fails afterwards
    must still fail the run instead of being masked by that result."""
    run = RunResult[str](output_type=str)
    with_output = SpeechHandle.create()
    with_output._maybe_run_final_output = "task result"
    failing = SpeechHandle.create()

    run._watch_handle(with_output)
    run._watch_handle(failing)

    with_output._mark_done()
    await _settle_done_callbacks()
    failing._mark_done(error=RealtimeError("provider exploded"))
    await _settle_done_callbacks()

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
        self.session.say("task done")
        return result.value


async def test_run_raises_when_a_later_speech_fails_after_the_task_output() -> None:
    """A failure on a watched speech that completes after the task output must
    fail the run; returning the task result would hide the failure."""
    llm = _FailingReplyLLM(
        failing_input="task-result",
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
        ],
    )

    async with AgentSession(
        llm=llm,
        # a single attempt keeps the failure deterministic
        conn_options=SessionConnectOptions(
            llm_conn_options=APIConnectOptions(max_retry=0, timeout=5.0)
        ),
    ) as sess:
        await sess.start(RootAgent())

        with pytest.raises(APIConnectionError, match="provider exploded"):
            await asyncio.wait_for(
                sess.run(
                    user_input="go",
                    output_type=TaskOutput,
                    output_options={"max_retries": 0},
                ),
                timeout=10.0,
            )
