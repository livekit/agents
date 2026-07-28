from __future__ import annotations

from typing import Any

import pytest

from livekit.agents import llm
from livekit.agents.evals.judge import _evaluate_with_llm
from livekit.agents.llm import (
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    FunctionToolCall,
    LLMStream,
    Tool,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.voice.run_result import ChatMessageAssert, ChatMessageEvent

pytestmark = pytest.mark.unit


class _CapturingLLM(llm.LLM):
    """LLM that records the tool_choice it was asked to use and replies with a
    single, well-formed tool call."""

    def __init__(self, tool_call: FunctionToolCall) -> None:
        super().__init__()
        self._tool_call = tool_call
        self.tool_choice: Any = None

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        self.tool_choice = tool_choice
        return _CapturingStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            tool_call=self._tool_call,
        )


class _CapturingStream(LLMStream):
    def __init__(
        self,
        llm: _CapturingLLM,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        tool_call: FunctionToolCall,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._tool_call = tool_call

    async def _run(self) -> None:
        self._event_ch.send_nowait(
            ChatChunk(
                id="test",
                delta=ChoiceDelta(role="assistant", tool_calls=[self._tool_call]),
            )
        )


@pytest.mark.asyncio
async def test_message_judge_uses_required_tool_choice() -> None:
    fake_llm = _CapturingLLM(
        FunctionToolCall(
            type="function",
            name="check_intent",
            arguments='{"success": true, "reason": "ok"}',
            call_id="call_1",
        )
    )

    event = ChatMessageEvent(item=llm.ChatMessage(role="assistant", content=["Hello there"]))
    await ChatMessageAssert(event, parent=None, index=0).judge(fake_llm, intent="greets the user")  # type: ignore[arg-type]

    assert fake_llm.tool_choice == "required"


@pytest.mark.asyncio
async def test_evals_judge_uses_required_tool_choice() -> None:
    fake_llm = _CapturingLLM(
        FunctionToolCall(
            type="function",
            name="submit_verdict",
            arguments='{"verdict": "pass", "reasoning": "ok"}',
            call_id="call_1",
        )
    )

    result = await _evaluate_with_llm(fake_llm, "does the conversation meet the criteria?")

    assert result.verdict == "pass"
    assert fake_llm.tool_choice == "required"
