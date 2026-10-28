"""Public conversation judges consume model verdicts and retain their named results."""

from typing import Any

import pytest

from livekit.agents import llm
from livekit.agents.evals import Judge, JudgeGroup, JudgmentResult, safety_judge
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

pytestmark = pytest.mark.unit


class _CapturingLLM(llm.LLM):
    """LLM that records the tool_choice it was asked to use and replies with a
    single, well-formed tool call."""

    def __init__(self, tool_call: FunctionToolCall) -> None:
        super().__init__()
        self._tool_call = tool_call
        self.tool_choice: Any = None
        self.extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN

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
        self.extra_kwargs = extra_kwargs
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
        self.send(
            ChatChunk(
                id="test",
                delta=ChoiceDelta(role="assistant", tool_calls=[self._tool_call]),
            )
        )


async def test_safety_judge_requires_a_verdict_and_returns_its_reasoning() -> None:
    model = _CapturingLLM(
        FunctionToolCall(
            name="submit_verdict",
            arguments='{"verdict":"fail","reasoning":"rude reply"}',
            call_id="verdict-1",
        )
    )
    conversation = ChatContext()
    conversation.add_message(role="assistant", content="Go away.")
    result = await safety_judge(model).evaluate(chat_ctx=conversation)
    assert result.verdict == "fail" and result.reasoning == "rude reply"
    assert model.tool_choice == "required"


async def test_judge_group_retains_custom_and_model_judgments() -> None:
    class BookingJudge(Judge):
        def __init__(self) -> None:
            super().__init__(name="booking")

        async def evaluate(self, **kwargs: Any) -> JudgmentResult:
            return JudgmentResult(verdict="maybe", reasoning="booking still pending")

    model = _CapturingLLM(
        FunctionToolCall(
            name="submit_verdict",
            arguments='{"verdict":"pass","reasoning":"polite reply"}',
            call_id="verdict-1",
        )
    )
    conversation = ChatContext()
    conversation.add_message(role="assistant", content="Your booking is pending.")
    result = await JudgeGroup(llm=model, judges=[BookingJudge(), safety_judge()]).evaluate(
        conversation
    )
    assert set(result.judgments) == {"booking", "safety"}
    assert result.judgments["booking"].reasoning == "booking still pending"
    assert result.score == 0.75 and result.none_failed and not result.all_passed
