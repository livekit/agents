"""Public conversation judges retain their named results.

The core runs the application's judges, calling them back with the conversation the caller
supplied, and the intent judge, which consumes a model verdict."""

from typing import Any, Literal

import pytest

from livekit.agents import llm
from livekit.agents.evals import (
    EvaluationResult,
    Judge,
    JudgeGroup,
    JudgmentResult,
)
from livekit.agents.llm import (
    ChatChunk,
    ChatContext,
    ChatMessage,
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
from livekit.agents.voice.run_result import ChatMessageEvent, RunAssert

pytestmark = pytest.mark.unit


class _CapturingLLM(llm.LLM[Any]):
    """LLM that records the tool_choice it was asked to use and replies with a
    single, well-formed tool call."""

    def __init__(self, tool_call: FunctionToolCall | None) -> None:
        super().__init__()
        self._tool_call = tool_call
        self.tool_choice: Any = None
        self.extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN
        self.prompts: list[str] = []

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
        self.prompts.append(chat_ctx.messages()[-1].text_content or "")
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
        tool_call: FunctionToolCall | None,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._tool_call = tool_call

    async def _run(self) -> None:
        if self._tool_call is None:
            self.send(ChatChunk(id="test", delta=ChoiceDelta(role="assistant", content="Fine.")))
            return
        self.send(
            ChatChunk(
                id="test",
                delta=ChoiceDelta(role="assistant", tool_calls=[self._tool_call]),
            )
        )


async def test_judge_group_retains_named_judgments() -> None:
    class Fixed(Judge):
        def __init__(self, name: str, verdict: Literal["pass", "fail", "maybe"]) -> None:
            super().__init__(name=name)
            self._verdict: Literal["pass", "fail", "maybe"] = verdict

        async def evaluate(self, **kwargs: Any) -> JudgmentResult:
            return JudgmentResult(verdict=self._verdict, reasoning=f"{self.name} reasoning")

    conversation = ChatContext()
    conversation.add_message(role="assistant", content="Your booking is pending.")
    result = await JudgeGroup(
        llm=_CapturingLLM(None), judges=[Fixed("booking", "maybe"), Fixed("tone", "pass")]
    ).evaluate(conversation)
    assert set(result.judgments) == {"booking", "tone"}
    assert result.judgments["booking"].reasoning == "booking reasoning"
    assert result.score == 0.75 and result.none_failed and not result.all_passed


async def test_evaluators_judge_the_supplied_conversation_and_reference() -> None:
    seen: list[tuple[ChatContext, ChatContext | None, llm.LLM[Any] | None]] = []

    class Recorder(Judge):
        def __init__(self) -> None:
            super().__init__(name="recorder")

        async def evaluate(
            self,
            *,
            chat_ctx: ChatContext,
            reference: ChatContext | None = None,
            llm: llm.LLM[Any] | None = None,
        ) -> JudgmentResult:
            seen.append((chat_ctx, reference, llm))
            return JudgmentResult(verdict="pass", reasoning="recorded")

    conversation = ChatContext()
    conversation.add_message(role="user", content="Book the 9am slot.")
    conversation.add_message(role="assistant", content="Booked for 9am.")
    reference = ChatContext()
    reference.add_message(role="system", content="secret reference instructions")
    reference.add_message(role="assistant", content="Booked for 10am.")
    model = _CapturingLLM(None)
    group = JudgeGroup(llm=model, judges=[Recorder()])

    without = await group.evaluate(conversation)
    with_reference = await group.evaluate(conversation, reference=reference)

    assert seen == [(conversation, None, model), (conversation, reference, model)]
    for evaluation in (without, with_reference):
        assert evaluation.judgments["recorder"].passed


async def test_a_judge_without_a_usable_verdict_is_absent_never_a_pass() -> None:
    class Broken(Judge):
        def __init__(self, name: str, verdict: str | None) -> None:
            super().__init__(name=name)
            self._verdict = verdict

        async def evaluate(self, **kwargs: Any) -> JudgmentResult:
            if self._verdict is None:
                raise RuntimeError("booking service unreachable")
            return JudgmentResult(verdict=self._verdict, reasoning="?")  # type: ignore[arg-type]

    conversation = ChatContext()
    conversation.add_message(role="assistant", content="Done.")
    group = JudgeGroup(
        llm=_CapturingLLM(None), judges=[Broken("raised", None), Broken("shouted", "PASS")]
    )
    result = await group.evaluate(conversation)
    assert result.judgments == {}
    assert result.score == 0.0 and not result.any_passed
    assert EvaluationResult().score == 0.0


async def test_a_message_judged_off_intent_fails_with_the_event_context() -> None:
    message = ChatMessage(role="assistant", content=["The room is booked."])
    expect = RunAssert([ChatMessageEvent(item=message)])
    model = _CapturingLLM(
        FunctionToolCall(
            name="check_intent",
            arguments='{"success": false, "reason": "it confirms instead of declining"}',
            call_id="intent-1",
        )
    )
    with pytest.raises(AssertionError) as failed:
        await (
            expect.next_event()
            .is_message(role="assistant")
            .judge(model, intent="Explains that the room is unavailable.")
        )
    assert str(failed.value).startswith("Judgement failed: it confirms instead of declining")
    assert ">>> [0] ChatMessageEvent(role='assistant'" in str(failed.value)
    assert "Intent:\nExplains that the room is unavailable." in model.prompts[0]
