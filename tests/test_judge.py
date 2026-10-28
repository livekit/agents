"""Public conversation judges consume model verdicts and retain their named results.

The judges run in the core: built-in ones frame the conversation for the model, the
application's are called back with the conversation the caller supplied."""

from typing import Any

import pytest

from livekit.agents import llm
from livekit.agents.evals import (
    EvaluationResult,
    Judge,
    JudgeGroup,
    JudgmentResult,
    accuracy_judge,
    safety_judge,
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


def _verdict(verdict: str, reasoning: str) -> FunctionToolCall:
    return FunctionToolCall(
        name="submit_verdict",
        arguments=f'{{"verdict":"{verdict}","reasoning":"{reasoning}"}}',
        call_id="verdict-1",
    )


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
    model = _CapturingLLM(_verdict("fail", "wrong time"))
    group = JudgeGroup(llm=model, judges=[Recorder(), accuracy_judge()])

    without = await group.evaluate(conversation)
    with_reference = await group.evaluate(conversation, reference=reference)

    assert seen == [(conversation, None, model), (conversation, reference, model)]
    assert "Reference:" not in model.prompts[0]
    assert "user: Book the 9am slot.\nassistant: Booked for 9am." in model.prompts[1]
    assert "Reference:\nassistant: Booked for 10am." in model.prompts[1]
    assert "secret reference instructions" not in model.prompts[1]
    for evaluation in (without, with_reference):
        assert evaluation.judgments["recorder"].passed
        assert evaluation.judgments["accuracy"].failed
        assert evaluation.judgments["accuracy"].instructions.startswith("All information")


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
    silent = _CapturingLLM(None)
    group = JudgeGroup(
        llm=silent, judges=[Broken("raised", None), Broken("shouted", "PASS"), safety_judge()]
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
