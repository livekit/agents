from __future__ import annotations

import pytest

from livekit.agents import llm
from livekit.agents.llm import DelegationCheck, DelegationDetector, FunctionToolCall

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


@llm.function_tool
async def cancel_appointment() -> None:
    """Cancel the caller's appointment."""


def _check() -> DelegationCheck:
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="Cancel my 3pm.")
    chat_ctx.add_message(role="assistant", content="Done, it's cancelled.")
    return DelegationCheck(chat_ctx=chat_ctx, tools=[cancel_appointment], instructions="Be brief.")


# what the detector sends: the backend tools, the agent's instructions, then the conversation
_INPUT = (
    "## Backend tools\n\n- cancel_appointment: Cancel the caller's appointment.\n\n"
    "## The voice agent's instructions\n\nBe brief.\n\n"
    "## Conversation\n\nuser: Cancel my 3pm.\nassistant: Done, it's cancelled."
)


def _reporting(arguments: str) -> FakeLLM:
    call = FunctionToolCall(name="report", arguments=arguments, call_id="c1")
    return FakeLLM(
        fake_responses=[
            FakeLLMResponse(input=_INPUT, content="", ttft=0, duration=0, tool_calls=[call])
        ]
    )


async def test_an_llm_reports_its_verdict_through_a_tool_call() -> None:
    assert await DelegationDetector(_reporting('{"delegate": true}'))(_check()) is True
    assert await DelegationDetector(_reporting('{"delegate": false}'))(_check()) is False
    # an answer it cannot read leaves the turn alone
    assert await DelegationDetector(_reporting("not json"))(_check()) is False


class _Decided(llm.DecisionModel):
    def __init__(self, probability: float) -> None:
        super().__init__(capabilities=llm.DecisionCapabilities(decision_kinds=["probability"]))
        self._probability = probability

    async def _evaluate_impl(self, *, input, decisions, conn_options) -> llm.DecisionResponse:  # type: ignore[no-untyped-def]
        assert input == _INPUT
        return llm.DecisionResponse(
            results={"delegate": llm.ProbabilityResult(value=self._probability)}
        )


async def test_a_decision_model_delegates_at_even_odds_or_more() -> None:
    assert await DelegationDetector(_Decided(0.5))(_check()) is True
    assert await DelegationDetector(_Decided(0.4))(_check()) is False
