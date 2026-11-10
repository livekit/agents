from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping

import pytest

from livekit.agents import (
    Agent,
    Block,
    ConversationItemAddedEvent,
    DecisionEvaluated,
    DecisionGuardrail,
    Steer,
)
from livekit.agents.llm import decisions
from livekit.agents.llm.decisions import DecisionResponse, ProbabilityResult
from livekit.agents.types import APIConnectOptions

from .fake_session import FakeActions, create_session, run_session
from .fake_tts import FakeTTS

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60.0


class KeywordModel(decisions.DecisionModel):
    """Scores every decision 1.0 when the input contains ``keyword``, else 0.0."""

    def __init__(self, keyword: str) -> None:
        super().__init__(
            capabilities=decisions.DecisionCapabilities(decision_kinds=["probability"])
        )
        self._keyword = keyword
        self.inputs: list[str] = []

    async def _evaluate_impl(
        self,
        *,
        input: str,
        decisions: Mapping[str, decisions.Decision],
        conn_options: APIConnectOptions,
    ) -> DecisionResponse:
        self.inputs.append(input)
        value = 1.0 if self._keyword in input.lower() else 0.0
        return DecisionResponse(
            results={name: ProbabilityResult(value=value) for name in decisions}
        )


FLAG = {"flag": decisions.Probability("Flag this.")}


def _messages(events: list[ConversationItemAddedEvent]) -> list[tuple[str, str]]:
    return [
        (ev.item.role, ev.item.text_content or "") for ev in events if ev.item.type == "message"
    ]


async def _run(
    actions: FakeActions, setup: Callable[[DecisionGuardrail], None], keyword: str
) -> list[tuple[str, str]]:
    session = create_session(actions)
    tts = session.tts
    assert isinstance(tts, FakeTTS)
    tts._fake_audio_duration = 0.5  # any text the test didn't script still plays

    guardrail = DecisionGuardrail(session, model=KeywordModel(keyword))
    setup(guardrail)

    events: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", events.append)
    await asyncio.wait_for(
        run_session(session, Agent(instructions="You are a helpful assistant.")),
        timeout=SESSION_TIMEOUT,
    )
    return _messages(events)


async def test_user_turn_block_skips_the_reply_and_says_the_replacement() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "you are an idiot")
    actions.add_llm("This reply must never be generated.")

    def setup(guardrail: DecisionGuardrail) -> None:
        @guardrail.on_user_turn(decisions=FLAG)
        def before_reply(ev: DecisionEvaluated) -> Block | None:
            return Block(say="I'm going to end the call here.") if ev.value("flag") else None

    messages = await _run(actions, setup, keyword="idiot")
    assert messages == [
        ("user", "you are an idiot"),
        ("assistant", "I'm going to end the call here."),
    ]


async def test_user_turn_steer_adds_instructions_for_the_reply() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "my order is still broken")
    actions.add_llm("Unsteered reply.")
    # the fake LLM answers by its last message, which is the steer's instructions
    actions.add_llm("What is still wrong?", input="The issue is not resolved.")

    def setup(guardrail: DecisionGuardrail) -> None:
        @guardrail.on_user_turn(decisions=FLAG)
        def before_reply(ev: DecisionEvaluated) -> Steer | None:
            return Steer(instructions="The issue is not resolved.") if ev.value("flag") else None

    messages = await _run(actions, setup, keyword="broken")
    assert messages == [
        ("user", "my order is still broken"),
        ("assistant", "What is still wrong?"),
    ]


async def test_agent_sentence_block_keeps_allowed_sentences_and_says_the_replacement() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "can I get my money back")
    actions.add_llm(
        "Thanks for asking about that order. You will get a full refund today. "
        "It should arrive within three days."
    )

    def setup(guardrail: DecisionGuardrail) -> None:
        @guardrail.on_agent_sentence(decisions=FLAG)
        def during_reply(ev: DecisionEvaluated) -> Block | None:
            return Block(say="Let me check with a colleague.") if ev.value("flag") else None

    messages = await _run(actions, setup, keyword="refund")
    assert messages[0] == ("user", "can I get my money back")
    assert messages[1][0] == "assistant"
    assert messages[1][1].strip() == "Thanks for asking about that order."
    assert messages[2] == ("assistant", "Let me check with a colleague.")
    assert len(messages) == 3


async def test_agent_turn_sees_each_reply() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "hello")
    actions.add_llm("Sorry for the wait, how can I help?")
    seen: list[str] = []

    def setup(guardrail: DecisionGuardrail) -> None:
        @guardrail.on_agent_turn(decisions=FLAG)
        def after_reply(ev: DecisionEvaluated) -> None:
            seen.append(ev.input_text)

    await _run(actions, setup, keyword="sorry")
    assert seen == ["Sorry for the wait, how can I help?"]
