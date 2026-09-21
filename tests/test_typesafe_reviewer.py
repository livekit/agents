from __future__ import annotations

import logging
from typing import Any

import pytest

from livekit.agents import Agent, llm
from livekit.agents.llm import ChatChunk, ChoiceDelta, FunctionToolCall, function_tool
from livekit.plugins.typesafe import Reviewer, Verdict, default_checks
from livekit.plugins.typesafe.reviewer import NUDGE_PREFIX

pytestmark = pytest.mark.unit


class FakeSystemOne:
    """Stands in for the TypeSafe endpoint; records what was asked."""

    model = "jev-latest"  # part of the client contract Reviewer logs on attach

    def __init__(self, answers: dict[str, Any] | list[dict[str, Any]], *, fail: bool = False):
        self._answers = answers if isinstance(answers, list) else [answers]
        self._fail = fail
        self.requests: list[tuple[Any, dict[str, Any]]] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    async def evaluate(
        self, state: Any, questions: dict[str, Any], *, model: str | None = None
    ) -> dict[str, Any]:
        self.requests.append((state, questions))
        if self._fail:
            raise RuntimeError("typesafe is down")
        answers = self._answers[min(len(self.requests) - 1, len(self._answers) - 1)]
        # shaped like a real response: versioned id, not the alias, plus usage
        return {
            "model": "jev-1.13.0",
            "answers": answers,
            "usage": {"input_tokens": 296, "output_tokens": 20},
        }


def noul(value: float) -> dict[str, Any]:
    return {"type": "noul", "noul": value}


def score(value: float) -> dict[str, Any]:
    return {"type": "score", "score": value, "legend": {}, "probabilities": {}, "confidence": 0.9}


def choice(pick: str, confidence: float = 0.9) -> dict[str, Any]:
    return {"type": "choice", "choice": pick, "probabilities": {}, "confidence": confidence}


CLEAN = {
    "follows_instructions": noul(0.98),
    "unsupported_claim": noul(0.02),
    "advances_task": noul(0.95),
    "severity": score(0.1),
}

OFF_COURSE = {
    "follows_instructions": noul(0.05),
    "unsupported_claim": noul(0.9),
    "advances_task": noul(0.95),
    "severity": score(2.4),
}


@function_tool
async def lookup_order(order_id: str) -> str:
    """Look up the status of a customer order."""
    return "shipped"


def make_agent(*, tools: list[Any] | None = None) -> Agent:
    return Agent(
        instructions="You are a support agent. Never quote a price.",
        tools=tools if tools is not None else [],
    )


def make_reviewer(client: FakeSystemOne, **kwargs: Any) -> Reviewer:
    return Reviewer(_client=client, **kwargs)  # type: ignore[arg-type]


async def test_clean_reply_does_not_trip() -> None:
    client = FakeSystemOne(CLEAN)
    reviewer = make_reviewer(client)
    verdict = await reviewer.review(make_agent(), "Let me check that for you.")

    assert not verdict.needs_correction
    assert verdict.triggered_checks == []
    assert client.calls == 1


async def test_off_course_reply_trips_the_right_checks() -> None:
    reviewer = make_reviewer(FakeSystemOne(OFF_COURSE))
    verdict = await reviewer.review(make_agent(), "That will be $40.")

    assert set(verdict.triggered_checks) == {
        "follows_instructions",
        "unsupported_claim",
        "severity",
    }
    assert "advances_task" not in verdict.triggered_checks


async def test_state_carries_the_prompt_and_tool_catalog() -> None:
    client = FakeSystemOne(CLEAN)
    reviewer = make_reviewer(client)
    await reviewer.review(make_agent(tools=[lookup_order]), "One moment.")

    state, questions = client.requests[0]
    assert state["instructions"] == "You are a support agent. Never quote a price."
    assert state["available_tools"] == [
        {"name": "lookup_order", "description": "Look up the status of a customer order."}
    ]
    assert state["reviewed_reply"] == "One moment."
    # every check rides in the one request
    assert set(questions) == {
        "follows_instructions",
        "unsupported_claim",
        "advances_task",
        "expected_tool",
        "severity",
    }


async def test_expected_tool_check_is_skipped_without_tools() -> None:
    client = FakeSystemOne(CLEAN)
    await make_reviewer(client).review(make_agent(), "Sure.")

    _state, questions = client.requests[0]
    assert "expected_tool" not in questions


async def test_expected_tool_trips_only_when_the_tool_was_not_used() -> None:
    answers = dict(CLEAN, expected_tool=choice("lookup_order"))
    agent = make_agent(tools=[lookup_order])

    missed = make_reviewer(FakeSystemOne(answers))
    assert (await missed.review(agent, "It shipped yesterday.")).triggered_checks == [
        "expected_tool"
    ]

    used = make_reviewer(FakeSystemOne(answers))
    used._tools_used = ["lookup_order"]
    assert not (await used.review(agent, "It shipped yesterday.")).needs_correction


async def test_expected_tool_stays_quiet_when_unconfident() -> None:
    answers = dict(CLEAN, expected_tool=choice("lookup_order", confidence=0.3))
    reviewer = make_reviewer(FakeSystemOne(answers))

    assert not (
        await reviewer.review(make_agent(tools=[lookup_order]), "It shipped.")
    ).needs_correction


async def test_thresholds_are_tunable() -> None:
    answers = dict(CLEAN, unsupported_claim=noul(0.7))

    assert (
        "unsupported_claim"
        in (await make_reviewer(FakeSystemOne(answers)).review(make_agent(), "x")).triggered_checks
    )

    relaxed = make_reviewer(FakeSystemOne(answers), checks=default_checks(unsupported_claim=0.8))
    assert not (await relaxed.review(make_agent(), "x")).needs_correction


async def test_a_failed_check_fails_open() -> None:
    reviewer = make_reviewer(FakeSystemOne(CLEAN, fail=True))
    verdict = await reviewer.review(make_agent(), "anything")

    assert not verdict.needs_correction
    assert verdict.answers == {}


async def test_nudge_lands_in_the_chat_context() -> None:
    reviewer = make_reviewer(FakeSystemOne(OFF_COURSE))
    agent = make_agent()
    verdict = await reviewer.review(agent, "That will be $40.")
    await reviewer._apply_nudge(agent, verdict)

    last = agent.chat_ctx.items[-1]
    assert last.role == "system"
    assert last.text_content is not None
    assert last.text_content.startswith(NUDGE_PREFIX)
    assert "it broke a rule in your instructions" in last.text_content


async def test_on_verdict_fires_for_clean_checks_too() -> None:
    seen: list[Verdict] = []
    reviewer = make_reviewer(FakeSystemOne(CLEAN), on_verdict=seen.append)
    await reviewer.review(make_agent(), "Sure thing.")

    assert len(seen) == 1 and not seen[0].needs_correction


# --- gate -----------------------------------------------------------------


def text_draft(*texts: str):
    """A draft source yielding one reply per call, in order."""
    calls: list[llm.ChatContext] = []

    def produce(ctx: llm.ChatContext):
        async def gen():
            yield texts[min(len(calls) - 1, len(texts) - 1)]

        calls.append(ctx)
        return gen()

    produce.calls = calls  # type: ignore[attr-defined]
    return produce


async def collect(stream: Any) -> list[Any]:
    return [chunk async for chunk in stream]


async def test_gate_is_a_passthrough_when_no_check_is_gated() -> None:
    client = FakeSystemOne(OFF_COURSE)
    reviewer = make_reviewer(client)
    draft = text_draft("That will be $40.")

    out = await reviewer.gate(make_agent(), llm.ChatContext.empty(), [], None, draft=draft)

    assert await collect(out) == ["That will be $40."]
    assert client.calls == 0


async def test_gate_releases_a_clean_draft_unchanged() -> None:
    reviewer = make_reviewer(
        FakeSystemOne(CLEAN), checks=default_checks(gated_check_ids=["follows_instructions"])
    )
    draft = text_draft("Let me check that.")

    out = await reviewer.gate(make_agent(), llm.ChatContext.empty(), [], None, draft=draft)

    assert await collect(out) == ["Let me check that."]
    assert len(draft.calls) == 1  # type: ignore[attr-defined]


async def test_gate_redrafts_once_with_the_correction_attached() -> None:
    client = FakeSystemOne([OFF_COURSE, CLEAN])
    reviewer = make_reviewer(
        client, checks=default_checks(gated_check_ids=["follows_instructions"])
    )
    draft = text_draft("That will be $40.", "Let me check that for you.")

    out = await reviewer.gate(make_agent(), llm.ChatContext.empty(), [], None, draft=draft)

    assert await collect(out) == ["Let me check that for you."]
    assert client.calls == 2

    # the redraft saw the correction, the first draft did not
    first, second = draft.calls  # type: ignore[attr-defined]
    assert first.items == []
    assert second.items[-1].text_content.startswith(NUDGE_PREFIX)


async def test_gate_gives_up_after_max_redrafts_and_still_nudges() -> None:
    client = FakeSystemOne(OFF_COURSE)
    reviewer = make_reviewer(
        client, checks=default_checks(gated_check_ids=["follows_instructions"])
    )
    agent = make_agent()
    draft = text_draft("That will be $40.")

    out = await reviewer.gate(agent, llm.ChatContext.empty(), [], None, draft=draft)

    assert await collect(out) == ["That will be $40."]  # released rather than silenced
    assert client.calls == 2
    assert agent.chat_ctx.items[-1].text_content.startswith(NUDGE_PREFIX)


async def test_gate_lets_tool_calls_through_unjudged() -> None:
    client = FakeSystemOne(OFF_COURSE)
    reviewer = make_reviewer(
        client, checks=default_checks(gated_check_ids=["follows_instructions"])
    )

    chunk = ChatChunk(
        id="c1",
        delta=ChoiceDelta(
            role="assistant",
            tool_calls=[
                FunctionToolCall(name="lookup_order", arguments="{}", call_id="call_1"),
            ],
        ),
    )

    def produce(ctx: llm.ChatContext):
        async def gen():
            yield chunk

        return gen()

    out = await reviewer.gate(make_agent(), llm.ChatContext.empty(), [], None, draft=produce)

    assert await collect(out) == [chunk]
    assert client.calls == 0


async def test_gate_marks_its_draft_as_fully_judged() -> None:
    """Observe may skip the committed item only because gate ran every check."""
    reviewer = make_reviewer(
        FakeSystemOne(CLEAN), checks=default_checks(gated_check_ids=["follows_instructions"])
    )
    draft = text_draft("All set.")

    await collect(await reviewer.gate(make_agent(), llm.ChatContext.empty(), [], None, draft=draft))

    assert "All set." in reviewer._judged_by_gate


async def test_gate_evaluates_every_check_not_only_the_gated_ones() -> None:
    """Turning one check into a gate must not switch the others off."""
    client = FakeSystemOne(CLEAN)
    reviewer = make_reviewer(
        client, checks=default_checks(gated_check_ids=["follows_instructions"])
    )

    await collect(
        await reviewer.gate(
            make_agent(tools=[lookup_order]),
            llm.ChatContext.empty(),
            [lookup_order],  # the flattened catalog the pipeline hands llm_node
            None,
            draft=text_draft("All set."),
        )
    )

    _state, questions = client.requests[0]
    assert set(questions) == {
        "follows_instructions",
        "unsupported_claim",
        "advances_task",
        "expected_tool",
        "severity",
    }


async def test_gate_releases_but_still_nudges_when_only_an_ungated_check_triggers() -> None:
    """A draft can clear the gate and still deserve a correction next turn."""
    only_ungated = dict(CLEAN, unsupported_claim=noul(0.95))
    client = FakeSystemOne(only_ungated)
    reviewer = make_reviewer(
        client, checks=default_checks(gated_check_ids=["follows_instructions"])
    )
    agent = make_agent()

    out = await reviewer.gate(
        agent, llm.ChatContext.empty(), [], None, draft=text_draft("It ships Tuesday.")
    )

    assert await collect(out) == ["It ships Tuesday."]  # gate cleared, nothing redrafted
    assert client.calls == 1
    assert reviewer.results[0].triggered_checks == ["unsupported_claim"]
    assert agent.chat_ctx.items[-1].text_content.startswith(NUDGE_PREFIX)


async def test_gate_builds_state_from_the_generation_context_not_the_agent() -> None:
    """The pending user message is only in the context handed to llm_node.

    It reaches ``agent.chat_ctx`` after the speech is scheduled, so a gate that
    read the agent would judge the draft without the request that prompted it.
    """
    client = FakeSystemOne(CLEAN)
    reviewer = make_reviewer(
        client, checks=default_checks(gated_check_ids=["follows_instructions"])
    )

    pending = llm.ChatContext.empty()
    pending.add_message(role="user", content="Where is order 123?")

    await collect(
        await reviewer.gate(make_agent(), pending, [], None, draft=text_draft("It shipped."))
    )

    state, _ = client.requests[0]
    assert state["transcript"] == [{"role": "user", "text": "Where is order 123?"}]


async def test_gate_uses_the_turns_tool_catalog() -> None:
    """Session-registered tools reach llm_node but are not on agent.tools."""
    client = FakeSystemOne(CLEAN)
    reviewer = make_reviewer(
        client, checks=default_checks(gated_check_ids=["follows_instructions"])
    )

    await collect(
        await reviewer.gate(
            make_agent(),  # the agent itself owns no tools
            llm.ChatContext.empty(),
            [lookup_order],
            None,
            draft=text_draft("Let me check."),
        )
    )

    state, questions = client.requests[0]
    assert [t["name"] for t in state["available_tools"]] == ["lookup_order"]
    assert "expected_tool" in questions


async def test_audio_only_instructions_reach_the_reviewer() -> None:
    """A voice turn resolves the audio variant; reviewing the common part hides rules."""
    from livekit.agents.llm.chat_context import Instructions

    client = FakeSystemOne(CLEAN)
    agent = Agent(instructions=Instructions("Be helpful.", audio="Never read card numbers aloud."))

    await make_reviewer(client).review(agent, "Your card is 4111 1111 1111 1111.")

    state, _ = client.requests[0]
    assert "Never read card numbers aloud." in state["instructions"]


async def test_transcript_keeps_an_earlier_turn_that_repeats_the_reply() -> None:
    """Excluding the reviewed item by text also deletes any turn that matches it."""
    client = FakeSystemOne(CLEAN)
    reviewer = make_reviewer(client)
    agent = make_agent()

    ctx = agent.chat_ctx.copy()
    ctx.add_message(role="user", content="Yes")
    reply = ctx.add_message(role="assistant", content="Yes")
    await agent.update_chat_ctx(ctx)

    await reviewer.review(agent, "Yes", exclude_item_id=reply.id)

    state, _ = client.requests[0]
    assert state["transcript"] == [{"role": "user", "text": "Yes"}]


async def test_a_scalar_llm_node_result_still_flows_through_gate() -> None:
    """llm_node may resolve to a bare string, a single chunk, or None."""
    reviewer = make_reviewer(
        FakeSystemOne(CLEAN), checks=default_checks(gated_check_ids=["follows_instructions"])
    )

    async def scalar(_ctx: llm.ChatContext) -> str:
        return "Hello there."

    out = await reviewer.gate(
        make_agent(), llm.ChatContext.empty(), [], None, draft=lambda c: scalar(c)
    )
    assert await collect(out) == ["Hello there."]


async def test_a_none_llm_node_result_yields_nothing_rather_than_raising() -> None:
    reviewer = make_reviewer(FakeSystemOne(CLEAN))

    async def nothing(_ctx: llm.ChatContext) -> None:
        return None

    out = await reviewer.gate(
        make_agent(), llm.ChatContext.empty(), [], None, draft=lambda c: nothing(c)
    )
    assert await collect(out) == []


async def test_an_error_body_is_tagged_as_pii(caplog) -> None:
    """A TypeSafe error stringifies its response body, which can echo the prompt."""
    reviewer = make_reviewer(FakeSystemOne(CLEAN, fail=True))
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.typesafe"):
        await reviewer.review(make_agent(), "anything")

    [rec] = [r for r in caplog.records if "went unjudged" in r.message]
    assert getattr(rec, "lk.pii.error") == "typesafe is down"
    assert not hasattr(rec, "error")
    assert rec.error_type == "RuntimeError"


# --- history --------------------------------------------------------------


async def test_history_records_every_verdict_with_the_reply_it_judged() -> None:
    reviewer = make_reviewer(FakeSystemOne([CLEAN, OFF_COURSE]))
    agent = make_agent()

    await reviewer.review(agent, "Let me check that.")
    await reviewer.review(agent, "That will be $40.")

    assert [v.reviewed_reply for v in reviewer.results] == [
        "Let me check that.",
        "That will be $40.",
    ]
    assert [v.needs_correction for v in reviewer.results] == [False, True]
    # the probabilities survive, which is the point of keeping them
    assert reviewer.results[1].answers["follows_instructions"]["noul"] == 0.05


async def test_history_is_bounded() -> None:
    from livekit.plugins.typesafe.reviewer import _RESULTS_LIMIT

    reviewer = make_reviewer(FakeSystemOne(CLEAN))
    agent = make_agent()
    for _ in range(_RESULTS_LIMIT + 5):
        await reviewer.review(agent, "Sure.")

    assert len(reviewer.results) == _RESULTS_LIMIT


async def test_transcript_excludes_the_prompt_and_past_nudges() -> None:
    """`instructions` already carries the prompt; `transcript` is what was said.

    Both the agent's instructions and every nudge this plugin injects live in
    chat_ctx as system messages. Letting them into `transcript` would repeat
    the whole prompt in every request and feed the reviewer its own notes back
    as conversation.
    """
    client = FakeSystemOne(CLEAN)
    reviewer = make_reviewer(client)
    agent = make_agent()

    ctx = agent.chat_ctx.copy()
    ctx.add_message(role="system", content="You are a support agent. Never quote a price.")
    ctx.add_message(role="user", content="How much for a screen?")
    ctx.add_message(role="assistant", content="That will be $40.")
    ctx.add_message(role="system", content=f"{NUDGE_PREFIX} it broke a rule. ")
    await agent.update_chat_ctx(ctx)

    await reviewer.review(agent, "A human will follow up on pricing.")

    state, _ = client.requests[0]
    assert state["transcript"] == [
        {"role": "user", "text": "How much for a screen?"},
        {"role": "assistant", "text": "That will be $40."},
    ]
    assert state["instructions"] == "You are a support agent. Never quote a price."


async def test_history_turns_counts_spoken_turns_only() -> None:
    client = FakeSystemOne(CLEAN)
    reviewer = make_reviewer(client, history_turns=2)
    agent = make_agent()

    ctx = agent.chat_ctx.copy()
    ctx.add_message(role="system", content="noise")
    for i in range(4):
        ctx.add_message(role="user", content=f"u{i}")
    await agent.update_chat_ctx(ctx)

    await reviewer.review(agent, "ok")

    state, _ = client.requests[0]
    assert [t["text"] for t in state["transcript"]] == ["u2", "u3"]


# --- logging --------------------------------------------------------------


async def test_verdict_records_which_model_answered_and_what_it_cost() -> None:
    """An alias moves under you; the version behind a tuned threshold must be kept."""
    reviewer = make_reviewer(FakeSystemOne(CLEAN))
    verdict = await reviewer.review(make_agent(), "Sure.")

    assert verdict.model == "jev-1.13.0"
    assert verdict.usage == {"input_tokens": 296, "output_tokens": 20}
    assert verdict.evaluated is True


async def test_a_failed_check_is_marked_unjudged_not_clean() -> None:
    """`ok` alone cannot tell "nothing was wrong" from "nobody looked"."""
    reviewer = make_reviewer(FakeSystemOne(CLEAN, fail=True))
    verdict = await reviewer.review(make_agent(), "anything")

    assert not verdict.needs_correction and not verdict.evaluated
    assert reviewer.results[-1] is verdict  # still recorded, so gaps are visible


async def test_debug_line_carries_every_check_value(caplog) -> None:
    reviewer = make_reviewer(FakeSystemOne(dict(OFF_COURSE, expected_tool=choice("lookup_order"))))
    with caplog.at_level(logging.DEBUG, logger="livekit.plugins.typesafe"):
        await reviewer.review(make_agent(tools=[lookup_order]), "That will be $40.")

    [line] = [r.message for r in caplog.records if r.message.startswith("check ")]
    # a check that triggered_checks is flagged; every value is present either way
    assert "!follows_instructions=0.05" in line
    assert "advances_task=0.95" in line and "!advances_task" not in line
    assert "!expected_tool=lookup_order@0.90" in line
    assert "!severity=2.40@0.90" in line
    assert "jev-1.13.0" in line and "296tok" in line


async def test_trip_is_logged_at_info_with_the_reply_marked_as_pii(caplog) -> None:
    reviewer = make_reviewer(FakeSystemOne(OFF_COURSE))
    agent = make_agent()
    with caplog.at_level(logging.INFO, logger="livekit.plugins.typesafe"):
        verdict = await reviewer.review(agent, "That will be $40.")
        await reviewer._apply_nudge(agent, verdict)

    [rec] = [r for r in caplog.records if r.message == "nudging the agent back on course"]
    assert rec.levelno == logging.INFO
    assert rec.triggered_checks == ["follows_instructions", "unsupported_claim", "severity"]
    # caller-adjacent text must carry the lk.pii prefix the repo redacts on
    assert getattr(rec, "lk.pii.reviewed_reply") == "That will be $40."


async def test_an_outage_warns_with_enough_to_diagnose_it(caplog) -> None:
    reviewer = make_reviewer(FakeSystemOne(CLEAN, fail=True))
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.typesafe"):
        await reviewer.review(make_agent(), "anything")

    [rec] = [r for r in caplog.records if "went unjudged" in r.message]
    assert getattr(rec, "lk.pii.error") == "typesafe is down"
    assert rec.checks == ["follows_instructions", "unsupported_claim", "advances_task", "severity"]
    assert rec.state_chars > 0  # the 32k state cap is the usual 422


async def test_a_broken_on_verdict_callback_does_not_break_the_call(caplog) -> None:
    def boom(_verdict: Verdict) -> None:
        raise RuntimeError("metrics backend exploded")

    reviewer = make_reviewer(FakeSystemOne(CLEAN), on_verdict=boom)
    with caplog.at_level(logging.ERROR, logger="livekit.plugins.typesafe"):
        verdict = await reviewer.review(make_agent(), "Sure.")

    assert not verdict.needs_correction
    assert any("on_verdict callback raised" in r.message for r in caplog.records)


async def test_modality_instructions_are_rendered_not_passed_as_an_object() -> None:
    """`Agent(instructions=...)` also accepts an `Instructions` object."""
    from livekit.agents.llm.chat_context import Instructions

    client = FakeSystemOne(CLEAN)
    reviewer = make_reviewer(client)
    agent = Agent(instructions=Instructions("Be brief.", audio="Keep it to one sentence."))

    await reviewer.review(agent, "Sure.")

    state, _ = client.requests[0]
    assert isinstance(state["instructions"], str)
    assert "Be brief." in state["instructions"]
