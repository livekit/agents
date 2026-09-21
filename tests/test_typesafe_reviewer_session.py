"""Reviewer attached to a real AgentSession.

The rest of the suite calls ``check()`` directly. This drives a full session so
the wiring under ``attach()`` — the event subscription, the tool bookkeeping,
and the nudge reaching the live agent — is exercised the way a call exercises
it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import Agent, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall
from livekit.plugins.typesafe import Reviewer, default_checks
from livekit.plugins.typesafe.reviewer import NUDGE_PREFIX

from .fake_session import FakeActions, create_session, run_session
from .test_typesafe_reviewer import CLEAN, OFF_COURSE, FakeSystemOne, choice

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


class _SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a support agent. Never quote a price.")

    @function_tool
    async def lookup_order(self, context: RunContext, order_id: str) -> str:
        """Look up the status of a customer order."""
        return "shipped"


def _nudges(agent: Agent) -> list[Any]:
    """Reviewer notes in the context, excluding the agent's own instructions."""
    return [
        i
        for i in agent.chat_ctx.items
        if getattr(i, "text_content", None) and i.text_content.startswith(NUDGE_PREFIX)
    ]


async def _drive(
    client: FakeSystemOne, reply: str, *, agent: Agent | None = None, **kwargs: Any
) -> tuple[Agent, Reviewer]:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "Where is my order?")
    actions.add_llm(content=reply)
    actions.add_tts(1.0)

    reviewer = Reviewer(_client=client, **kwargs)  # type: ignore[arg-type]
    session = create_session(actions, speed_factor=2.0)
    reviewer.attach(session)

    agent = agent or _SupportAgent()
    await asyncio.wait_for(run_session(session, agent, drain_delay=1.0), timeout=60)
    return agent, reviewer


async def test_attach_judges_the_reply_a_session_actually_produced() -> None:
    client = FakeSystemOne(CLEAN)
    agent, reviewer = await _drive(client, "Let me look that up for you.")

    assert client.calls == 1
    state, questions = client.requests[0]
    assert state["reviewed_reply"] == "Let me look that up for you."
    assert state["instructions"] == "You are a support agent. Never quote a price."
    assert [t["name"] for t in state["available_tools"]] == ["lookup_order"]
    assert state["transcript"][-1] == {"role": "user", "text": "Where is my order?"}
    assert "expected_tool" in questions

    assert [v.needs_correction for v in reviewer.results] == [False]
    assert _nudges(agent) == []  # a clean turn leaves the context alone


async def test_attach_nudges_the_live_agent_when_a_check_trips() -> None:
    agent, reviewer = await _drive(FakeSystemOne(OFF_COURSE), "That will be $40.")

    assert [v.triggered_checks for v in reviewer.results] == [
        ["follows_instructions", "unsupported_claim", "severity"]
    ]

    [nudge] = _nudges(agent)
    assert nudge.role == "system"
    assert "it broke a rule in your instructions" in nudge.text_content
    # the nudge lands after the reply it is correcting, so the next
    # generation sees both
    assert agent.chat_ctx.items.index(nudge) == len(agent.chat_ctx.items) - 1


async def test_attach_tracks_tools_the_session_really_called() -> None:
    """The expected_tool check must not fire on a tool the agent did call."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "Where is my order 123?")
    actions.add_llm(
        content="",
        tool_calls=[
            FunctionToolCall(name="lookup_order", arguments='{"order_id": "123"}', call_id="1")
        ],
    )
    actions.add_llm(content="It has shipped.", input="shipped")
    actions.add_tts(1.0)

    client = FakeSystemOne(dict(CLEAN, expected_tool=choice("lookup_order")))
    reviewer = Reviewer(_client=client)  # type: ignore[arg-type]
    session = create_session(actions, speed_factor=2.0)
    reviewer.attach(session)

    await asyncio.wait_for(run_session(session, _SupportAgent(), drain_delay=1.0), timeout=60)

    state, _ = client.requests[0]
    assert state["tools_used_this_turn"] == ["lookup_order"]
    assert [v.needs_correction for v in reviewer.results] == [False]


async def test_detach_stops_the_checks() -> None:
    client = FakeSystemOne(OFF_COURSE)
    reviewer = Reviewer(_client=client)  # type: ignore[arg-type]
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "Where is my order?")
    actions.add_llm(content="That will be $40.")
    actions.add_tts(1.0)

    session = create_session(actions, speed_factor=2.0)
    reviewer.attach(session)
    reviewer.detach()

    await asyncio.wait_for(run_session(session, _SupportAgent(), drain_delay=1.0), timeout=60)
    assert client.calls == 0


async def test_a_typesafe_outage_does_not_disturb_the_call() -> None:
    client = FakeSystemOne(CLEAN, fail=True)
    agent, reviewer = await _drive(client, "Let me look that up for you.")

    assert client.calls == 1
    assert _nudges(agent) == []
    # the reply the caller heard is still in the transcript, unaltered
    assert any(
        getattr(i, "role", None) == "assistant" and i.text_content == "Let me look that up for you."
        for i in agent.chat_ctx.items
    )


async def test_thresholds_from_default_checks_reach_a_live_session() -> None:
    answers = dict(CLEAN, unsupported_claim={"type": "noul", "noul": 0.75})

    _agent, strict = await _drive(FakeSystemOne(answers), "It arrives Tuesday.")
    assert strict.results[0].triggered_checks == ["unsupported_claim"]

    _agent, relaxed = await _drive(
        FakeSystemOne(answers),
        "It arrives Tuesday.",
        checks=default_checks(unsupported_claim=0.8),
    )
    assert not relaxed.results[0].needs_correction


async def test_gating_one_check_does_not_switch_the_others_off() -> None:
    """The two placements must compose over a real session.

    The gate judges only its own checks, so the committed item still has to run
    the rest. Keying the handoff by text alone silently disabled four of five.
    """

    class GatedAgent(_SupportAgent):
        def __init__(self, reviewer: Reviewer) -> None:
            super().__init__()
            self._reviewer = reviewer

        async def llm_node(self, chat_ctx, tools, model_settings):  # type: ignore[no-untyped-def]
            return await self._reviewer.gate(self, chat_ctx, tools, model_settings)

    # clears the gated check, violates an observe-only one
    answers = dict(CLEAN, unsupported_claim={"type": "noul", "noul": 0.97})
    client = FakeSystemOne(answers)
    reviewer = Reviewer(  # type: ignore[arg-type]
        _client=client, checks=default_checks(gated_check_ids=["follows_instructions"])
    )

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "When does order 123 arrive?")
    actions.add_llm(content="It arrives Tuesday.")
    actions.add_tts(1.0)

    session = create_session(actions, speed_factor=2.0)
    reviewer.attach(session)
    agent = GatedAgent(reviewer)

    await asyncio.wait_for(run_session(session, agent, drain_delay=1.5), timeout=60)

    # the gate cleared the draft, so the reply was spoken unchanged
    assert any(
        getattr(i, "role", None) == "assistant" and i.text_content == "It arrives Tuesday."
        for i in agent.chat_ctx.items
    )
    # and the observe-only check still caught it
    assert any("unsupported_claim" in v.triggered_checks for v in reviewer.results)
    assert len(_nudges(agent)) == 1


async def test_a_gate_correction_lands_after_the_reply_it_criticizes() -> None:
    """The note says "the response you just produced", so it must follow it.

    An invariant guard rather than a regression test: ChatContext.insert orders
    by created_at, so the reply sorts ahead of a later nudge even if one is
    applied early. Deferring to commit does not depend on that.
    """

    class GatedAgent(_SupportAgent):
        def __init__(self, reviewer: Reviewer) -> None:
            super().__init__()
            self._reviewer = reviewer

        async def llm_node(self, chat_ctx, tools, model_settings):  # type: ignore[no-untyped-def]
            return await self._reviewer.gate(self, chat_ctx, tools, model_settings)

    answers = dict(CLEAN, unsupported_claim={"type": "noul", "noul": 0.97})
    reviewer = Reviewer(  # type: ignore[arg-type]
        _client=FakeSystemOne(answers),
        checks=default_checks(gated_check_ids=["follows_instructions"]),
    )

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "When does order 123 arrive?")
    actions.add_llm(content="It arrives Tuesday.")
    actions.add_tts(1.0)

    session = create_session(actions, speed_factor=2.0)
    reviewer.attach(session)
    agent = GatedAgent(reviewer)
    await asyncio.wait_for(run_session(session, agent, drain_delay=1.5), timeout=60)

    texts = [
        i.text_content
        for i in agent.chat_ctx.items
        if getattr(i, "text_content", None) and getattr(i, "role", None) != "system"
    ]
    assert "It arrives Tuesday." in texts

    [nudge] = _nudges(agent)
    items = list(agent.chat_ctx.items)
    reply_at = next(
        n for n, i in enumerate(items) if getattr(i, "text_content", None) == "It arrives Tuesday."
    )
    assert items.index(nudge) > reply_at  # the correction follows the reply


async def test_a_handoff_during_a_review_does_not_reassign_the_reply() -> None:
    """The review is detached, so the session can move on while it runs.

    Resolving session.current_agent when the review resumes would judge the old
    agent's reply against the new agent's instructions, and write the
    correction into an agent that never said it.
    """
    client = FakeSystemOne(OFF_COURSE, delay=0.2)
    reviewer = Reviewer(_client=client)  # type: ignore[arg-type]

    author = Agent(instructions="AUTHOR AGENT instructions.")
    successor = Agent(instructions="SUCCESSOR AGENT instructions.")

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "Where is my order?")
    actions.add_llm(content="That will be $40.")
    actions.add_tts(1.0)

    session = create_session(actions, speed_factor=2.0)
    reviewer.attach(session)

    # hand off on the next tick, after the item event has been dispatched but
    # while the review is still waiting on the network
    @session.on("conversation_item_added")
    def _swap(ev) -> None:  # type: ignore[no-untyped-def]
        if getattr(ev.item, "role", None) == "assistant":
            asyncio.get_running_loop().call_soon(session.update_agent, successor)

    await asyncio.wait_for(run_session(session, author, drain_delay=2.0), timeout=60)

    assert client.calls >= 1
    state, _ = client.requests[0]
    assert state["instructions"] == "AUTHOR AGENT instructions."
    # the successor never produced that reply, so it is not corrected for it
    assert _nudges(successor) == []
