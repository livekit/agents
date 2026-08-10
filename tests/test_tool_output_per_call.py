"""Every tool call is answered, whatever the tool did.

An unanswered call is not a neutral omission: an LLM re-issues it on the next inference and
runs its side effects again, and a realtime model holds the turn open waiting for it. A tool
that has nothing to say says so with `reply_required` instead of leaving its call open.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, function_tool
from livekit.agents.llm import FunctionToolCall, StopResponse
from livekit.agents.voice.events import FunctionToolsExecutedEvent

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60.0


def _tool_turn(actions: FakeActions) -> None:
    actions.add_user_speech(0.5, 2.5, "Do the thing.")
    actions.add_llm(
        content="Working on it.",
        tool_calls=[FunctionToolCall(name="do_the_thing", arguments="{}", call_id="1")],
    )
    actions.add_tts(1.0)


async def _run(agent: Agent) -> tuple[AgentSession, list[FunctionToolsExecutedEvent]]:
    actions = FakeActions()
    _tool_turn(actions)
    session = create_session(actions)

    events: list[FunctionToolsExecutedEvent] = []
    session.on("function_tools_executed", events.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    return session, events


def _assert_answered_once(agent: Agent, session: AgentSession) -> None:
    for label, items in (
        ("agent chat_ctx", agent.chat_ctx.items),
        ("session history", session.history.items),
    ):
        calls = [i for i in items if i.type == "function_call"]
        outs = [i for i in items if i.type == "function_call_output"]
        assert len(calls) == 1, f"{label}: the call must be recorded once"
        assert len(outs) == 1, f"{label}: the call must be answered exactly once"
        assert outs[0].call_id == calls[0].call_id


async def test_stop_response_answers_the_call_and_asks_for_no_reply() -> None:
    """StopResponse means "say nothing", not "leave the call open"."""

    class StopAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        @function_tool
        async def do_the_thing(self) -> None:
            """Do the thing."""
            raise StopResponse

    agent = StopAgent()
    session, events = await _run(agent)

    _assert_answered_once(agent, session)
    output = events[0].function_call_outputs[0]
    assert output.output == ""
    assert not output.is_error
    assert not output.reply_required
    assert not events[0].has_tool_reply


async def test_invalid_output_answers_the_call_as_an_error() -> None:
    """A tool returning something unserializable is a failure the model can recover from."""

    class BadOutputAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        @function_tool
        async def do_the_thing(self) -> object:
            """Do the thing."""
            return object()

    agent = BadOutputAgent()
    session, events = await _run(agent)

    _assert_answered_once(agent, session)
    output = events[0].function_call_outputs[0]
    assert output.is_error
    assert "invalid output" in output.output


async def test_multiple_agents_answer_the_call_as_an_error() -> None:
    """Returning two agents is ambiguous, so the handoff is dropped and the call reports it."""

    class TwoAgentsAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        @function_tool
        async def do_the_thing(self) -> list[Agent]:
            """Do the thing."""
            return [Agent(instructions="first"), Agent(instructions="second")]

    agent = TwoAgentsAgent()
    session, events = await _run(agent)

    assert session.current_agent is agent, "an ambiguous handoff must not be applied"
    _assert_answered_once(agent, session)
    output = events[0].function_call_outputs[0]
    assert output.is_error
    assert "more than one agent" in output.output


async def test_bare_handoff_answers_the_call_and_asks_for_no_reply() -> None:
    """A handoff with nothing to say is answered, but the new agent speaks, not the old one."""

    class TransferAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        @function_tool
        async def do_the_thing(self) -> Agent:
            """Do the thing."""
            return Agent(instructions="You are the next agent.")

    agent = TransferAgent()
    session, events = await _run(agent)

    assert session.current_agent is not agent, "the handoff must be applied"
    output = events[0].function_call_outputs[0]
    assert output.output == ""
    assert not output.reply_required
