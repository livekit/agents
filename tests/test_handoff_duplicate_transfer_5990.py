"""Regression test for duplicate same-target handoff tool calls (livekit/agents#5990).

A real LLM can emit the same handoff tool call more than once in a single turn
(parallel / duplicate tool calls). Every call returns the same target ``Agent``, so the
tool-output loop in ``AgentActivity`` sees more than one ``AgentTask`` and the
``"expected to receive only one AgentTask"`` guard previously set
``ignore_task_switch=True`` -- discarding the handoff entirely. ``update_agent`` was never
called, the target agent never started, and the follow-up tool turn was stranded.

The scripted ``FakeLLM`` in the other handoff tests emits exactly one ``transfer`` call, so
this went unseen; only a real LLM produced the duplicates. Scripting two identical
``transfer`` calls reproduces it deterministically -- no keys, audio, or LiveKit server
needed.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from livekit.agents import Agent, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 30


class Specialist(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="specialist; call search per property query")
        self.searches: list[str] = []

    @function_tool
    async def search(self, ctx: RunContext, query: str) -> str:
        """Search listings."""
        self.searches.append(query)
        return f"no listings for {query}"


class Root(Agent):
    def __init__(self, specialist: Specialist) -> None:
        super().__init__(instructions="receptionist; transfer on property ask")
        self._s = specialist

    @function_tool
    async def transfer(self, ctx: RunContext) -> Agent:
        """Hand off to the specialist."""
        return self._s


def _call(name: str, args: str, cid: str) -> FunctionToolCall:
    return FunctionToolCall(name=name, arguments=args, call_id=cid)


async def test_duplicate_transfer_calls_complete_handoff(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two identical ``transfer`` calls in one turn must still complete the handoff.

    The pre-fix guard treated the second same-target ``AgentTask`` as an error and aborted
    the switch, so the Specialist never ran and its follow-up ``search`` was stranded.
    """
    specialist = Specialist()
    root = Root(specialist)

    actions = FakeActions()
    # Handoff turn: the LLM emits `transfer` TWICE in one turn -- the #5990 trigger
    # (parallel duplicate handoff calls, both returning the same Specialist instance).
    actions.add_user_speech(0.5, 1.4, "transfer me to a specialist", stt_delay=0.1)
    actions.add_llm("", tool_calls=[_call("transfer", "{}", "1"), _call("transfer", "{}", "2")])
    # Follow-up turn the specialist must handle (the reporter's stranded tool turn).
    actions.add_user_speech(2.0, 2.9, "please search oak road", stt_delay=0.1)
    actions.add_llm(
        "",
        input="please search oak road",
        tool_calls=[_call("search", '{"query": "oak road"}', "3")],
    )
    actions.add_llm("nothing on oak road", input="no listings for oak road", duration=0.4)
    actions.add_tts(0.6, input="nothing on oak road")

    session = create_session(actions, speed_factor=5.0, extra_kwargs={"max_tool_steps": 3})

    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(run_session(session, root, drain_delay=6), timeout=SESSION_TIMEOUT)

    # The duplicate same-target transfer calls must still complete the handoff...
    assert type(session.current_agent).__name__ == "Specialist"
    # ...so the specialist's follow-up tool actually executes (the #5990 symptom).
    assert specialist.searches == ["oak road"]
