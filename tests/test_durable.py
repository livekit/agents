"""Durable tools persisted on LocalStore: a frame written at each effect, resumed by a new worker."""

from __future__ import annotations

import asyncio
import pathlib
from collections.abc import AsyncIterator
from typing import Any

import pytest

from livekit.agents import (
    Agent,
    AgentSession,
    AgentTask,
    EffectCall,
    RunContext,
    function_tool,
    store,
)
from livekit.agents.llm import ToolFlag
from livekit.durable import registry

from .test_a2a_runner import _AnsweringLLM, _says, _tool_call
from .test_store import Database

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

LEASE_TTL = 0.3

# what each effect saw, in order, across both workers
CALLS: list[tuple[str, str]] = []
# while unset, the hold effect never returns, the way a worker killed mid-effect never does
RELEASED = asyncio.Event()


async def charge(key: str) -> str:
    CALLS.append(("charge", key))
    return "charged"


async def hold(key: str) -> str:
    CALLS.append(("hold", key))
    if not RELEASED.is_set():
        await asyncio.Event().wait()
    return "held"


class Desk(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You book seats.")

    @function_tool(flags=ToolFlag.DURABLE)
    async def book(self, ctx: RunContext, flight: str) -> str:
        """Book a seat on a flight."""
        charged = await EffectCall(charge(ctx.idempotency_key))
        held = await EffectCall(hold(ctx.idempotency_key))
        return f"{flight}: {charged}, {held}"


class Confirm(AgentTask[bool]):
    def __init__(self) -> None:
        super().__init__(instructions="Ask the caller to confirm.")


class ConfirmingDesk(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You change bookings.")

    @function_tool(flags=ToolFlag.DURABLE)
    async def change(self, ctx: RunContext) -> str:
        """Change the booking once the caller confirms."""
        confirmed = await EffectCall(Confirm())
        return "changed" if confirmed else "kept"


@pytest.fixture
async def database(tmp_path: pathlib.Path) -> AsyncIterator[Database]:
    CALLS.clear()
    RELEASED.clear()
    local = store.LocalStore(tmp_path, lease_ttl=LEASE_TTL)
    yield Database(local, await local.create_database())
    await local.aclose()


def _llm(call: str, arguments: str = "{}") -> _AnsweringLLM:
    return _AnsweringLLM(
        fake_responses=[_says("go", "", calls=[_tool_call(call, "call_1", arguments)])],
        fallbacks=["Done.", "Done again."],
    )


async def _until(predicate: Any, timeout: float = 5.0) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(wait(), timeout)


def _outputs(session: AgentSession) -> list[Any]:
    return [i for i in session.history.items if i.type == "function_call_output"]


async def test_a_resume_reruns_only_the_effect_in_flight(database: Database) -> None:
    crashed = AgentSession(llm=_llm("book", '{"flight": "NW812"}'))
    await crashed.start(agent=Desk(), persist=database.session("s1"))
    crashed.generate_reply(user_input="go")
    await _until(lambda: ("hold", "call_1:1") in CALLS)

    # the frame was written once the charge resolved, before the hold was sent
    (row,) = await database.rows("SELECT durable_state FROM agents WHERE agent_id = 'desk'")
    assert row["durable_state"]

    # the first worker never closes: the second takes the session once its lease lapses
    RELEASED.set()
    resumed = AgentSession(llm=_llm("book"))
    await resumed.start(agent=Desk(), persist=database.session("s1"))
    await _until(lambda: _outputs(resumed))
    (output,) = _outputs(resumed)
    assert (output.call_id, output.output, output.is_error) == (
        "call_1",
        "NW812: charged, held",
        False,
    )
    # the charge ran once; the hold ran again under the same key, which is how it dedups
    assert CALLS == [("charge", "call_1:0"), ("hold", "call_1:1"), ("hold", "call_1:1")]

    # the answered call is not resumed a second time, and its frame is gone at the next checkpoint
    await resumed.aclose()
    (row,) = await database.rows("SELECT durable_state FROM agents WHERE agent_id = 'desk'")
    assert row["durable_state"] == b""
    again = AgentSession(llm=_llm("book"))
    await again.start(agent=Desk(), persist=database.session("s1"))
    assert len(_outputs(again)) == 1
    await again.aclose()
    await crashed.aclose()


async def test_a_frame_whose_code_changed_ends_in_a_tool_error(database: Database) -> None:
    crashed = AgentSession(llm=_llm("book", '{"flight": "NW812"}'))
    await crashed.start(agent=Desk(), persist=database.session("s1"))
    crashed.generate_reply(user_input="go")
    await _until(lambda: ("hold", "call_1:1") in CALLS)

    registered = registry.lookup_function(Desk.book._raw_func.__qualname__)  # type: ignore[attr-defined]
    code_hash = registered.hash
    registered.hash = "sha256:a-later-deploy"
    try:
        resumed = AgentSession(llm=_llm("book"))
        await resumed.start(agent=Desk(), persist=database.session("s1"))
        await _until(lambda: _outputs(resumed))
    finally:
        registered.hash = code_hash
    (output,) = _outputs(resumed)
    assert output.is_error and "code changed" in output.output
    # the call and its output land at the end, and the model is asked to reply to them
    assert resumed.history.items[-1].type != "function_call" or resumed.history.items[-1] is output
    assert ("hold", "call_1:1") in CALLS and CALLS.count(("hold", "call_1:1")) == 1
    await _until(
        lambda: any(
            m.role == "assistant" and m.text_content == "Done." for m in resumed.history.messages()
        )
    )
    await resumed.aclose()
    await crashed.aclose()


async def test_a_task_awaited_from_a_durable_tool_resumes(database: Database) -> None:
    crashed = AgentSession(llm=_llm("change"))
    await crashed.start(agent=ConfirmingDesk(), persist=database.session("s1"))
    crashed.generate_reply(user_input="go")
    await _until(lambda: isinstance(crashed.current_agent, Confirm))
    await crashed._persistence.checkpoint()  # type: ignore[union-attr]

    resumed = AgentSession(llm=_llm("change"))
    await resumed.start(agent=ConfirmingDesk(), persist=database.session("s1"))
    task = resumed.current_agent
    assert isinstance(task, Confirm)
    task.complete(True)
    await _until(lambda: _outputs(resumed))
    (output,) = _outputs(resumed)
    assert (output.call_id, output.output) == ("call_1", "changed")
    await _until(lambda: isinstance(resumed.current_agent, ConfirmingDesk))
    await resumed.aclose()
    await crashed.aclose()
