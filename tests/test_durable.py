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
from livekit.agents.durable_scheduler import EffectException
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
# while unset, the slow charge does not return
CHARGE_GATE = asyncio.Event()


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


ENTERED: list[str] = []


class Confirm(AgentTask[bool]):
    def __init__(self) -> None:
        super().__init__(instructions="Ask the caller to confirm.")

    async def on_enter(self) -> None:
        ENTERED.append(self.id)


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
    global CHARGE_GATE
    CHARGE_GATE = asyncio.Event()  # an event binds to the loop of its first wait
    CALLS.clear()
    ENTERED.clear()
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
    # the task goes on where it was rather than entering again
    assert ENTERED == ["confirm"]
    task.complete(True)
    await _until(lambda: _outputs(resumed))
    (output,) = _outputs(resumed)
    assert (output.call_id, output.output) == ("call_1", "changed")
    await _until(lambda: isinstance(resumed.current_agent, ConfirmingDesk))
    await resumed.aclose()
    await crashed.aclose()




async def slow_charge(key: str) -> str:
    CALLS.append(("charge", key))
    await CHARGE_GATE.wait()
    return "charged"


class SlowDesk(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You book seats.")

    @function_tool(flags=ToolFlag.DURABLE)
    async def book(self, ctx: RunContext) -> str:
        """Book a seat."""
        charged = await EffectCall(slow_charge(ctx.idempotency_key))
        held = await EffectCall(hold(ctx.idempotency_key))
        return f"{charged}, {held}"


class _GatedStore(store.LocalStore):
    """Holds the next checkpoint's batch until ``gate`` opens, the way a slow round trip does."""

    def __init__(self, directory: pathlib.Path) -> None:
        super().__init__(directory, lease_ttl=LEASE_TTL)
        self.hold_next = False
        self.gate = asyncio.Event()

    async def _connect(self, database_id: str) -> Any:
        executor = await super()._connect(database_id)
        batch = executor.batch

        async def gated_batch(*statements: Any) -> Any:
            if self.hold_next and any(
                sql.startswith("UPDATE sessions SET current_agent_id") for sql, _ in statements
            ):
                self.hold_next = False
                await self.gate.wait()
            return await batch(*statements)

        executor.batch = gated_batch  # type: ignore[method-assign]
        return executor


@pytest.mark.usefixtures("database")
async def test_a_slow_checkpoint_does_not_rewind_a_later_boundary(tmp_path: pathlib.Path) -> None:
    gated = _GatedStore(tmp_path / "gated")
    database = Database(gated, await gated.create_database())
    crashed = AgentSession(llm=_llm("book"))
    await crashed.start(agent=SlowDesk(), persist=database.session("s1"))
    crashed.generate_reply(user_input="go")
    await _until(lambda: ("charge", "call_1:0") in CALLS)

    # a checkpoint reads the frame from before the charge, and lands after the charge resolved
    gated.hold_next = True
    checkpoint = asyncio.create_task(crashed._persistence.checkpoint())  # type: ignore[union-attr]
    await asyncio.sleep(0.05)
    CHARGE_GATE.set()
    await asyncio.sleep(0.1)
    gated.gate.set()
    await checkpoint
    await _until(lambda: ("hold", "call_1:1") in CALLS)

    RELEASED.set()
    resumed = AgentSession(llm=_llm("book"))
    await resumed.start(agent=SlowDesk(), persist=database.session("s1"))
    await _until(lambda: _outputs(resumed))
    # the charge resolved before the crash, so the resume goes on from after it
    assert CALLS.count(("charge", "call_1:0")) == 1
    await resumed.aclose()
    await crashed.aclose()
    await gated.aclose()


async def lookup(key: str) -> str:
    CALLS.append(("lookup", key))
    return "found"


class TwoToolDesk(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You book seats.")

    @function_tool(flags=ToolFlag.DURABLE)
    async def find(self, ctx: RunContext) -> str:
        """Find the booking."""
        return await EffectCall(lookup(ctx.idempotency_key))

    @function_tool(flags=ToolFlag.DURABLE)
    async def book(self, ctx: RunContext) -> str:
        """Book a seat."""
        charged = await EffectCall(slow_charge(ctx.idempotency_key))
        held = await EffectCall(hold(ctx.idempotency_key))
        return f"{charged}, {held}"


async def test_a_durable_tool_that_finished_beside_a_running_one_survives(
    database: Database,
) -> None:
    calls = [_tool_call("find", "call_1"), _tool_call("book", "call_2")]
    crashed = AgentSession(
        llm=_AnsweringLLM(fake_responses=[_says("go", "", calls=calls)], fallbacks=["Done."])
    )
    await crashed.start(agent=TwoToolDesk(), persist=database.session("s1"))
    crashed.generate_reply(user_input="go")
    await _until(lambda: ("charge", "call_2:0") in CALLS and ("lookup", "call_1:0") in CALLS)
    await asyncio.sleep(0.05)
    # the booking writes its frame after the lookup finished, and the hold never returns, so
    # the step's outputs are never committed
    CHARGE_GATE.set()
    await _until(lambda: ("hold", "call_2:1") in CALLS)

    RELEASED.set()
    resumed = AgentSession(llm=_AnsweringLLM(fake_responses=[], fallbacks=["Done."]))
    await resumed.start(agent=TwoToolDesk(), persist=database.session("s1"))
    await _until(lambda: len(_outputs(resumed)) == 2)
    # the finished lookup is answered from its frame, without running again
    assert {o.call_id: o.output for o in _outputs(resumed)} == {
        "call_1": "found",
        "call_2": "charged, held",
    }
    assert CALLS.count(("lookup", "call_1:0")) == 1
    await resumed.aclose()
    await crashed.aclose()


async def refuse(key: str) -> str:
    CALLS.append(("refuse", key))
    raise ValueError("the seat is taken")


class RetryingDesk(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You book seats.")

    @function_tool(flags=ToolFlag.DURABLE)
    async def book(self, ctx: RunContext) -> str:
        """Book a seat."""
        try:
            await EffectCall(refuse(ctx.idempotency_key))
        except EffectException as e:
            refused = str(e)
        held = await EffectCall(hold(ctx.idempotency_key))
        return f"{refused}, {held}"


async def test_a_frame_written_after_a_failed_effect_resumes(database: Database) -> None:
    crashed = AgentSession(llm=_llm("book"))
    await crashed.start(agent=RetryingDesk(), persist=database.session("s1"))
    crashed.generate_reply(user_input="go")
    await _until(lambda: ("hold", "call_1:1") in CALLS)

    RELEASED.set()
    resumed = AgentSession(llm=_llm("book"))
    await resumed.start(agent=RetryingDesk(), persist=database.session("s1"))
    await _until(lambda: _outputs(resumed))
    (output,) = _outputs(resumed)
    assert output.output == "ValueError: the seat is taken, held"
    assert CALLS.count(("refuse", "call_1:0")) == 1
    await resumed.aclose()
    await crashed.aclose()


async def test_a_worker_that_lost_the_session_stops_its_durable_tool(database: Database) -> None:
    stale = AgentSession(llm=_llm("book"))
    await stale.start(agent=SlowDesk(), persist=database.session("s1"))
    stale.generate_reply(user_input="go")
    await _until(lambda: ("charge", "call_1:0") in CALLS)

    # the first worker stalls in the charge past its lease, and a second one takes the session
    owner = AgentSession(llm=_llm("book"))
    await owner.start(agent=SlowDesk(), persist=database.session("s1"))
    await _until(lambda: CALLS.count(("charge", "call_1:0")) == 2)
    CHARGE_GATE.set()
    await _until(lambda: ("hold", "call_1:1") in CALLS)
    await asyncio.sleep(0.2)
    # the stale worker's boundary write is fenced, so it sends nothing after it
    assert CALLS.count(("hold", "call_1:1")) == 1
    await owner.aclose()
    await stale.aclose()
