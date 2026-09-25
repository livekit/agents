"""Durable tools persisted on LocalStore: a frame captured at a boundary by the save on close."""

from __future__ import annotations

import asyncio
import logging
import pathlib
from collections.abc import AsyncIterator
from typing import Any

import pytest

from livekit.agents import (
    Agent,
    AgentSession,
    AgentTask,
    CloseReason,
    EffectCall,
    RunContext,
    function_tool,
    store,
)
from livekit.agents.llm import ToolError, ToolFlag
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.durable_tool import EffectException
from livekit.durable import registry

from .test_a2a_runner import _AnsweringLLM, _says, _tool_call
from .test_store import Database

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

# what each effect ran, in order, across both sessions
CALLS: list[str] = []
# while unset, the hold and the slow charge do not return
HOLD_GATE = asyncio.Event()
CHARGE_GATE = asyncio.Event()


async def charge() -> str:
    CALLS.append("charge")
    return "charged"


async def hold() -> str:
    CALLS.append("hold")
    await HOLD_GATE.wait()
    return "held"


ENTERED: list[str] = []


class Desk(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You book seats.")

    async def on_enter(self) -> None:
        ENTERED.append(self.id)

    @function_tool(flags=ToolFlag.DURABLE)
    async def book(self, ctx: RunContext, flight: str) -> str:
        """Book a seat on a flight."""
        charged = await EffectCall(charge())
        held = await EffectCall(hold())
        return f"{flight}: {charged}, {held}"


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
    global HOLD_GATE, CHARGE_GATE
    # an event binds to the loop of its first wait
    HOLD_GATE, CHARGE_GATE = asyncio.Event(), asyncio.Event()
    CALLS.clear()
    ENTERED.clear()
    local = store.LocalStore(tmp_path)
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


async def _close_in_hold(session: AgentSession) -> None:
    """Close the session while the hold is in flight, which the close waits out."""
    await _until(lambda: "hold" in CALLS)
    closing = asyncio.create_task(session.aclose())
    await asyncio.sleep(0.05)
    assert not closing.done()
    HOLD_GATE.set()
    await closing
    HOLD_GATE.clear()


async def test_a_tool_at_a_boundary_survives_close_and_resume(database: Database) -> None:
    first = AgentSession(llm=_llm("book", '{"flight": "NW812"}'))
    await first.start(agent=Desk(), persist=database.session("s1"))
    first.generate_reply(user_input="go")
    await _close_in_hold(first)
    (row,) = await database.rows("SELECT durable_state FROM agents WHERE agent_id = 'desk'")
    assert row["durable_state"]

    resumed = AgentSession(llm=_llm("book"))
    await resumed.start(agent=Desk(), persist=database.session("s1"))
    await _until(lambda: _outputs(resumed))
    (output,) = _outputs(resumed)
    assert (output.call_id, output.output, output.is_error) == (
        "call_1",
        "NW812: charged, held",
        False,
    )
    # the close waited for the hold, so the resumed tool sends nothing again
    assert CALLS == ["charge", "hold"]
    # the agent resumed with its frame is not entered again
    assert ENTERED == ["desk"]

    # an answered tool has no frame left to save
    await resumed.aclose()
    (row,) = await database.rows("SELECT durable_state FROM agents WHERE agent_id = 'desk'")
    assert row["durable_state"] == b""


async def slow_charge() -> str:
    CALLS.append("charge")
    await CHARGE_GATE.wait()
    return "charged"


class SlowDesk(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You book seats.")

    @function_tool(flags=ToolFlag.DURABLE)
    async def book(self, ctx: RunContext) -> str:
        """Book a seat."""
        charged = await EffectCall(slow_charge())
        held = await EffectCall(hold())
        return f"{charged}, {held}"


async def test_a_save_holds_a_tool_at_its_next_boundary(database: Database) -> None:
    session = AgentSession(llm=_llm("book"))
    await session.start(agent=SlowDesk(), persist=database.session("s1"))
    HOLD_GATE.set()
    session.generate_reply(user_input="go")
    await _until(lambda: "charge" in CALLS)

    # the charge is in flight when the save starts, so the save waits for it
    saving = asyncio.create_task(session.save())
    await asyncio.sleep(0.05)
    assert not saving.done()
    CHARGE_GATE.set()
    await saving
    # the save captured the tool after the charge, and the hold waited for it to land
    assert CALLS == ["charge"]
    (row,) = await database.rows("SELECT durable_state FROM agents WHERE agent_id = 'slow_desk'")
    assert row["durable_state"]
    await _until(lambda: _outputs(session))
    assert CALLS == ["charge", "hold"]
    await session.aclose()


async def refuse_with_tool_error() -> str:
    CALLS.append("refuse")
    raise ToolError("the seat is taken")


class RefusingDesk(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You book seats.")

    @function_tool(flags=ToolFlag.DURABLE)
    async def book(self, ctx: RunContext) -> str:
        """Book a seat."""
        return await EffectCall(refuse_with_tool_error())


async def test_an_effect_raising_a_tool_error_reaches_the_model_as_it(database: Database) -> None:
    session = AgentSession(llm=_llm("book"))
    await session.start(agent=RefusingDesk(), persist=database.session("s1"))
    session.generate_reply(user_input="go")
    await _until(lambda: _outputs(session))
    (output,) = _outputs(session)
    assert (output.is_error, output.output) == (True, "the seat is taken")
    await session.aclose()


async def refuse() -> str:
    CALLS.append("refuse")
    raise ValueError("the seat is taken")


class RetryingDesk(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You book seats.")

    @function_tool(flags=ToolFlag.DURABLE)
    async def book(self, ctx: RunContext) -> str:
        """Book a seat."""
        try:
            await EffectCall(refuse())
        except EffectException as e:
            refused = str(e)
        try:
            await EffectCall(refuse_with_tool_error())
        except ToolError as e:
            refused += f" / {e.message}"
        held = await EffectCall(hold())
        return f"{refused}, {held}"


async def test_a_frame_saved_after_failed_effects_resumes(database: Database) -> None:
    first = AgentSession(llm=_llm("book"))
    await first.start(agent=RetryingDesk(), persist=database.session("s1"))
    first.generate_reply(user_input="go")
    await _close_in_hold(first)

    resumed = AgentSession(llm=_llm("book"))
    await resumed.start(agent=RetryingDesk(), persist=database.session("s1"))
    await _until(lambda: _outputs(resumed))
    (output,) = _outputs(resumed)
    assert output.output == "ValueError: the seat is taken / the seat is taken, held"
    assert CALLS == ["refuse", "refuse", "hold"]
    await resumed.aclose()


async def test_a_frame_whose_code_changed_ends_in_a_tool_error(database: Database) -> None:
    first = AgentSession(llm=_llm("book", '{"flight": "NW812"}'))
    await first.start(agent=Desk(), persist=database.session("s1"))
    first.generate_reply(user_input="go")
    await _close_in_hold(first)

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
    await _until(
        lambda: any(
            m.role == "assistant" and m.text_content == "Done." for m in resumed.history.messages()
        )
    )
    await resumed.aclose()


async def test_a_task_awaited_from_a_durable_tool_resumes(
    database: Database, caplog: pytest.LogCaptureFixture
) -> None:
    first = AgentSession(llm=_llm("change"))
    await first.start(agent=ConfirmingDesk(), persist=database.session("s1"))
    first.generate_reply(user_input="go")
    await _until(lambda: isinstance(first.current_agent, Confirm))
    with caplog.at_level(logging.ERROR, logger="livekit.agents"):
        await first.aclose()
    # the close stops the tool awaiting the task and cancels the task, in that order
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

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


async def test_a_rehydrate_that_fails_midway_leaves_no_running_frame(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = AgentSession(llm=_llm("change"))
    await first.start(agent=ConfirmingDesk(), persist=database.session("s1"))
    first.generate_reply(user_input="go")
    await _until(lambda: isinstance(first.current_agent, Confirm))
    await first.aclose()

    # the desk's frame restores, and the task above it fails to
    rehydrate = AgentActivity._rehydrate
    restored: list[AgentActivity] = []

    async def failing(self: AgentActivity, tasks: list[Any]) -> list[str]:
        if isinstance(self.agent, Confirm):
            raise RuntimeError("the task's toolset did not set up")
        restored.append(self)
        return await rehydrate(self, tasks)

    monkeypatch.setattr(AgentActivity, "_rehydrate", failing)
    desk = ConfirmingDesk()
    resumed = AgentSession(llm=_llm("change"))
    with pytest.raises(RuntimeError, match="did not set up"):
        await resumed.start(agent=desk, persist=database.session("s1"))
    await asyncio.sleep(0.05)

    (activity,) = restored
    assert desk._activity is None and activity._restored_tools == []
    assert not any(t.get_name() == "AgentActivity.resume_durable_tool" for t in asyncio.all_tasks())
    assert activity._tool_executor._durable_tasks == {}


async def note() -> str:
    CALLS.append("note")
    return "noted"


class NotingDesk(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You change bookings and take notes.")

    @function_tool(flags=ToolFlag.DURABLE)
    async def take_note(self, ctx: RunContext) -> str:
        """Take a note."""
        return await EffectCall(note())

    @function_tool(flags=ToolFlag.DURABLE)
    async def change(self, ctx: RunContext) -> str:
        """Change the booking once the caller confirms."""
        confirmed = await EffectCall(Confirm())
        return "changed" if confirmed else "kept"


async def test_a_tool_that_finished_beside_one_awaiting_a_task_is_saved(
    database: Database,
) -> None:
    def both() -> _AnsweringLLM:
        calls = [_tool_call("take_note", "call_note"), _tool_call("change", "call_change")]
        return _AnsweringLLM(
            fake_responses=[_says("go", "", calls=calls)], fallbacks=["Done.", "Done again."]
        )

    first = AgentSession(llm=both())
    await first.start(agent=NotingDesk(), persist=database.session("s1"))
    first.generate_reply(user_input="go")
    await _until(lambda: isinstance(first.current_agent, Confirm) and "note" in CALLS)
    # the step commits the note only once the change returns, which the close cuts short
    await first.aclose()

    resumed = AgentSession(llm=both())
    await resumed.start(agent=NotingDesk(), persist=database.session("s1"))
    task = resumed.current_agent
    assert isinstance(task, Confirm)
    task.complete(True)
    await _until(lambda: len(_outputs(resumed)) == 2)
    assert {(o.call_id, o.output) for o in _outputs(resumed)} == {
        ("call_note", "noted"),
        ("call_change", "changed"),
    }
    assert CALLS == ["note"]
    await resumed.aclose()


async def test_a_start_that_fails_after_the_rehydrate_lets_everything_go(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = AgentSession(llm=_llm("change"))
    await first.start(agent=ConfirmingDesk(), persist=database.session("s1"))
    first.generate_reply(user_input="go")
    await _until(lambda: isinstance(first.current_agent, Confirm))
    await first.aclose()

    async def failing(self: AgentSession, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the room did not connect")

    monkeypatch.setattr(AgentSession, "_update_activity_task", failing)
    desk = ConfirmingDesk()
    resumed = AgentSession(llm=_llm("change"))
    with pytest.raises(RuntimeError, match="did not connect"):
        await resumed.start(agent=desk, persist=database.session("s1"))
    await asyncio.sleep(0.05)

    assert desk._activity is None
    assert not any(t.get_name() == "AgentActivity.resume_durable_tool" for t in asyncio.all_tasks())
    # the handle is let go, so the session reads as closed and its connection is shut
    (row,) = await database.rows("SELECT closed_at FROM sessions")
    assert row["closed_at"] is not None


async def test_a_save_during_a_start_that_fails_keeps_the_restored_frame(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = AgentSession(llm=_llm("book", '{"flight": "NW812"}'))
    await first.start(agent=Desk(), persist=database.session("s1"))
    first.generate_reply(user_input="go")
    await _close_in_hold(first)

    load = store.StoredSession.load

    async def load_then_save(self: store.StoredSession) -> Any:
        stored = await load(self)
        # before the session has an agent it holds only what it loaded
        await resumed.save()
        return stored

    async def save_then_fail(self: AgentSession, *args: Any, **kwargs: Any) -> None:
        (row,) = await database.rows("SELECT current_agent_id FROM sessions")
        assert row["current_agent_id"] == "desk"
        # the restored frame has not started yet, and is still the session's
        await self.save()
        raise RuntimeError("the room did not connect")

    monkeypatch.setattr(store.StoredSession, "load", load_then_save)
    monkeypatch.setattr(AgentSession, "_update_activity_task", save_then_fail)
    resumed = AgentSession(llm=_llm("book"))
    with pytest.raises(RuntimeError, match="did not connect"):
        await resumed.start(agent=Desk(), persist=database.session("s1"))

    monkeypatch.undo()
    again = AgentSession(llm=_llm("book"))
    await again.start(agent=Desk(), persist=database.session("s1"))
    await _until(lambda: _outputs(again))
    (output,) = _outputs(again)
    assert output.output == "NW812: charged, held"
    assert CALLS == ["charge", "hold"]
    await again.aclose()


async def test_a_close_cut_short_in_an_effect_still_saves(
    database: Database, caplog: pytest.LogCaptureFixture
) -> None:
    session = AgentSession(llm=_llm("book", '{"flight": "NW812"}'))
    await session.start(agent=Desk(), persist=database.session("s1"))
    session.generate_reply(user_input="go")
    await _until(lambda: "hold" in CALLS)
    # the hold never returns, so the job's guard on the close is what ends it
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(session.aclose(), 0.2)
    assert any("so it is lost" in r.getMessage() for r in caplog.records)
    # what the guard cut short, the process exit would end
    await session._teardown_activity(reason=CloseReason.JOB_SHUTDOWN, drain=False)

    (row,) = await database.rows("SELECT closed_at FROM sessions")
    assert row["closed_at"] is not None
    messages = await database.rows(
        "SELECT item FROM chat_items WHERE owner = 'session' AND item LIKE '%\"go\"%'"
    )
    assert messages
    (agent,) = await database.rows("SELECT durable_state FROM agents WHERE agent_id = 'desk'")
    assert agent["durable_state"] == b""
