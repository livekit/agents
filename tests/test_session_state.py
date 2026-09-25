"""AgentSession.start(persist=...): what a session writes as it runs, and what it gets back."""

from __future__ import annotations

import asyncio
import logging
import pathlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool, store
from livekit.agents.beta.workflows import GetEmailTask
from livekit.agents.delegation import Delegate
from livekit.agents.llm import ChatContext

from .test_a2a_runner import _AnsweringLLM, _says, _tool_call
from .test_store import Database

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@dataclass
class Userdata:
    airline: str
    rebooked: list[str] = field(default_factory=list)


@dataclass
class Keyed:
    seats: dict[tuple[str, str], str]


class FareDesk(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You answer fare questions.")

    @function_tool
    async def rebook(self, ctx: RunContext[Userdata], flight: str) -> Agent:
        """Hand the caller to rebooking for a flight."""
        ctx.userdata.rebooked.append(flight)
        return Rebooking(flight=flight)


class Confirming(AgentTask[bool]):
    def __init__(self) -> None:
        super().__init__(instructions="Ask the caller to confirm.")


CONFIRMING = asyncio.Event()


class Rebooking(Agent):
    def __init__(self, *, flight: str) -> None:
        super().__init__(instructions=f"You rebook flight {flight}.")
        self._flight = flight

    @function_tool
    async def confirm(self, ctx: RunContext) -> str:
        """Ask the caller to confirm the change."""
        CONFIRMING.set()
        return "confirmed" if await Confirming() else "declined"


class Billing(Agent):
    def __init__(self, customer: object) -> None:
        super().__init__(instructions="You handle billing.")
        self.account = customer  # kept under another name, so the default cannot rebuild it


class Transferring(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You transfer to billing.")

    @function_tool
    async def to_billing(self, ctx: RunContext) -> Agent:
        """Hand the caller to billing."""
        return Billing(customer=object())


ENTERED: list[str] = []


class Greeter(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You greet the caller.")

    async def on_enter(self) -> None:
        ENTERED.append(self.id)


class Rebooker(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You rebook flights.")

    async def on_enter(self) -> None:
        ENTERED.append(self.id)


@pytest.fixture
async def database(tmp_path: pathlib.Path) -> AsyncIterator[Database]:
    local = store.LocalStore(tmp_path)
    yield Database(local, await local.create_database())
    await local.aclose()


class _RecordingLLM(_AnsweringLLM):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.seen: list[str] = []

    def chat(self, *, chat_ctx: Any, **kwargs: Any) -> Any:
        self.seen.append(" | ".join(m.text_content or "" for m in chat_ctx.messages()))
        return super().chat(chat_ctx=chat_ctx, **kwargs)


def _session(llm: _AnsweringLLM, userdata: Any = None) -> AgentSession:
    return AgentSession(llm=llm, userdata=userdata or Userdata(airline="Northwind"))


async def test_a_close_saves_the_items_and_the_mutable_part(database: Database) -> None:
    llm = _AnsweringLLM(
        fake_responses=[
            _says("hello", "Hi, how can I help?"),
            _says("what is the change fee", "It is $75."),
        ],
        fallbacks=[],
    )
    session = _session(llm)
    await session.start(agent=FareDesk(), persist=database.session("s1"))
    await session.run(user_input="hello")
    await session.run(user_input="what is the change fee")
    # nothing is written while the session runs
    assert await database.rows("SELECT * FROM chat_items") == []
    assert await database.rows("SELECT * FROM agents") == []
    await session.aclose()

    (row,) = await database.rows("SELECT * FROM sessions")
    assert row["current_agent_id"] == "fare_desk"
    assert row["closed_at"] is not None
    assert row["userdata"] == '{"airline": "Northwind", "rebooked": []}'
    history = await database.rows(
        "SELECT item FROM chat_items WHERE owner = 'session' ORDER BY created_at"
    )
    texts = [r["item"] for r in history if '"type": "message"' in r["item"]]
    assert ["hello" in texts[0], "Hi, how" in texts[1], "change fee" in texts[2]] == [True] * 3
    assert "$75" in texts[3]
    (agent,) = await database.rows("SELECT * FROM agents")
    assert agent["cls"] == "tests.test_session_state:FareDesk" and agent["state"] == "{}"
    agent_items = await database.rows("SELECT * FROM chat_items WHERE owner = 'fare_desk'")
    assert len(agent_items) >= 4


async def test_a_second_start_resumes_the_session(database: Database) -> None:
    llm = _RecordingLLM(
        fake_responses=[_says("my flight is NW812", "Noted, NW812.")], fallbacks=["You said NW812."]
    )
    first = _session(llm, Userdata(airline="Northwind", rebooked=["NW100"]))
    await first.start(agent=FareDesk(), persist=database.session("s1"))
    assert not first.resumed
    await first.run(user_input="my flight is NW812")
    history = [item.id for item in first.history.items]
    await first.aclose()

    second = _session(llm, Userdata(airline="a fresh seed"))
    agent = FareDesk()
    await second.start(agent=agent, persist=database.session("s1"))
    assert second.resumed
    assert [item.id for item in second.history.items][: len(history)] == history
    # the stored JSON is loaded into the type of the userdata the handler passed
    assert second.userdata == Userdata(airline="Northwind", rebooked=["NW100"])
    assert any("NW812" in (m.text_content or "") for m in agent.chat_ctx.messages())
    await second.run(user_input="which flight did I give you")
    # the model answering the follow-up sees the turn from before the restart
    assert "my flight is NW812" in llm.seen[-1]
    await second.aclose()


async def test_a_resumed_start_records_no_handoff_and_no_configuration(
    database: Database,
) -> None:
    llm = _AnsweringLLM(fake_responses=[], fallbacks=[])

    def recorded(session: AgentSession) -> list[int]:
        kinds = ("agent_handoff", "agent_config_update")
        return [sum(item.type == kind for item in session.history.items) for kind in kinds]

    first = _session(llm)
    await first.start(agent=FareDesk(), persist=database.session("s1"))
    await first.aclose()
    assert recorded(first) == [1, 1]

    # the session goes on with the agent it left off on, so its start records nothing
    second = _session(llm)
    agent = FareDesk()
    await second.start(agent=agent, persist=database.session("s1"))
    assert recorded(second) == [1, 1]
    assert [item.type for item in agent.chat_ctx.items].count("agent_config_update") == 1

    # a real handoff records both
    second.update_agent(Transferring())
    assert second._update_activity_atask is not None
    await second._update_activity_atask
    assert recorded(second) == [2, 2]
    await second.aclose()

    # a resumed agent whose configuration changed records the change, and still no handoff
    class Changed(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You transfer to billing and sales.", id="transferring")

    third = _session(llm)
    await third.start(agent=Changed(), persist=database.session("s1"))
    assert recorded(third) == [2, 3]
    await third.aclose()

    # the same session and agent started again without persistence is a start like any other
    await (fresh := _session(llm)).start(agent=FareDesk(), persist=database.session("s2"))
    await fresh.aclose()
    fourth = _session(llm)
    agent = FareDesk()
    await fourth.start(agent=agent, persist=database.session("s2"))
    await fourth.aclose()
    await fourth.start(agent=agent)
    assert recorded(fourth)[0] == 2
    await fourth.aclose()


async def test_on_enter_runs_when_an_agent_is_entered_not_when_it_resumes(
    database: Database,
) -> None:
    ENTERED.clear()
    llm = _AnsweringLLM(fake_responses=[], fallbacks=[])
    first = _session(llm)
    await first.start(agent=Greeter(), persist=database.session("s1"))
    await asyncio.sleep(0)
    # a new row starts like any session, and a real handoff enters the next agent
    assert ENTERED == ["greeter"]
    first.update_agent(Rebooker())
    assert first._update_activity_atask is not None
    await first._update_activity_atask
    assert ENTERED == ["greeter", "rebooker"]
    await first.aclose()

    # the rebuilt agent the session left off on resumes, and so does one the handler passed
    for agent in (Greeter(), Rebooker()):
        resumed = _session(llm)
        await resumed.start(agent=agent, persist=database.session("s1"))
        await asyncio.sleep(0)
        assert resumed.current_agent.id == "rebooker"
        await resumed.aclose()
    assert ENTERED == ["greeter", "rebooker"]


async def test_userdata_is_json_only(database: Database) -> None:
    session = AgentSession(llm=_AnsweringLLM(fake_responses=[], fallbacks=[]), userdata=object())
    with pytest.raises(TypeError, match="builtins:object cannot be persisted.*dataclass"):
        await session.start(agent=FareDesk(), persist=database.session("s1"))
    # tuple keys write as JSON strings and do not read back as tuples
    keyed = AgentSession(
        llm=_AnsweringLLM(fake_responses=[], fallbacks=[]),
        userdata=Keyed(seats={("NW812", "12A"): "held"}),
    )
    with pytest.raises(TypeError, match="Keyed cannot be persisted"):
        await keyed.start(agent=FareDesk(), persist=database.session("s1"))

    # plain JSON is stored as is, and comes back as a dict to a handler that passes none
    first = AgentSession(
        llm=_AnsweringLLM(fake_responses=[], fallbacks=[]), userdata={"seats": ["12A"]}
    )
    await first.start(agent=FareDesk(), persist=database.session("s2"))
    await first.aclose()
    second = AgentSession(llm=_AnsweringLLM(fake_responses=[], fallbacks=[]))
    await second.start(agent=FareDesk(), persist=database.session("s2"))
    assert second.userdata == {"seats": ["12A"]}
    await second.aclose()


async def test_a_save_called_twice_writes_only_the_difference(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _AnsweringLLM(fake_responses=[_says("hello", "Hi, how can I help?")], fallbacks=[])
    session = _session(llm)
    await session.start(agent=FareDesk(), persist=database.session("s1"))
    executor = await database.executor()
    written: list[list[str]] = []
    batch = executor.batch

    async def recording(*statements: Any) -> Any:
        written.append([params[2] for sql, params in statements if "INTO chat_items" in sql])
        return await batch(*statements)

    monkeypatch.setattr(executor, "batch", recording)
    await session.save()
    await session.run(user_input="hello")
    await session.save()
    await session.save()

    history = {item.id for item in session.history.items}
    first, second, third = written
    # the second save writes the turn, once for the history and once for the agent's context
    assert set(second).isdisjoint(first) and set(second) <= history
    assert {"hello", "Hi, how can I help?"} <= {
        m.text_content for m in session.history.messages() if m.id in second
    }
    assert third == []
    await session.aclose()


async def test_saves_from_conversation_item_added_and_the_close_write_each_item_once(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    from livekit.agents.a2a import TASK_ID_KEY, TaskUpdate
    from livekit.agents.delegation import DELEGATE_TOOL_NAME

    from .test_delegation import _Scripted, _ScriptedStream

    class _Named(_ScriptedStream):
        task_id = "task-1"

        async def __anext__(self) -> TaskUpdate:
            # the expert acknowledges once the call is recorded and saved
            await asyncio.sleep(0.2)
            return await super().__anext__()

    class _Desk(_Scripted):
        def submit(self, task_input: Any) -> Any:
            return _Named(list(self._updates))

    delegate = _Desk(
        TaskUpdate(state="working", text="looking it up"),
        TaskUpdate(state="completed", text="It is 240 USD."),
    )
    call = _tool_call(DELEGATE_TOOL_NAME, "d1", '{"task": "what is the fare"}')
    llm = _AnsweringLLM(
        fake_responses=[_says("how much is it", "one sec", calls=[call])],
        fallbacks=["Sure, let me check.", "It is 240 USD."],
    )
    session = AgentSession(llm=llm, delegate=delegate)
    await session.start(agent=Agent(instructions="voice"), persist=database.session("s1"))
    executor = await database.executor()
    written: list[tuple[str, str]] = []
    batch = executor.batch

    async def recording(*statements: Any) -> Any:
        written.extend((p[1], p[2]) for sql, p in statements if "INTO chat_items" in sql)
        return await batch(*statements)

    monkeypatch.setattr(executor, "batch", recording)
    saves: list[asyncio.Task[None]] = []
    session.on(
        "conversation_item_added", lambda _: saves.append(asyncio.create_task(session.save()))
    )
    session.generate_reply(user_input="how much is it")
    for _ in range(100):
        if any(
            i.type == "function_call_output" and i.call_id == "d1_final"
            for i in session.history.items
        ):
            break
        await asyncio.sleep(0.05)
    await asyncio.gather(*saves)
    assert saves
    await session.aclose()

    # the framework never edits an item it recorded, so no save writes one a second time
    assert written and len(written) == len(set(written))
    rows = await database.rows(
        "SELECT json_extract(item, '$.call_id') AS call_id, "
        f"json_extract(item, '$.extra.\"{TASK_ID_KEY}\"') AS task_id FROM chat_items "
        "WHERE owner = 'session' AND json_extract(item, '$.type') = 'function_call_output'"
    )
    assert {row["call_id"]: row["task_id"] for row in rows}["d1_final"] == "task-1"


async def test_a_resumed_session_writes_nothing_it_loaded(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _AnsweringLLM(fake_responses=[_says("hello", "Hi, how can I help?")], fallbacks=[])
    first = _session(llm)
    await first.start(agent=FareDesk(), persist=database.session("s1"))
    await first.run(user_input="hello")
    await first.aclose()

    second = _session(llm)
    await second.start(agent=FareDesk(), persist=database.session("s1"))
    executor = await database.executor()
    written: list[str] = []
    batch = executor.batch

    async def recording(*statements: Any) -> Any:
        written.extend(sql.split()[0] for sql, _ in statements if "chat_items" in sql)
        return await batch(*statements)

    monkeypatch.setattr(executor, "batch", recording)
    await second.save()
    await second.aclose()
    assert written == []


async def test_a_save_that_queued_behind_the_close_is_a_no_op(database: Database) -> None:
    llm = _AnsweringLLM(fake_responses=[_says("hello", "Hi, how can I help?")], fallbacks=[])
    session = _session(llm)
    await session.start(agent=FareDesk(), persist=database.session("s1"))
    await session.run(user_input="hello")
    # an application's save, such as one from conversation_item_added during the teardown,
    # takes the lock only once the close saved and let the rows go
    persisted = session.persisted
    assert persisted is not None
    await session.aclose()
    # a save that held the persisted session from before the close
    session._persisted = persisted
    await session.save()
    (row,) = await database.rows("SELECT closed_at FROM sessions")
    assert row["closed_at"] is not None


async def test_a_session_closed_with_an_error_still_saves(database: Database) -> None:
    from livekit.agents.llm import LLMError
    from livekit.agents.voice.agent_session import SessionConnectOptions

    llm = _AnsweringLLM(fake_responses=[_says("hello", "Hi, how can I help?")], fallbacks=[])
    session = AgentSession(
        llm=llm,
        userdata=Userdata(airline="Northwind"),
        conn_options=SessionConnectOptions(max_unrecoverable_errors=1),
    )
    await session.start(agent=FareDesk(), persist=database.session("s1"))
    await session.run(user_input="hello")
    error = LLMError(
        timestamp=0.0, label="test", error=RuntimeError("llm unavailable"), recoverable=False
    )
    session._on_error(error)
    session._on_error(error)
    assert session._closing_task is not None
    await session._closing_task

    (row,) = await database.rows("SELECT current_agent_id, closed_at FROM sessions")
    assert row["current_agent_id"] == "fare_desk" and row["closed_at"] is not None
    items = await database.rows("SELECT item FROM chat_items WHERE owner = 'session'")
    assert any("Hi, how can I help?" in r["item"] for r in items)


async def test_userdata_that_stops_being_json_keeps_the_rest_saved(database: Database) -> None:
    session = _session(_AnsweringLLM(fake_responses=[], fallbacks=[]))
    await session.start(agent=FareDesk(), persist=database.session("s1"))
    await session.save()
    (before,) = await database.rows("SELECT userdata, updated_at FROM sessions")

    # a tool puts something JSON cannot hold in the userdata, and the agent changes
    session.userdata.rebooked.append(object())  # type: ignore[arg-type]
    session.update_agent(Transferring())
    assert session._update_activity_atask is not None
    await session._update_activity_atask
    await session.save()

    # the userdata keeps its last good value, and the agent is still written
    (after,) = await database.rows("SELECT * FROM sessions")
    assert after["current_agent_id"] == "transferring"
    assert after["userdata"] == before["userdata"]
    assert after["updated_at"] > before["updated_at"]
    session.userdata.rebooked.clear()
    await session.aclose()


async def test_a_handoff_resumes_on_the_rebuilt_agent(database: Database) -> None:
    llm = _AnsweringLLM(
        fake_responses=[
            _says(
                "move me to NW812",
                "",
                calls=[_tool_call("rebook", "call_1", '{"flight": "NW812"}')],
            )
        ],
        fallbacks=["Rebooking here.", "Still rebooking."],
    )
    first = _session(llm)
    await first.start(agent=FareDesk(), persist=database.session("s1"))
    await first.run(user_input="move me to NW812")
    assert isinstance(first.current_agent, Rebooking)
    await first.aclose()

    (agent,) = await database.rows("SELECT cls, state FROM agents WHERE agent_id = 'rebooking'")
    assert agent == {"cls": "tests.test_session_state:Rebooking", "state": '{"flight": "NW812"}'}

    second = _session(llm)
    await second.start(agent=FareDesk(), persist=database.session("s1"))
    assert isinstance(second.current_agent, Rebooking)
    assert second.current_agent._flight == "NW812"
    assert second.userdata.rebooked == ["NW812"]
    await second.aclose()


async def test_a_task_lost_with_its_tool_resumes_the_agent_that_awaited_it(
    database: Database, caplog: pytest.LogCaptureFixture
) -> None:
    CONFIRMING.clear()
    llm = _AnsweringLLM(
        fake_responses=[
            _says(
                "move me to NW812", "", calls=[_tool_call("rebook", "c1", '{"flight": "NW812"}')]
            ),
            _says("go ahead", "", calls=[_tool_call("confirm", "c2")]),
        ],
        fallbacks=["Rebooking here.", "Please confirm."],
    )
    first = _session(llm)
    await first.start(agent=FareDesk(), persist=database.session("s1"))
    await first.run(user_input="move me to NW812")
    first.generate_reply(user_input="go ahead")
    await asyncio.wait_for(CONFIRMING.wait(), 5)
    for _ in range(50):
        if isinstance(first.current_agent, Confirming):
            break
        await asyncio.sleep(0.02)
    assert isinstance(first.current_agent, Confirming)
    await first.aclose()
    (row,) = await database.rows("SELECT current_agent_id FROM sessions")
    assert row["current_agent_id"] == "confirming"

    # the task ran in a tool that is not durable, so the nearest agent that rebuilds resumes
    resumed = _session(llm)
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        await resumed.start(agent=FareDesk(), persist=database.session("s1"))
    assert isinstance(resumed.current_agent, Rebooking)
    (lost,) = [r for r in caplog.records if "not durable" in r.getMessage()]
    assert (lost.agent_id, lost.resumed_agent_id) == ("confirming", "rebooking")  # type: ignore[attr-defined]
    await resumed.aclose()


async def test_a_class_that_cannot_be_rebuilt_warns_and_falls_back(
    database: Database, caplog: pytest.LogCaptureFixture
) -> None:
    llm = _AnsweringLLM(
        fake_responses=[_says("billing please", "", calls=[_tool_call("to_billing", "call_1")])],
        fallbacks=["Billing here.", "Back at the desk."],
    )
    first = _session(llm)
    await first.start(agent=Transferring(), persist=database.session("s1"))
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        await first.run(user_input="billing please")
        assert isinstance(first.current_agent, Billing)
        history = [item.id for item in first.history.items]
        # the check runs at the save on close
        await first.aclose()
    assert any(
        "Billing cannot be rebuilt on resume" in r.getMessage()
        and "'customer' has no matching attribute" in r.getMessage()
        for r in caplog.records
    )

    caplog.clear()
    root = Transferring()
    second = _session(llm)
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        await second.start(agent=root, persist=database.session("s1"))
    assert second.current_agent is root
    assert any("could not be rebuilt" in r.getMessage() for r in caplog.records)
    # the root resumes with the whole history in front of it
    root_ids = {item.id for item in root.chat_ctx.items}
    messages = [i.id for i in first.history.items if i.id in history and i.type == "message"]
    assert set(messages) <= root_ids
    await second.aclose()


def test_workflow_tasks_rebuild_with_the_default() -> None:
    task = GetEmailTask(require_confirmation=False)
    state = task._snapshot_state()
    rebuilt = GetEmailTask._from_state(state)
    assert rebuilt._require_confirmation is False
    assert rebuilt.instructions == task.instructions


async def test_an_unchanged_configuration_is_recorded_once() -> None:
    llm = _AnsweringLLM(fake_responses=[], fallbacks=[])
    first = AgentSession(llm=llm)
    await first.start(agent=FareDesk())
    await first.aclose()
    carried = first.history.copy()

    # a restart with the same instructions and tools records nothing new
    agent = FareDesk()
    await agent.update_chat_ctx(carried)
    second = AgentSession(llm=llm)
    await second.start(agent=agent)
    configs = [i for i in agent.chat_ctx.items if i.type == "agent_config_update"]
    assert len(configs) == 1
    # the update names every tool in force, which is what the next start compares against
    assert configs[0].tools is not None and set(configs[0].tools) == set(
        configs[0].tools_added or ()
    )
    await second.aclose()

    # a changed instruction is recorded
    changed = Agent(
        instructions="You answer baggage questions.", chat_ctx=ChatContext(carried.items)
    )
    third = AgentSession(llm=llm)
    await third.start(agent=changed)
    configs = [i for i in changed.chat_ctx.items if i.type == "agent_config_update"]
    assert len(configs) == 2
    await third.aclose()


class _Addressed(Delegate):
    """A delegate with an address and no a2a: all a resumed session needs from it."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def submit(self, task_input: Any) -> Any:
        raise AssertionError("nothing is delegated here")


async def test_a_resumed_session_knows_its_child_behind_its_delegate(
    database: Database, caplog: pytest.LogCaptureFixture
) -> None:
    earlier = database.session("voice")
    await earlier.load()
    for child, endpoint in (("ctx-9", "desk"), ("ctx-2", "baggage")):
        expert = database.session(child, parent="voice", endpoint=endpoint)
        await expert.load()
        await expert.release()
    await earlier.release()

    delegate = _Addressed("desk")
    session = AgentSession(llm=_AnsweringLLM(fake_responses=[], fallbacks=[]), delegate=delegate)
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        await session.start(agent=Agent(instructions="voice"), persist=database.session("voice"))
    assert session.persisted is not None
    assert session.persisted.child_session(delegate.endpoint) == "ctx-9"
    # a child on an endpoint the session has no delegate for is reported, and left alone
    (warning,) = [r for r in caplog.records if "no delegate for" in r.getMessage()]
    assert warning.endpoint == "baggage"  # type: ignore[attr-defined]
    await session.aclose()

    # a new session has nothing to go back to
    session = AgentSession(
        llm=_AnsweringLLM(fake_responses=[], fallbacks=[]), delegate=_Addressed("desk")
    )
    await session.start(agent=Agent(instructions="voice"), persist=database.session("new"))
    assert session.persisted is not None
    assert session.persisted.child_session("desk") is None
    await session.aclose()
