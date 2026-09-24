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

LEASE_TTL = 0.3


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


@pytest.fixture
async def database(tmp_path: pathlib.Path) -> AsyncIterator[Database]:
    local = store.LocalStore(tmp_path, lease_ttl=LEASE_TTL)
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


async def test_turns_are_written_as_they_happen(database: Database) -> None:
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
    await session.aclose()

    (row,) = await database.rows("SELECT * FROM sessions")
    assert row["current_agent_id"] == "fare_desk"
    assert row["lease_owner"] is None and row["closed_at"] is not None
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


async def test_a_second_start_resumes_the_conversation(database: Database) -> None:
    llm = _RecordingLLM(
        fake_responses=[_says("my flight is NW812", "Noted, NW812.")], fallbacks=["You said NW812."]
    )
    first = _session(llm, Userdata(airline="Northwind", rebooked=["NW100"]))
    await first.start(agent=FareDesk(), persist=database.session("s1"))
    await first.run(user_input="my flight is NW812")
    history = [item.id for item in first.history.items]
    await first.aclose()

    second = _session(llm, Userdata(airline="a fresh seed"))
    agent = FareDesk()
    await second.start(agent=agent, persist=database.session("s1"))
    assert [item.id for item in second.history.items][: len(history)] == history
    # the stored JSON is loaded into the type of the userdata the handler passed
    assert second.userdata == Userdata(airline="Northwind", rebooked=["NW100"])
    assert any("NW812" in (m.text_content or "") for m in agent.chat_ctx.messages())
    await second.run(user_input="which flight did I give you")
    # the model answering the follow-up sees the turn from before the restart
    assert "my flight is NW812" in llm.seen[-1]
    await second.aclose()


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


async def test_a_call_running_at_a_crash_leaves_nothing_behind(database: Database) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    @function_tool
    async def rebook(ctx: RunContext, flight: str) -> str:
        """Rebook a flight."""
        started.set()
        await release.wait()
        return "rebooked"

    llm = _AnsweringLLM(
        fake_responses=[
            _says("rebook NW812", "", calls=[_tool_call("rebook", "call_1", '{"flight": "NW812"}')])
        ],
        fallbacks=["done"],
    )
    crashed = _session(llm)
    await crashed.start(
        agent=Agent(instructions="fare desk", tools=[rebook]), persist=database.session("s1")
    )
    crashed.generate_reply(user_input="rebook NW812")
    await asyncio.wait_for(started.wait(), 5)

    # the first worker never closes: the second waits out its lease and takes the session
    resumed = _session(llm)
    agent = Agent(instructions="fare desk", tools=[rebook])
    await resumed.start(agent=agent, persist=database.session("s1"))
    # the call and its output are committed together when it ends, so a crash leaves neither
    for ctx in (resumed.history, agent.chat_ctx):
        assert not any(getattr(i, "call_id", None) == "call_1" for i in ctx.items)
    assert any(m.text_content == "rebook NW812" for m in resumed.history.messages())

    release.set()
    await resumed.aclose()
    await crashed.aclose()


async def test_a_stale_worker_still_lets_the_database_go(database: Database) -> None:
    llm = _AnsweringLLM(fake_responses=[_says("hello", "Hi.")], fallbacks=["Hi again."])
    stale = _session(llm)
    await stale.start(agent=FareDesk(), persist=database.session("s1"))
    await stale.run(user_input="hello")

    # the stale worker idles past its lease, and a second one takes the session
    resumed = _session(llm)
    await resumed.start(agent=FareDesk(), persist=database.session("s1"))
    await resumed.aclose()
    await stale.aclose()
    with pytest.raises(store.StoreError):
        _ = database.store._databases[database.database_id].executor


async def test_a_start_that_cannot_claim_the_session_lets_it_go(database: Database) -> None:
    holder = database.session("s1")
    await holder.load()

    async def renew() -> None:
        while True:
            await holder.checkpoint(current_agent_id=None, userdata=None, agents=[])
            await asyncio.sleep(LEASE_TTL / 3)

    renewing = asyncio.create_task(renew())
    session = _session(_AnsweringLLM(fake_responses=[], fallbacks=[]))
    with pytest.raises(store.LeaseHeldError):
        await session.start(agent=FareDesk(), persist=database.session("s1"))
    assert session.persisted is None
    renewing.cancel()
    await holder.release()
    with pytest.raises(store.StoreError):
        _ = database.store._databases[database.database_id].executor


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
    crashed = _session(llm)
    await crashed.start(agent=FareDesk(), persist=database.session("s1"))
    await crashed.run(user_input="move me to NW812")
    crashed.generate_reply(user_input="go ahead")
    await asyncio.wait_for(CONFIRMING.wait(), 5)
    for _ in range(50):
        if isinstance(crashed.current_agent, Confirming):
            break
        await asyncio.sleep(0.02)
    assert isinstance(crashed.current_agent, Confirming)
    await crashed._persistence.checkpoint()  # type: ignore[union-attr]

    # the task ran in a tool that is not durable, so the nearest agent that rebuilds resumes
    resumed = _session(llm)
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        await resumed.start(agent=FareDesk(), persist=database.session("s1"))
    assert isinstance(resumed.current_agent, Rebooking)
    assert any(
        "nearest agent" in r.getMessage() and "AgentTask ends with the tool call" in str(r.skipped)  # type: ignore[attr-defined]
        for r in caplog.records
    )
    await resumed.aclose()
    await crashed.aclose()


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
    assert any(
        "Billing cannot be rebuilt on resume" in r.getMessage()
        and "'customer' has no matching attribute" in r.getMessage()
        for r in caplog.records
    )
    history = [item.id for item in first.history.items]
    await first.aclose()

    caplog.clear()
    root = Transferring()
    second = _session(llm)
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        await second.start(agent=root, persist=database.session("s1"))
    assert second.current_agent is root
    assert any("could not be rebuilt" in r.getMessage() for r in caplog.records)
    # the root resumes with the whole conversation in front of it
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


class _Remembering(Delegate):
    """A delegate with an address and no a2a: all a resumed session needs from it."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self.resumed: list[str] = []

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def resume(self, context_id: str) -> bool:
        self.resumed.append(context_id)
        return True

    def submit(self, task_input: Any) -> Any:
        raise AssertionError("nothing is delegated here")


async def test_a_resumed_session_points_its_delegate_back_at_its_child(
    database: Database, caplog: pytest.LogCaptureFixture
) -> None:
    earlier = database.session("voice")
    await earlier.load()
    for child, endpoint in (("ctx-9", "desk"), ("ctx-2", "baggage")):
        expert = database.session(child, parent="voice", endpoint=endpoint)
        await expert.load()
        await expert.release()
    await earlier.release()

    delegate = _Remembering("desk")
    session = AgentSession(llm=_AnsweringLLM(fake_responses=[], fallbacks=[]), delegate=delegate)
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        await session.start(agent=Agent(instructions="voice"), persist=database.session("voice"))
    assert delegate.resumed == ["ctx-9"]
    # a child on an endpoint the session has no delegate for is reported, and left alone
    (warning,) = [r for r in caplog.records if "no delegate for" in r.getMessage()]
    assert warning.endpoint == "baggage"  # type: ignore[attr-defined]
    await session.aclose()

    # a new session has nothing to go back to
    fresh = _Remembering("desk")
    session = AgentSession(llm=_AnsweringLLM(fake_responses=[], fallbacks=[]), delegate=fresh)
    await session.start(agent=Agent(instructions="voice"), persist=database.session("new"))
    assert fresh.resumed == []
    await session.aclose()
