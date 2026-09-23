"""AgentSession.start(state=...): what a session writes as it runs, and what it gets back."""

from __future__ import annotations

import asyncio
import logging
import pathlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool, store
from livekit.agents.beta.workflows import GetEmailTask
from livekit.agents.store.session_state import INTERRUPTED_OUTPUT

from .test_a2a_runner import _AnsweringLLM, _says, _tool_call

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

LEASE_TTL = 0.3


@dataclass
class Userdata:
    airline: str
    rebooked: list[str] = field(default_factory=list)


class FareDesk(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You answer fare questions.")

    @function_tool
    async def rebook(self, ctx: RunContext[Userdata], flight: str) -> Agent:
        """Hand the caller to rebooking for a flight."""
        ctx.userdata.rebooked.append(flight)
        return Rebooking(flight=flight)


class Rebooking(Agent):
    def __init__(self, *, flight: str) -> None:
        super().__init__(instructions=f"You rebook flight {flight}.")
        self._flight = flight


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
async def conversation(tmp_path: pathlib.Path) -> AsyncIterator[store.Conversation]:
    sqlite = store.SQLite(tmp_path, lease_ttl=LEASE_TTL)
    yield await sqlite.create_conversation()
    await sqlite.aclose()


async def _rows(conversation: store.Conversation, sql: str) -> list[dict]:
    return [row async for row in conversation.executor.query(sql)]


class _RecordingLLM(_AnsweringLLM):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.seen: list[str] = []

    def chat(self, *, chat_ctx: Any, **kwargs: Any) -> Any:
        self.seen.append(" | ".join(m.text_content or "" for m in chat_ctx.messages()))
        return super().chat(chat_ctx=chat_ctx, **kwargs)


def _session(llm: _AnsweringLLM, userdata: Userdata | None = None) -> AgentSession:
    return AgentSession(llm=llm, userdata=userdata or Userdata(airline="Northwind"))


async def test_turns_are_written_as_they_happen(conversation: store.Conversation) -> None:
    llm = _AnsweringLLM(
        fake_responses=[
            _says("hello", "Hi, how can I help?"),
            _says("what is the change fee", "It is $75."),
        ],
        fallbacks=[],
    )
    session = _session(llm)
    await session.start(agent=FareDesk(), state=conversation.session("s1", kind="text"))
    await session.run(user_input="hello")
    await session.run(user_input="what is the change fee")
    await session.aclose()

    (row,) = await _rows(conversation, "SELECT * FROM sessions")
    assert row["current_agent_id"] == "fare_desk"
    assert row["kind"] == "text" and row["lease_owner"] is None and row["closed_at"] is not None
    assert row["userdata"] == '{"airline": "Northwind", "rebooked": []}'
    history = await _rows(
        conversation,
        "SELECT item_json FROM chat_items WHERE owner = 'session' ORDER BY created_at",
    )
    texts = [r["item_json"] for r in history if '"type": "message"' in r["item_json"]]
    assert [("hello" in texts[0]), ("Hi, how" in texts[1]), ("change fee" in texts[2])] == [
        True,
        True,
        True,
    ]
    assert "$75" in texts[3]
    (agent,) = await _rows(conversation, "SELECT * FROM agents")
    assert agent["cls"] == "tests.test_session_state:FareDesk" and agent["state_json"] == "{}"
    agent_items = await _rows(conversation, "SELECT * FROM chat_items WHERE owner = 'fare_desk'")
    assert len(agent_items) >= 4


async def test_a_second_start_resumes_the_conversation(conversation: store.Conversation) -> None:
    llm = _RecordingLLM(
        fake_responses=[_says("my flight is NW812", "Noted, NW812.")], fallbacks=["You said NW812."]
    )
    first = _session(llm, Userdata(airline="Northwind", rebooked=["NW100"]))
    await first.start(agent=FareDesk(), state=conversation.session("s1"))
    await first.run(user_input="my flight is NW812")
    history = [item.id for item in first.history.items]
    await first.aclose()

    second = _session(llm, Userdata(airline="a fresh seed"))
    agent = FareDesk()
    await second.start(agent=agent, state=conversation.session("s1"))
    assert [item.id for item in second.history.items][: len(history)] == history
    assert second.userdata == Userdata(airline="Northwind", rebooked=["NW100"])
    assert any("NW812" in (m.text_content or "") for m in agent.chat_ctx.messages())
    await second.run(user_input="which flight did I give you")
    # the model answering the follow-up sees the turn from before the restart
    assert "my flight is NW812" in llm.seen[-1]
    await second.aclose()


async def test_a_call_running_at_a_crash_is_interrupted(
    conversation: store.Conversation,
) -> None:
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
        agent=Agent(instructions="fare desk", tools=[rebook]), state=conversation.session("s1")
    )
    crashed.generate_reply(user_input="rebook NW812")
    await asyncio.wait_for(started.wait(), 5)

    # the first worker never closes: the second waits out its lease and takes the session
    resumed = _session(llm)
    agent = Agent(instructions="fare desk", tools=[rebook])
    await resumed.start(agent=agent, state=conversation.session("s1"))
    outputs = [i for i in resumed.history.items if i.type == "function_call_output"]
    assert [(o.call_id, o.output, o.is_error) for o in outputs] == [
        ("call_1", INTERRUPTED_OUTPUT, True)
    ]
    assert any(i.type == "function_call" and i.call_id == "call_1" for i in agent.chat_ctx.items)
    (task,) = await _rows(conversation, "SELECT call_id, status FROM tasks")
    assert task == {"call_id": "call_1", "status": "interrupted"}

    release.set()
    await resumed.aclose()
    await crashed.aclose()
    # the stale worker's close is fenced: its outcome is not written over the new owner's
    (task,) = await _rows(conversation, "SELECT status FROM tasks")
    assert task["status"] in ("interrupted", "done")


async def test_a_handoff_resumes_on_the_rebuilt_agent(conversation: store.Conversation) -> None:
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
    await first.start(agent=FareDesk(), state=conversation.session("s1"))
    await first.run(user_input="move me to NW812")
    assert isinstance(first.current_agent, Rebooking)
    await first.aclose()

    (agent,) = await _rows(
        conversation, "SELECT cls, state_json FROM agents WHERE agent_id = 'rebooking'"
    )
    assert agent == {
        "cls": "tests.test_session_state:Rebooking",
        "state_json": '{"flight": "NW812"}',
    }

    second = _session(llm)
    await second.start(agent=FareDesk(), state=conversation.session("s1"))
    assert isinstance(second.current_agent, Rebooking)
    assert second.current_agent._flight == "NW812"
    assert second.userdata.rebooked == ["NW812"]
    await second.aclose()


async def test_a_class_that_cannot_be_rebuilt_warns_and_falls_back(
    conversation: store.Conversation, caplog: pytest.LogCaptureFixture
) -> None:
    llm = _AnsweringLLM(
        fake_responses=[
            _says("billing please", "", calls=[_tool_call("to_billing", "call_1")]),
        ],
        fallbacks=["Billing here.", "Back at the desk."],
    )
    first = _session(llm)
    await first.start(agent=Transferring(), state=conversation.session("s1"))
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
        await second.start(agent=root, state=conversation.session("s1"))
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
