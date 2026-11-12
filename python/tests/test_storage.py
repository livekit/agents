"""The storage package (plan 5): the Rust turn with the door's host, the lease across workers,
resolution, the voice partition_key; against the local agent-db of ``storage_fixture.py``, every test
skipped when the data plane is not up. The run itself (a raising run, a committed run id, a
stale commit) is tested in the crate: ``rust/crates/lk-agents-storage/tests``."""

from __future__ import annotations

import asyncio
import itertools
import json
import sys
from asyncio.subprocess import PIPE
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from livekit.agents import (
    Agent,
    RunContext,
    RunResult,
    SessionState,
    _core,
    persisted,
)
from livekit.agents._generated.a2a_metadata import (
    CHANNEL,
    REASON,
)
from livekit.agents._generated.storage_class import PartitionRow
from livekit.agents._generated.storage_options import TurnOptions
from livekit.agents._generated.storage_turn import TurnResult
from livekit.agents.llm import ChatItem, FunctionToolCall
from livekit.agents.llm.utils import build_legacy_openai_schema
from livekit.agents.storage import (
    ContextStore,
    SQLiteStorage,
    Storage,
    Thread,
    TooLarge,
)
from livekit.agents.storage.session import A2ASessionContext, _Handler, _TurnHost

from .fake_llm import FakeLLM, FakeLLMResponse
from .storage_fixture import (
    LOCAL_LEASE_TTL,
    let_the_lease_lapse,
    new_session,
    project_storage,
)

pytestmark = [pytest.mark.unit]

PARTITION_KEY = "front-desk"


@dataclass
class Patient:
    verified: int = 0


@dataclass
class AppContext:
    patient: Patient = persisted(default_factory=Patient)


@dataclass
class BoundTextContext:
    session: SessionState = persisted(default_factory=SessionState)
    count: int = persisted(default=0)


def a2a(text: str, message_id: str, *, channel: str | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "messageId": message_id,
        "role": "ROLE_USER",
        "parts": [{"text": text}],
    }
    if channel:
        msg["metadata"] = {CHANNEL: channel}
    return msg


async def echo(ctx: A2ASessionContext) -> str:
    return "reply to: " + ctx.message.text_content


Workers = tuple[Storage, Storage]


async def run_turn(
    storage: Storage,
    thread_id: str,
    partition_key: str,
    message: dict[str, Any],
    policy: str | None,
    handler: _Handler,
    *,
    debounce: timedelta | None = None,
    task_id: str | None = None,
) -> TurnResult:
    """One message through the Rust turn with the door's host, as the A2A service runs it."""
    options = TurnOptions(inbox=policy, debounce=debounce)
    host = _TurnHost(
        handler,
        thread=Thread(storage, thread_id, partition_key=partition_key),
    )
    core = await storage._open(thread_id)
    try:
        return await _core.Turn(core, partition_key, host, options).run(message, task_id)
    finally:
        core.close()


async def history_items(storage: Storage, thread_id: str, partition_key: str) -> list[ChatItem]:
    return (await Thread(storage, thread_id, partition_key=partition_key).chat_history()).items


async def partition_row(
    storage: Storage, thread_id: str, partition_key: str
) -> PartitionRow | None:
    core = await storage._open(thread_id)
    try:
        return await core.partition_info(partition_key)
    finally:
        core.close()


async def run_outcomes(storage: Storage, thread_id: str) -> list[str]:
    """What ``run_ended`` recorded, oldest first."""
    core = await storage._open(thread_id)
    try:
        events = await core.journal()
    finally:
        core.close()
    return [str(e.payload["outcome"]) for e in events if e.type == "run_ended"]


# -- the thin slice --


@dataclass
class Notes:
    text: str = ""


@dataclass
class NotesContext:
    notes: Notes = persisted(default_factory=Notes)


async def test_a_save_over_64_kib_raises_too_large_and_commits_nothing(
    thread_storage: Storage,
) -> None:
    """docs/internals/storage.md "What is saved": state is under 64 KiB; a larger save raises
    ``TooLarge`` and the turn's saved fields stay as they were."""
    sid = await new_session(thread_storage, PARTITION_KEY)
    raised: list[BaseException] = []

    async def handler(ctx: A2ASessionContext) -> str:
        app = NotesContext()
        await ctx.store.load_context(app)
        app.notes.text = "small"
        await ctx.store.save_context(app)
        app.notes.text = "x" * (64 * 1024)
        try:
            await ctx.store.save_context(app)
        except TooLarge as e:
            raised.append(e)
        return "ok"

    async def reader(ctx: A2ASessionContext) -> str:
        app = NotesContext()
        await ctx.store.load_context(app)
        return app.notes.text

    await run_turn(thread_storage, sid, PARTITION_KEY, a2a("save", "m1"), "queue", handler)
    read = await run_turn(thread_storage, sid, PARTITION_KEY, a2a("read", "m2"), "queue", reader)
    assert len(raised) == 1
    assert read.output == "small"


async def test_a_raising_handler_fails_the_task_and_keeps_the_message(
    thread_storage: Storage,
) -> None:
    storage = thread_storage
    sid = await new_session(storage, PARTITION_KEY)

    async def boom(ctx: A2ASessionContext) -> str:
        raise RuntimeError("boom")

    result = await run_turn(storage, sid, PARTITION_KEY, a2a("hello", "m1"), "queue", boom)
    assert result.task.status == "failed"
    assert result.task.data.status_message["parts"][0]["text"] == "RuntimeError: boom"
    assert [i.text_content for i in await history_items(storage, sid, PARTITION_KEY)] == ["hello"]
    assert await run_outcomes(storage, sid) == ["failed"]


# -- the journal --


async def test_the_journal_records_a_tool_call_and_its_progress(thread_storages: Workers) -> None:
    a, _ = thread_storages
    sid = await new_session(a, PARTITION_KEY)
    front = Agent(
        instructions="front desk",
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="first",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[FunctionToolCall(name="lookup", arguments="{}", call_id="c1")],
                ),
                FakeLLMResponse(input="found", content="done", ttft=0, duration=0),
            ]
        ),
    )

    @front.tool
    async def lookup(ctx: RunContext) -> str:
        """Look the caller up."""
        await ctx.update("looking")
        return "found"

    async def handler(ctx: A2ASessionContext) -> RunResult[str]:
        return await front.run(ctx.message, message_history=ctx.message_history)

    r = await run_turn(a, sid, PARTITION_KEY, a2a("first", "m1"), "queue", handler)
    assert r.task.status == "completed"

    journal = await Thread(a, sid, partition_key=PARTITION_KEY).journal()
    assert journal[0].type == "run_started" and journal[-1].type == "run_ended"
    tools = [e.payload for e in journal if e.type == "tool_execution_updated"]
    assert [p["phase"] for p in tools] == ["started", "ended", "update"]
    assert tools[0]["name"] == "lookup" and tools[0]["callId"] == "c1"
    assert tools[1]["isError"] is False and tools[2]["text"] == "looking"
    assert "conversation_item_added" in {e.type for e in journal}


# -- interrupt --


# -- the lease --


async def _dying_worker(kind: str, location: str, thread_id: str) -> None:
    """Worker ``wa`` in its own process: it takes the turn, says so, and hangs until killed."""
    if kind == "sqlite":
        ttl = timedelta(seconds=LOCAL_LEASE_TTL)
        storage: Storage = SQLiteStorage(
            directory=location, lease_ttl=ttl, renew_interval=ttl / 4, holder="wa"
        )
    else:
        storage = project_storage("wa")

    async def stuck(ctx: A2ASessionContext) -> str:
        print("entered", flush=True)
        await asyncio.Event().wait()
        return "late"

    await run_turn(
        storage, thread_id, PARTITION_KEY, a2a("hello", "m1"), "queue", stuck, task_id="AST_x"
    )


async def test_a_resend_after_the_holder_died_fails_worker_lost(
    thread_storage_worker: Callable[[str], Storage],
) -> None:
    """Under ``queue``: the holder's process dies mid-turn (no cleanup runs, its lease
    lapses); a resend of the same messageId on another worker fails the orphaned ``working`` row
    ``worker-lost``, nothing runs again, and the message sent under a new messageId runs as a new
    task. A turn cancelled in-process instead cancels its task."""
    a, b = (thread_storage_worker(h) for h in ("wa", "wb"))
    sid = await new_session(a, PARTITION_KEY)
    kind = "sqlite" if isinstance(a, SQLiteStorage) else "agentdb"
    root = Path(__file__).resolve().parents[1]
    script = "import asyncio, sys; from tests.test_storage import _dying_worker; "
    script += "asyncio.run(_dying_worker(*sys.argv[1:]))"
    location = a._core.location if isinstance(a, SQLiteStorage) else ""
    dying = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, kind, location, sid, cwd=root, stdout=PIPE
    )
    assert dying.stdout is not None
    try:
        assert await asyncio.wait_for(dying.stdout.readline(), 60) == b"entered\n"
    finally:
        dying.kill()
        await dying.wait()
    await let_the_lease_lapse(a)  # the lease lapses with no renewal

    lost = await run_turn(b, sid, PARTITION_KEY, a2a("hello", "m1"), "queue", echo)
    assert lost.task.task_id == "AST_x" and lost.task.status == "failed"
    assert lost.task.data.status_message["metadata"][REASON] == "worker-lost"
    assert lost.task.data.history == [a2a("hello", "m1")]
    assert await history_items(b, sid, PARTITION_KEY) == []

    again = await run_turn(b, sid, PARTITION_KEY, a2a("hello again", "m2"), "queue", echo)
    assert again.task.task_id != "AST_x" and again.task.status == "completed"
    assert again.output == "reply to: hello again"
    items = await history_items(b, sid, PARTITION_KEY)
    assert [i.text_content for i in items] == ["hello again", "reply to: hello again"]


# -- input required --


async def test_needs_approval_asks_then_runs_on_the_approved_call(
    thread_storage: Storage,
) -> None:
    """Plan 7.4: ``needs_approval=True`` returns the ``InputRequired`` for you. The first call
    asks yes or no and does not run; the answer is its output; the model's repeat of the same
    call after the answer runs the body. The schema has no approval flag for the model."""
    storage = thread_storage
    sid = await new_session(storage, PARTITION_KEY)
    desk = Agent(
        instructions="front desk",
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="cancel monday",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(
                            name="cancel_appointment", arguments='{"date": "monday"}', call_id="c1"
                        )
                    ],
                ),
                FakeLLMResponse(
                    input="yes",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(
                            name="cancel_appointment", arguments='{"date": "monday"}', call_id="c2"
                        )
                    ],
                ),
                FakeLLMResponse(input="canceled monday", content="done", ttft=0, duration=0),
            ]
        ),
    )
    ran: list[str] = []

    @desk.tool(needs_approval=True)
    async def cancel_appointment(ctx: RunContext, date: str) -> str:
        """Cancel the appointment on a date."""
        ran.append(date)
        return f"canceled {date}"

    async def handler(ctx: A2ASessionContext) -> RunResult[str]:
        return await desk.run(ctx.message, message_history=ctx.message_history)

    assert "approved" not in json.dumps(build_legacy_openai_schema(cancel_appointment))
    first = await run_turn(
        storage, sid, PARTITION_KEY, a2a("cancel monday", "m1"), "queue", handler
    )
    assert first.task.status == "input-required" and ran == []
    assert first.task.data.status_message["parts"][0]["text"] == (
        "Please confirm: Cancel the appointment on a date (date='monday'). Yes or no?"
    )
    second = await run_turn(storage, sid, PARTITION_KEY, a2a("yes", "m2"), "queue", handler)
    assert second.output == "done" and ran == ["monday"]
    items = await history_items(storage, sid, PARTITION_KEY)
    outputs = {i.call_id: i.output for i in items if i.type == "function_call_output"}
    assert outputs == {"c1": "yes", "c2": "canceled monday"}


# -- session resolution and the Session handle --


# -- a text during a call (plan 3.3.7) --


def dispatched_job(thread_id: str = "") -> Any:
    from livekit.protocol import agent

    job = agent.Job(id="AJ_1", agent_name="desk").SerializeToString()
    # `thread_id` (field 15) as wire bytes until the Python protocol package declares it.
    raw = thread_id.encode()
    return agent.Job.FromString(job + bytes([15 << 3 | 2, len(raw)]) + raw if raw else job)


async def test_a_call_opens_only_its_jobs_thread(thread_storage: Storage) -> None:
    import pickle

    from livekit.agents.storage import _call_thread

    sid = await new_session(thread_storage, "desk")
    before = (await thread_storage.threads.list_threads("")).threads
    named = await _call_thread(thread_storage, dispatched_job(sid))
    assert named is not None and named.id == sid and named._partition_key == "default"
    # no thread_id runs the call without one, and an unknown id is refused: nothing is created
    assert await _call_thread(thread_storage, dispatched_job()) is None
    assert await _call_thread(thread_storage, dispatched_job("AT_unknown")) is None
    assert (await thread_storage.threads.list_threads("")).threads == before
    # a job process rebuilds the storage from its configuration
    assert pickle.loads(pickle.dumps(thread_storage))._args == thread_storage._args


async def test_live_session_reloads_when_a_text_moved_the_lane(thread_storages: Workers) -> None:
    from livekit.agents import VoiceSession

    call_worker, text_worker = thread_storages
    sid = await new_session(call_worker, "desk")
    agent = Agent(instructions="desk", name="desk")
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(input="hello", content="hi there", ttft=0, duration=0),
            FakeLLMResponse(
                input="still there?", content="yes, and I read your text", ttft=0, duration=0
            ),
        ]
    )
    state = Patient()
    thread = Thread(call_worker, sid, partition_key="desk")
    app = AppContext(patient=state)
    activation = await call_worker.threads._core_context_store(sid, "desk")
    store = ContextStore(activation=activation, thread=thread, scope_id="root:desk")
    await store.load_context(app)
    async with VoiceSession(llm=llm, context=store.bind(app)) as session:
        await session.start(agent)
        assert session.thread is thread
        await session._run(user_input="hello")
        await asyncio.sleep(0.05)  # the turn's commit is chained behind the speech
        items = await history_items(text_worker, sid, "desk")
        assert [i.text_content for i in items if i.type == "message"] == ["hello", "hi there"]
        row = await partition_row(text_worker, sid, "desk")
        assert row is not None and row.lease_expires_at_ms == 0, "released without draining"

        # a text lands between two voice turns and is answered by another worker
        async def by_text(ctx: A2ASessionContext) -> str:
            text_app = AppContext()
            await ctx.store.load_context(text_app)
            text_app.patient.verified = 5
            await ctx.store.save_context(text_app)
            return "texted back"

        await run_turn(
            text_worker,
            sid,
            "desk",
            a2a("a text", "m1", channel="sms"),
            None,
            by_text,
            debounce=timedelta(0),
        )
        # the next voice turn's acquire returns a newer version: the call reloads once, before
        # the reply is generated, so the turn is over the texted history by the time it returns
        result = await session._run(user_input="still there?")
        assert result.output == "yes, and I read your text"
        texts = [i.text_content for i in session.history.items if i.type == "message"]
        assert texts[:4] == ["hello", "hi there", "a text", "texted back"]
        await session.store.load_context(app)
        assert app.patient.verified == 5 and session.context is app
        seen = [
            i.text_content
            for i in session._agent_chat_ctx().items
            if i.type == "message" and i.role != "system"
        ]
        assert seen[:4] == texts[:4], "the agent's working history reads the text too"

    final = await history_items(text_worker, sid, "desk")
    assert [i.text_content for i in final if i.type == "message"][-2:] == [
        "still there?",
        "yes, and I read your text",
    ]


# -- the database behind a session --


async def test_create_thread_creates_the_database(thread_storage: Storage) -> None:
    """``CreateThread`` creates the thread's database; every worker reads it after a write."""
    storage = thread_storage
    created = await storage.threads.create_thread()
    thread_id = created.thread_id
    assert created.thread_id.startswith("AT_")

    assert await history_items(storage, thread_id, "front") == []
    first = await run_turn(storage, thread_id, "front", a2a("hi", "m1"), "queue", echo)
    assert first.output == "reply to: hi"
    second = await run_turn(storage, thread_id, "front", a2a("again", "m2"), "queue", echo)
    assert second.output == "reply to: again"
    core = await storage._open(thread_id)
    items = await core.history("front")
    assert [i.payload.text_content for i in items] == [
        "hi",
        "reply to: hi",
        "again",
        "reply to: again",
    ]
    tasks = await core.tasks("front")
    assert len(tasks) == 2 and all(t.status == "completed" for t in tasks)
    core.close()


async def test_handler_transactions_join_the_turn_or_commit_on_exit(
    thread_storage: Storage,
) -> None:
    """docs/context.md#application-records-and-transactions: ``transaction()`` commits or rolls
    back with the turn, like a nested savepoint; ``independent_transaction()`` commits when its
    block exits, whatever the turn does next."""
    storage = thread_storage
    thread_id = (await storage.threads.create_thread()).thread_id

    async def handler(ctx: A2ASessionContext) -> str:
        async with ctx.store.independent_transaction() as tx:
            await tx.execute("CREATE TABLE IF NOT EXISTS notes (body TEXT NOT NULL)")
            await tx.execute("INSERT INTO notes (body) VALUES (?)", ("prepared",))
        async with ctx.store.transaction() as tx:
            await tx.execute("INSERT INTO notes (body) VALUES (?)", ("with the turn",))
        with pytest.raises(RuntimeError):
            async with ctx.store.transaction() as tx:
                await tx.execute("INSERT INTO notes (body) VALUES ('rolled back')")
                raise RuntimeError("the block failed")
        async with ctx.store.independent_transaction() as tx:
            assert await tx.fetch_one("SELECT body FROM notes WHERE body = 'none'") is None
            rows = await tx.fetch("SELECT body FROM notes")
            staged_unseen = rows.count(["with the turn"]) == rows.count(["prepared"]) - 1
            assert staged_unseen, "the turn's own write is staged until it commits"
        if ctx.message.text_content == "fail":
            raise RuntimeError("the turn failed")
        return "ok"

    probes = itertools.count()

    async def bodies() -> list[str]:
        async def read(ctx: A2ASessionContext) -> str:
            async with ctx.store.independent_transaction() as tx:
                return ",".join(str(row[0]) for row in await tx.fetch("SELECT body FROM notes"))

        # each probe is a new message: a repeated id would replay the first probe's answer
        result = await run_turn(
            storage, thread_id, "probe", a2a("?", f"p{next(probes)}"), "queue", read
        )
        return str(result.output).split(",")

    await run_turn(storage, thread_id, PARTITION_KEY, a2a("go", "m1"), "queue", handler)
    assert await bodies() == ["prepared", "with the turn"]
    await run_turn(storage, thread_id, PARTITION_KEY, a2a("fail", "m2"), "queue", handler)
    assert await bodies() == ["prepared", "with the turn", "prepared"], "only the independent one"


async def test_the_thread_handle_pages_items_and_tasks(thread_storage: Storage) -> None:
    """``ctx.thread`` pages committed items and task records without a lease
    (docs/internals/storage.md#reading-thread-items-and-tasks)."""
    storage = thread_storage
    thread_id = (await storage.threads.create_thread()).thread_id
    for partition_key, text, message_id in [
        ("front", "a", "m1"),
        ("back", "b", "m2"),
        ("front", "c", "m3"),
    ]:
        await run_turn(storage, thread_id, partition_key, a2a(text, message_id), "queue", echo)
    thread = Thread(storage, thread_id, partition_key="front")

    texts: list[str] = []
    after: str | None = None
    while True:
        items = await thread.list_items(partition_key="front", after=after, limit=3)
        texts += [i.item.text_content or "" for i in items.items]
        if not items.after:
            break
        after = items.after
    assert texts == ["a", "reply to: a", "c", "reply to: c"]

    first = await thread.list_tasks(partition_key="front", limit=1)
    second = await thread.list_tasks(
        partition_key="front", page_token=first.next_page_token, limit=1
    )
    assert first.total_size == 2 and not second.next_page_token
    ids = [first.tasks[0].task_id, second.tasks[0].task_id]
    assert [t.message_id for t in first.tasks + second.tasks] == ["m1", "m3"], "oldest first"
    assert (await thread.task(ids[1])) == second.tasks[0]
    assert await thread.task("AST_missing") is None


async def test_bound_text_checkpoints_and_restores_without_saving_the_session_field(
    thread_storage: Storage,
) -> None:
    from livekit.agents import TextSession

    from .test_text_session import Reply, Script, call

    thread_id = await new_session(thread_storage, PARTITION_KEY)
    thread = Thread(thread_storage, thread_id, partition_key=PARTITION_KEY)
    model = Script(call("inspect", {}, "inspect"), Reply("done"), Reply("again"))
    agent = Agent(name="backend", instructions="Help", llm=model)
    snapshots: list[SessionState] = []

    @agent.tool
    async def inspect(ctx: RunContext[BoundTextContext]) -> str:
        """Read the durable conversation checkpoint before final turn settlement."""
        assert ctx.context.session.active_agent == "backend"
        snapshot = await ctx.store.load_session()
        snapshots.append(snapshot)
        assert snapshot is not ctx.context.session
        assert snapshot.active_agent == "backend"
        assert any(item.type == "function_call" for item in (await ctx.thread.chat_history()).items)
        ctx.context.count = 1
        await ctx.store.save_context(ctx.context)
        await ctx.store.flush()
        return "seen"

    async def handler(ctx: A2ASessionContext) -> object:
        app = await ctx.store.load_context(BoundTextContext())
        original = app.session
        turn = await TextSession(agent=agent).run(ctx.message, context=ctx.store.bind(app))
        assert app.session is original
        return turn.result

    core = await thread_storage._open(thread_id)
    try:
        turn = _core.Turn(
            core, PARTITION_KEY, _TurnHost(handler, thread=thread), TurnOptions(inbox="queue")
        )
        first = await turn.run(a2a("start", "bound-1"))
        second = await turn.run(a2a("continue", "bound-2"))
        assert first.task.status == second.task.status == "completed"
        assert first.output == "done" and second.output == "again"
        assert len(model.requests[-1].items) > len(model.requests[0].items)
        assert snapshots[0].active_agent == "backend"
    finally:
        core.close()
