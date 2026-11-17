"""Text journeys across two workers sharing one thread database.

Worker processes (``text_worker.py``): real Python workers, HTTP/SSE, bindings and live AgentDB,
with a scripted model and application service. Service calls, caller responses and persisted
history are the oracles.

In-process servers (``resources.serving``): two ``AgentServer``s on loopback listeners, sharing a
local SQLite store or the project's AgentDB, driven by a stock a2a-sdk client.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, aclosing
from dataclasses import dataclass
from typing import Literal

import httpx
import pytest
from a2a.client import Client
from a2a.types import a2a_pb2 as pb
from google.protobuf import json_format

from livekit.agents import Agent, AgentServer, SessionState, TextSession, persisted
from livekit.agents._generated.a2a_metadata import REASON
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.llm import ChatContext
from livekit.agents.storage import AgentDBStorage, Storage, Thread
from tests.fakes import FakeLLM, last_input

from .drivers import ENDPOINT, Workers, answer, first_event, rpc, send, settled_task, stock_client
from .resources import RecordingService, finish_cleanup, serving

COMPLETED = pb.TaskState.TASK_STATE_COMPLETED
INPUT_REQUIRED = pb.TaskState.TASK_STATE_INPUT_REQUIRED


async def said(thread: str) -> list[str | None]:
    """The thread's persisted message texts."""
    history = await Thread(AgentDBStorage(), thread, partition_key="e2e").chat_history()
    return [item.text_content for item in history.items if item.type == "message"]


# -- worker processes on live AgentDB


async def test_a_thread_started_on_one_worker_continues_on_another(
    workers: Workers, thread: str
) -> None:
    """B answers from the history A wrote, and the thread persists both turns once."""
    await workers.a.send(workers.a.message("I am Alice", thread))

    recalled = await workers.b.send(workers.b.message("recall", thread))

    assert (recalled.context_id, recalled.status.state) == (thread, COMPLETED)
    assert answer(recalled) == "I am Alice"
    assert await said(thread) == ["I am Alice", "reply: I am Alice", "recall", "I am Alice"]


async def test_a_resent_message_returns_its_task_without_running_again(
    workers: Workers, thread: str, service: RecordingService
) -> None:
    """The same message resent to another worker returns the first task; nothing re-executes."""
    message = workers.a.message("I am Alice", thread)
    sent = await workers.a.send(message)

    again = await workers.b.send(message)

    assert again.id == sent.id
    assert [(call.worker, call.terms) for call in service.calls] == [("worker-a", "I am Alice")]


async def test_an_answer_to_a_question_asked_on_one_worker_is_delivered_on_another(
    workers: Workers, thread: str
) -> None:
    """A asks for an email; B delivers the reply as the pending tool's output, once."""
    asked = await workers.a.send(workers.a.message("need-email", thread))

    replied = await workers.b.send(workers.b.message("alice@example.com", thread))

    assert (asked.status.state, asked.status.message.parts[0].text) == (
        INPUT_REQUIRED,
        "Which email?",
    )
    assert answer(replied) == "completed: alice@example.com"
    history = await Thread(AgentDBStorage(), thread, partition_key="e2e").chat_history()
    outputs = [item.output for item in history.items if item.type == "function_call_output"]
    assert outputs == ["alice@example.com"]


async def test_yes_on_another_worker_books_the_read_back_terms_exactly_once(
    workers: Workers, thread: str, service: RecordingService
) -> None:
    """A reads the terms back without booking; B's yes books them once, and a resent yes
    does not book again."""
    asked = await workers.a.send(workers.a.message("approve", thread))
    booked_before = [call for call in service.calls if call.operation == "book"]
    yes = workers.b.message("yes", thread)

    result = await workers.b.send(yes)
    await workers.a.send(yes)

    assert asked.status.message.parts[0].text == "Book slot-9 $100? Yes or no?"
    assert booked_before == []
    assert answer(result) == "completed: booked slot-9 $100"
    booked = [(c.worker, c.thread, c.terms) for c in service.calls if c.operation == "book"]
    assert booked == [("worker-b", thread, "slot-9 $100")]


async def test_no_on_another_worker_books_nothing(
    workers: Workers, thread: str, service: RecordingService
) -> None:
    """Declining the read-back on B completes the turn without any booking."""
    await workers.a.send(workers.a.message("approve", thread))

    result = await workers.b.send(workers.b.message("no", thread))

    assert answer(result) == "declined"
    assert [call for call in service.calls if call.operation == "book"] == []


async def test_a_worker_killed_mid_turn_leaves_a_failed_task_and_the_thread_usable(
    workers: Workers, thread: str, service: RecordingService
) -> None:
    """A dies inside a tool; B's next turn answers, the orphan reads as failed (worker-lost),
    and nothing is booked."""
    request = pb.SendMessageRequest(message=workers.a.message("block", thread))
    async with aclosing(workers.a.events("message/stream", request)) as events:
        submitted = await anext(events)
        await service.wait_for_call(thread, "wait")
        workers.a_process.process.kill()
        await finish_cleanup(workers.a_process.stop())

    result = await workers.b.send(workers.b.message("replacement", thread))

    assert answer(result) == "reply: replacement"
    orphan = json_format.ParseDict(
        await workers.b.rpc("tasks/get", pb.GetTaskRequest(id=submitted.task.id), thread=thread),
        pb.Task(),
    )
    assert orphan.status.state == pb.TaskState.TASK_STATE_FAILED
    assert orphan.metadata[REASON] == "worker-lost"
    assert [call for call in service.calls if call.operation == "book"] == []


# -- in-process servers sharing one storage


Handler = Callable[[A2ASessionContext], Awaitable[str]]
TwoServers = Callable[[Handler], Awaitable[tuple[httpx.AsyncClient, httpx.AsyncClient]]]
StockClients = Callable[[httpx.AsyncClient], Awaitable[Client]]


@pytest.fixture
async def two_servers(
    thread_storages: tuple[Storage, Storage], no_project_env: None
) -> AsyncIterator[TwoServers]:
    """``await two_servers(handler)``: HTTP clients of two servers on the shared storage, each
    serving ``handler`` at ``ENDPOINT``."""
    async with AsyncExitStack() as stack:

        async def start(handler: Handler) -> tuple[httpx.AsyncClient, httpx.AsyncClient]:
            clients = []
            for storage in thread_storages:
                server = AgentServer(
                    storage=storage,
                    agent_name="front-desk",
                    api_key="APItest",
                    api_secret="secret-secret-secret-secret-secret",
                )
                server.a2a_session(endpoint=ENDPOINT, description="Answers questions.")(handler)
                base = await stack.enter_async_context(serving(server))
                clients.append(
                    await stack.enter_async_context(httpx.AsyncClient(base_url=base, timeout=30))
                )
            return clients[0], clients[1]

        yield start


@pytest.fixture
async def stock_clients() -> AsyncIterator[StockClients]:
    """``await stock_clients(http)``: an a2a-sdk client, closed at teardown."""
    made: list[Client] = []

    async def make(http: httpx.AsyncClient) -> Client:
        made.append(await stock_client(http))
        return made[-1]

    yield make
    for client in made:
        await client.close()


@dataclass
class Conversation:
    session: SessionState = persisted(default_factory=SessionState)
    visits: int = persisted(default=0)


@pytest.mark.parametrize("persistence", ["bound", "unsaved"])
async def test_text_session_storage_survives_a_request_on_another_server(
    two_servers: TwoServers,
    stock_clients: StockClients,
    thread_storages: tuple[Storage, Storage],
    persistence: Literal["bound", "unsaved"],
) -> None:
    """Real HTTP/SQLite or AgentDB: fresh request contexts restore only explicitly saved or
    bound conversations. Binding alone never saves ordinary application field mutations."""

    def reply(history: ChatContext) -> str:
        if last_input(history) != "recall":
            return "noted"
        return next(
            (
                item.text_content or ""
                for item in history.items
                if item.type == "message" and item.role == "user" and item.text_content != "recall"
            ),
            "nothing saved",
        )

    async def handler(ctx: A2ASessionContext) -> str:
        app = await ctx.store.load_context(Conversation())
        if ctx.message.text_content == "inspect":
            messages = (
                [
                    item.text_content
                    for item in app.session.chat_history.items
                    if item.type == "message"
                ]
                if app.session.current_agent is not None
                else []
            )
            return json.dumps({"messages": messages, "visits": app.visits})
        # Nothing conversational survives in Python objects between requests or servers.
        agent = Agent(name="desk", instructions="Remember the conversation.", llm=FakeLLM(reply))
        app.visits += 1
        turn = await TextSession(agent=agent).run(
            ctx.message, context=ctx.store.bind(app) if persistence == "bound" else app
        )
        assert turn.status == "completed"
        return turn.output

    async def request(
        http: httpx.AsyncClient, client: Client, text: str, context_id: str = ""
    ) -> pb.Task:
        opened = await send(client, text, context_id=context_id)
        task = await settled_task(http, opened.id, opened.context_id)
        assert task.status.state == COMPLETED, task
        return task

    http_a, http_b = await two_servers(handler)
    a, b = await stock_clients(http_a), await stock_clients(http_b)
    opened = await send(a, "I am Alice")
    try:
        first = await settled_task(http_a, opened.id, opened.context_id)
        assert first.status.state == COMPLETED, first
        assert answer(first) == "noted"
        saved = persistence != "unsaved"
        visits = 0
        before = json.loads(answer(await request(http_b, b, "inspect", first.context_id)))
        assert before == {"messages": ["I am Alice", "noted"] if saved else [], "visits": visits}

        recalled = await request(http_b, b, "recall", first.context_id)
        assert answer(recalled) == ("I am Alice" if saved else "nothing saved")
        after = json.loads(answer(await request(http_a, a, "inspect", first.context_id)))
        assert after == {
            "messages": ["I am Alice", "noted", "recall", "I am Alice"] if saved else [],
            "visits": 0,
        }
    finally:
        await finish_cleanup(thread_storages[0].threads.delete_thread(opened.context_id))


async def test_a_thread_continues_on_another_server_with_its_history(
    two_servers: TwoServers, stock_clients: StockClients
) -> None:
    """A message on B with the thread A created sees A's turn in its history."""
    histories: list[list[str]] = []

    async def handler(ctx: A2ASessionContext) -> str:
        items = ctx.message_history.items
        histories.append([item.text_content or "" for item in items if item.type == "message"])
        return f"seen {len(ctx.message_history.items)}"

    http_a, http_b = await two_servers(handler)
    first = await send(await stock_clients(http_a), "hello")

    second = await send(await stock_clients(http_b), "again", context_id=first.context_id)

    assert first.context_id.startswith("AT_")
    assert second.context_id == first.context_id
    assert histories == [[], ["hello", "seen 0"]]


async def test_another_server_lists_every_task_of_the_thread(
    two_servers: TwoServers, stock_clients: StockClients
) -> None:
    """``tasks/list`` on B returns the thread's tasks from both servers, newest first."""

    async def handler(ctx: A2ASessionContext) -> str:
        return "done"

    http_a, http_b = await two_servers(handler)
    client_b = await stock_clients(http_b)
    first = await send(await stock_clients(http_a), "hello")
    second = await send(client_b, "again", context_id=first.context_id)

    listed = await client_b.list_tasks(pb.ListTasksRequest(context_id=first.context_id))

    assert [task.id for task in listed.tasks] == [second.id, first.id]
    assert [task.status.state for task in listed.tasks] == [COMPLETED, COMPLETED]


async def test_a_dropped_stream_leaves_the_task_to_get_task_not_resubscribe(
    two_servers: TwoServers,
    thread_storages: tuple[Storage, Storage],
) -> None:
    """After a caller drops its stream, B refuses resubscribe and GetTask returns the saved
    answer: events are live only."""
    release = asyncio.Event()

    async def handler(ctx: A2ASessionContext) -> str:
        await ctx.update("working on it")
        await release.wait()
        return "done"

    http_a, http_b = await two_servers(handler)
    ctx_id = (await thread_storages[0].threads.create_thread()).thread_id
    task_id, stream = await first_event(http_a, ctx_id)
    await stream.aclose()

    resubscribe = await rpc(http_b, "tasks/resubscribe", {"id": task_id, "contextId": ctx_id})
    release.set()
    task = await settled_task(http_b, task_id, ctx_id)

    assert resubscribe.json()["error"]["code"] == -32004  # UnsupportedOperation
    assert task.status.state == COMPLETED
    assert answer(task) == "done"
