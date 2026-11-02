"""Real Python workers, HTTP/SSE, bindings and live AgentDB; scripted LLM and service.

Each action must settle within REQUEST_SECONDS. Worker startup has its own bound.
No model/provider network is used. Service calls, caller responses and persisted history
are the oracles. An error, timeout or missing prerequisite fails the selected run.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing

import pytest
from a2a.types import a2a_pb2 as pb
from google.protobuf import json_format

from livekit.agents._generated.a2a_metadata import (
    INTERRUPTED,
    INTERRUPTION_REASON,
    MERGED_INTO,
    REASON,
)
from livekit.agents.storage import AgentDBStorage, ThreadNotFound
from livekit.agents.storage.resolve import Thread

from .drivers import Workers, answer
from .resources import REQUEST_SECONDS, RecordingService, finish_cleanup

pytestmark = pytest.mark.e2e


async def test_thread_history_and_idempotency_across_workers(
    workers: Workers, thread: str, service: RecordingService
) -> None:
    """Send on A, recall on B, retry on A; one history and no duplicate tool execution."""
    first = workers.a.message("I am Alice", thread)
    sent = await workers.a.send(first)
    recalled = await workers.b.send(workers.b.message("recall", thread))
    assert sent.context_id == recalled.context_id == thread
    assert sent.status.state == recalled.status.state == pb.TaskState.TASK_STATE_COMPLETED
    assert answer(recalled) == "I am Alice"
    again = await workers.b.send(first)
    assert again.id == sent.id
    assert [(call.worker, call.terms) for call in service.calls] == [
        ("worker-a", "I am Alice"),
        ("worker-b", "recall"),
    ]
    history = await Thread(AgentDBStorage(), thread, agent_name="e2e").chat_history()
    assert [item.text_content for item in history.items if item.type == "message"] == [
        "I am Alice",
        "reply: I am Alice",
        "recall",
        "I am Alice",
    ]


async def test_input_required_answer_is_delivered_on_other_worker(
    workers: Workers, thread: str, service: RecordingService
) -> None:
    """A asks; B receives the answer as the pending tool output, without asking again."""
    asked = await workers.a.send(workers.a.message("need-email", thread))
    assert asked.status.state == pb.TaskState.TASK_STATE_INPUT_REQUIRED
    assert asked.status.message.parts[0].text == "Which email?"
    replied = await workers.b.send(workers.b.message("alice@example.com", thread))
    assert replied.status.state == pb.TaskState.TASK_STATE_COMPLETED
    assert answer(replied) == "completed: alice@example.com"
    assert [call.worker for call in service.calls] == ["worker-a", "worker-b"]
    history = await Thread(AgentDBStorage(), thread, agent_name="e2e").chat_history()
    outputs = [item for item in history.items if item.type == "function_call_output"]
    assert len(outputs) == 1 and outputs[0].output == "alice@example.com"


@pytest.mark.parametrize("confirmation", ["yes", "no"])
async def test_approval_executes_exact_terms_once_only_after_confirmation_on_other_worker(
    workers: Workers, thread: str, service: RecordingService, confirmation: str
) -> None:
    """A reads back terms; only B may execute after yes. No executes zero effects."""
    asked = await workers.a.send(workers.a.message("approve", thread))
    assert asked.status.state == pb.TaskState.TASK_STATE_INPUT_REQUIRED
    assert asked.status.message.parts[0].text == "Book slot-9 $100? Yes or no?"
    assert not [call for call in service.calls if call.operation == "book"]
    message = workers.b.message(confirmation, thread)
    result = await workers.b.send(message)
    assert result.status.state == pb.TaskState.TASK_STATE_COMPLETED
    effects = [call for call in service.calls if call.operation == "book"]
    if confirmation == "yes":
        assert [(call.worker, call.thread, call.terms) for call in effects] == [
            ("worker-b", thread, "slot-9 $100")
        ]
        assert answer(result) == "completed: booked slot-9 $100"
    else:
        assert effects == []
        assert answer(result) == "declined"
    resent = await workers.a.send(message)
    assert resent.id == result.id
    assert [call for call in service.calls if call.operation == "book"] == effects


async def test_text_interrupt_across_workers_does_not_deliver_stale_answer(
    workers: Workers, thread: str, service: RecordingService
) -> None:
    """Interrupt a blocked tool on A from B; only B's final answer reaches the caller.

    This checks text delivery. Caller audio interruption is a separate pending contract.
    """
    async with asyncio.timeout(REQUEST_SECONDS), asyncio.TaskGroup() as tasks:
        first = tasks.create_task(
            workers.a.stream(
                "message/stream", pb.SendMessageRequest(message=workers.a.message("block", thread))
            )
        )
        await service.wait_for_call(thread, "wait")
        second = tasks.create_task(
            workers.b.stream(
                "message/stream",
                pb.SendMessageRequest(message=workers.b.message("replacement", thread)),
            )
        )
        # Observe the durable waiter before allowing A's tool to return.
        storage = AgentDBStorage()
        core = await storage._open(thread)
        try:
            while True:
                lane = await core.lane_info("e2e")
                if lane is not None and (lane.waiter or "").startswith("worker-b/"):
                    break
                await asyncio.sleep(0.01)
        finally:
            core.close()
        service.release(thread)
    old, new = first.result(), second.result()
    assert old[-1].status_update.status.state == pb.TaskState.TASK_STATE_CANCELED
    metadata = dict(old[-1].status_update.metadata)
    assert metadata[INTERRUPTED] is True
    assert metadata[INTERRUPTION_REASON] == "new-message"
    assert not any(frame.HasField("artifact_update") for frame in old)
    assert new[-1].status_update.status.state == pb.TaskState.TASK_STATE_COMPLETED
    assert (
        "".join(part.text for frame in new for part in frame.artifact_update.artifact.parts)
        == "reply: replacement"
    )


async def test_inbox_burst_merges_messages_into_one_reply(
    workers: Workers, thread: str, service: RecordingService
) -> None:
    """Concurrent SMS requests on two workers produce one model turn and one answer."""
    async with asyncio.TaskGroup() as tasks:
        first = tasks.create_task(
            workers.a.send(workers.a.message("where", thread, channel="sms"), endpoint="inbox")
        )
        second = tasks.create_task(
            workers.b.send(workers.b.message("when", thread, channel="sms"), endpoint="inbox")
        )
    a, b = first.result(), second.result()
    assert a.status.state == b.status.state == pb.TaskState.TASK_STATE_COMPLETED
    merged, final = (a, b) if not a.artifacts else (b, a)
    assert not merged.artifacts
    assert dict(merged.metadata)[MERGED_INTO] == final.id
    assert answer(final)
    turns = [call for call in service.calls if call.operation == "turn"]
    assert len(turns) == 1
    assert "where" in turns[0].terms and "when" in turns[0].terms


async def test_worker_killed_mid_turn_settles_orphan_and_other_worker_continues(
    workers: Workers, thread: str, service: RecordingService
) -> None:
    """Kill A during lookup; B's next turn fails the orphan and delivers its own answer.

    The failed lookup must not create a booking. A is observed exited and reaped;
    no fixed sleep substitutes for lease acquisition by B.
    """
    events = workers.a.events(
        "message/stream", pb.SendMessageRequest(message=workers.a.message("block", thread))
    )
    async with aclosing(events):
        submitted = await anext(events)
        assert submitted.HasField("task") and submitted.task.id
        await service.wait_for_call(thread, "wait")
        workers.a_process.process.kill()
        await finish_cleanup(workers.a_process.stop())
        assert workers.a_process.reaped and workers.a_process.process.returncode is not None
    result = await workers.b.send(workers.b.message("replacement", thread))
    assert result.status.state == pb.TaskState.TASK_STATE_COMPLETED
    assert answer(result) == "reply: replacement"
    orphan = json_format.ParseDict(
        await workers.b.rpc("tasks/get", pb.GetTaskRequest(id=submitted.task.id), thread=thread),
        pb.Task(),
    )
    assert orphan.status.state == pb.TaskState.TASK_STATE_FAILED
    assert dict(orphan.metadata)[REASON] == "worker-lost"
    assert not [call for call in service.calls if call.operation == "book"]


async def test_stateless_agent_serves_two_threads_without_history_crosstalk(
    workers: Workers, thread: str, other_thread: str
) -> None:
    """One worker's shared Agent serves concurrent histories with no identity bleed."""
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(workers.a.send(workers.a.message("Alice", thread)))
        tasks.create_task(workers.a.send(workers.a.message("Bob", other_thread)))
    async with asyncio.TaskGroup() as tasks:
        first = tasks.create_task(workers.a.send(workers.a.message("recall", thread)))
        second = tasks.create_task(workers.a.send(workers.a.message("recall", other_thread)))
    a, b = first.result(), second.result()
    assert answer(a) == "Alice"
    assert answer(b) == "Bob"


async def test_threads_create_update_list_delete(
    thread: str, live_credentials: tuple[str, str, str]
) -> None:
    """Create a real database, persist attributes, list its scope, then refuse deleted reads."""
    threads = AgentDBStorage().threads
    created = await threads.get_thread(thread)
    assert created.database_id.startswith("DB_")
    await threads.update_thread(thread, attributes={"title": "E2E appointment"})
    assert (await threads.get_thread(thread)).attributes["title"] == "E2E appointment"
    page = await threads.list_threads(created.scope, limit=10)
    assert [info.thread_id for info in page.threads] == [thread]
    await threads.delete_thread(thread)
    with pytest.raises(ThreadNotFound):
        await threads.get_thread(thread)
