"""When something fails (design §7): a failure is reported on its task, under the id the client
was acknowledged with, and nothing runs again by itself. Seen only on the A2A wire and in what
reaches the model."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from livekit.agents import Agent, AgentServer, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.a2a.codec import metadata_from_wire
from livekit.agents.a2a.extension import REASON
from livekit.agents.delegation import A2ADelegate
from livekit.agents.llm import ChatContext, FunctionCallOutput, LLMStream
from livekit.agents.storage import AgentDBStorage, StorageUnavailable

from . import fake_agentstore
from .a2a_driver import (
    API_KEY,
    API_SECRET,
    ENDPOINT,
    EXPERT,
    Door,
    Reply,
    ScriptedLLM,
    call,
    http_client,
    no_project_env,  # noqa: F401  (autouse: the demo environment must not shape the cards)
    output_of,
)
from .agentdb_fixture import ON_PROJECT, PROJECT_URL, let_the_lease_lapse

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parent.parent


def reason_of(task: dict[str, Any]) -> str | None:
    """The ``livekit/reason`` on a task's status message."""
    metadata = metadata_from_wire(task["status"].get("message", {}).get("metadata"))
    reason = metadata.get(REASON)
    return reason if isinstance(reason, str) else None


def tool_output(model: ScriptedLLM, call_id: str) -> FunctionCallOutput | None:
    """The output of ``call_id`` as the model's last request held it."""
    outputs = [
        item
        for item in model.requests[-1].items
        if isinstance(item, FunctionCallOutput) and item.call_id == call_id
    ]
    return outputs[0] if outputs else None


# ---------------------------------------------------------------------------------------------
# a failed turn
# ---------------------------------------------------------------------------------------------


async def unsaved(ctx: A2ASessionContext[Any]) -> str:
    await ctx.update("saving")
    raise StorageUnavailable("the database went away")


@pytest.mark.rule("R15", "B1")
async def test_a_failed_save_is_reported_under_the_acked_task_id(
    door: Callable[..., Door],
) -> None:
    desk = door(unsaved)
    sent = await desk.send("hi", context_id=await desk.thread())
    acked = sent.frames[0]["task"]["id"]
    final = sent.frames[-1]["statusUpdate"]
    assert final["taskId"] == acked
    assert final["status"]["state"] == "TASK_STATE_FAILED"
    assert reason_of(final) == "storage-unavailable"


@pytest.mark.rule("R15", "B1")
async def test_a_failed_save_leaves_the_acked_task_failed(door: Callable[..., Door]) -> None:
    desk = door(unsaved)
    sent = await desk.send("hi", context_id=await desk.thread())
    # ``sent.task`` is ``tasks/get`` of the acked id
    assert sent.state == "failed"
    assert reason_of(sent.task) == "storage-unavailable"


class ModelDownOnce(ScriptedLLM):
    """A scripted model that raises on the first request ``down`` matches."""

    def __init__(self, *replies: Reply, down: Callable[[ChatContext], bool]) -> None:
        super().__init__(*replies)
        self.down: Callable[[ChatContext], bool] | None = down

    def chat(self, *, chat_ctx: ChatContext, **kwargs: Any) -> LLMStream:  # type: ignore[override]
        if self.down is not None and self.down(chat_ctx):
            self.down = None
            raise RuntimeError("the model went down")
        return super().chat(chat_ctx=chat_ctx, **kwargs)


@pytest.mark.rule("B3")
async def test_a_failed_run_keeps_the_tool_calls_that_ran(door: Callable[..., Door]) -> None:
    booked: list[str] = []
    model = ModelDownOnce(
        Reply(tool_calls=[call("book", "c1", room="12")]),
        Reply("Yes, room 12 is booked."),
        down=output_of("c1"),
    )
    front = Agent(instructions="front desk", llm=model)

    @front.tool
    async def book(ctx: RunContext, room: str) -> str:
        """Book a room."""
        booked.append(room)
        return f"booked room {room}"

    async def handler(ctx: A2ASessionContext[Any]) -> str | None:
        return (await front.run(ctx.message, message_history=ctx.message_history)).output

    desk = door(handler)
    thread = await desk.thread()
    failed = await desk.send("book room 12", context_id=thread)
    assert failed.state == "failed"

    again = await desk.send("did it go through?", context_id=thread)
    assert again.answer == "Yes, room 12 is booked."
    assert booked == ["12"]
    # the next turn's model sees the booking that ran, so it does not book again
    out = tool_output(model, "c1")
    assert out is not None and out.output == "booked room 12"


# ---------------------------------------------------------------------------------------------
# a worker that dies
# ---------------------------------------------------------------------------------------------

DYING_WORKER = """
import asyncio, os, sys

from tests import fake_agentstore
from tests.a2a_driver import Door
from tests.agentdb_fixture import project_storage

if url := os.environ.get("FAKE_AGENTSTORE_URL"):
    fake_agentstore.serve = lambda: url  # the test's fake, not one of this process's own


async def main() -> None:
    async def stuck(ctx):
        print(ctx.task_id, flush=True)
        await asyncio.Event().wait()

    await Door(project_storage("dying-worker"), stuck).send("book room 12", context_id=sys.argv[1])


asyncio.run(main())
"""
"""A worker in a process of its own: it takes the lane for one message, says the task's id and
hangs until it is killed."""


@pytest.mark.rule("R16")
async def test_a_worker_that_dies_leaves_its_task_failed_worker_lost(
    door: Callable[..., Door],
) -> None:
    ran: list[str] = []

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        ran.append(ctx.message.text_content or "")
        return "here"

    desk = door(handler)
    thread = await desk.thread()
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    if ON_PROJECT:
        env["LIVEKIT_URL"] = PROJECT_URL
    else:
        env["FAKE_AGENTSTORE_URL"] = fake_agentstore.serve()
    worker = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        DYING_WORKER,
        thread,
        cwd=REPO,
        env=env,
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        assert worker.stdout is not None
        task_id = (await asyncio.wait_for(worker.stdout.readline(), 60)).decode().strip()
        assert task_id, "the worker ended before its turn started"
        held = await desk.get_task(task_id, thread)
        assert held["status"]["state"] == "TASK_STATE_WORKING"
    finally:
        worker.kill()
        await worker.wait()
    await let_the_lease_lapse()

    sent = await desk.send("are you there?", context_id=thread)
    assert sent.state == "completed"
    lost = await desk.get_task(task_id, thread)
    assert lost["status"]["state"] == "TASK_STATE_FAILED"
    assert reason_of(lost) == "worker-lost"
    # nothing ran the dead worker's message again
    assert ran == ["are you there?"]


# ---------------------------------------------------------------------------------------------
# a delegation that fails
# ---------------------------------------------------------------------------------------------


def front_desk(model: ScriptedLLM, delegate: A2ADelegate) -> Callable[..., Any]:
    front = Agent(instructions="front desk", llm=model, tools=[delegate])

    async def handler(ctx: A2ASessionContext[Any]) -> str | None:
        return (await front.run(ctx.message, message_history=ctx.message_history)).output

    return handler


def asks_once() -> ScriptedLLM:
    return ScriptedLLM(
        Reply(tool_calls=[call("ask", "c1", request="what is the fare to Oslo")]),
        Reply("Sorry, I cannot get the fare right now.", when=output_of("c1")),
    )


@pytest.mark.rule("R19")
async def test_a_delegation_to_the_callers_own_lane_is_refused_at_once(
    door: Callable[..., Door],
) -> None:
    model = asks_once()
    handler: Callable[..., Any] | None = None

    async def desk_turn(ctx: A2ASessionContext[Any]) -> Any:
        assert handler is not None
        return await handler(ctx)

    desk = door(desk_turn)
    # a delegation that waited for the lane would fail lease-timeout after 3 s, not 60
    desk.endpoint.turn_options.wait_timeout = 3.0
    # the desk delegates to its own door: the same lane of the same thread
    own = A2ADelegate(f"http://agents.test/{ENDPOINT}", httpx_client=http_client(desk.server))
    handler = front_desk(model, own)
    thread = await desk.thread()
    try:
        sent = await desk.send("fare to Oslo?", context_id=thread)
    finally:
        await own.aclose()
    assert sent.answer == "Sorry, I cannot get the fare right now."
    out = tool_output(model, "c1")
    assert out is not None and out.is_error
    # refused before it became a task: the lane holds the desk's task alone
    assert [t["id"] for t in await desk.list_tasks(thread)] == [sent.task["id"]]


@pytest.mark.rule("R26")
async def test_a_failed_expert_is_a_tool_error_the_front_desk_tells(
    door: Callable[..., Door], expert: Callable[..., A2ADelegate]
) -> None:
    async def broken(ctx: A2ASessionContext[Any]) -> str:
        raise ValueError("the fares service is down")

    model = asks_once()
    desk = door(front_desk(model, expert(broken)))
    sent = await desk.send("fare to Oslo?", context_id=await desk.thread())
    assert sent.state == "completed"
    assert sent.answer == "Sorry, I cannot get the fare right now."
    out = tool_output(model, "c1")
    assert out is not None and out.is_error and "the fares service is down" in out.output


class GoesDown(httpx.AsyncBaseTransport):
    """Serves the expert's card, then refuses every request: an expert that went down after the
    front desk read its card."""

    def __init__(self, app: Any) -> None:
        self._app = httpx.ASGITransport(app=app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return await self._app.handle_async_request(request)
        raise httpx.ConnectError("connection refused", request=request)


@pytest.mark.rule("R26")
async def test_an_unreachable_expert_is_a_tool_error_the_front_desk_tells(
    door: Callable[..., Door], agentdb_storage: AgentDBStorage
) -> None:
    async def never(ctx: A2ASessionContext[Any]) -> str:
        raise AssertionError("an unreachable expert never runs")

    server = AgentServer(
        storage=agentdb_storage, agent_name=EXPERT, api_key=API_KEY, api_secret=API_SECRET
    )
    server.a2a_session(endpoint=EXPERT, description="Answers fares.")(never)
    http = httpx.AsyncClient(transport=GoesDown(server.http.app), base_url="http://experts.test")
    delegate = A2ADelegate(f"http://experts.test/{EXPERT}", httpx_client=http)
    model = asks_once()
    desk = door(front_desk(model, delegate))
    try:
        sent = await desk.send("fare to Oslo?", context_id=await desk.thread())
    finally:
        await delegate.aclose()
        await http.aclose()
    assert sent.state == "completed"
    assert sent.answer == "Sorry, I cannot get the fare right now."
    out = tool_output(model, "c1")
    # the reason, not a bare internal error: the front desk can say the expert is unreachable
    assert out.is_error and "connection refused" in out.output
