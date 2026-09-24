"""Serving an ``AgentSession`` over A2A, mounted on the agent server's HTTP app.

The endpoint is a use of ``AgentServer.http``, not a server of its own: the SDK's routes for
the HTTP+JSON binding are registered under the endpoint's prefix with our executor behind
them, and ``cli.run_app(server)`` stays the one start point.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from ..log import logger
from ..utils import aio, shortuuid
from .codec import from_a2a_request, to_a2a_events
from .extension import EXTENSION_URI, REASON, agent_card, pb, struct
from .runner import RequestRun, SessionRunner
from .types import TaskInput, TaskUpdate

try:
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
    from a2a.server.routes.rest_routes import create_rest_routes
    from a2a.server.tasks import InMemoryTaskStore
    from google.protobuf import json_format
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route
except ImportError as e:
    raise ImportError(
        "The 'a2a-sdk' package is required to serve A2A but is not installed.\n"
        "To fix this, install the optional dependency: pip install 'livekit-agents[a2a]'"
    ) from e

if TYPE_CHECKING:
    from fastapi import FastAPI

    from ..voice.agent_session import AgentSession

AGENT_CARD_PATH = "/.well-known/agent-card.json"
VERSION_PREFIX = "/v1"
"""Where the binding's methods live under the endpoint, and what the card's URL points at."""


class A2ASessionContext:
    """What an A2A session handler is given: one context, and where to put its session.

    The handler runs once per context — build the session, start it, hand it over::

        @server.a2a_session(endpoint="fare-desk", description="Answers fare questions.")
        async def fare_desk(ctx: A2ASessionContext) -> None:
            session = AgentSession(llm="openai/gpt-4.1")
            await session.start(agent=FareDesk())
            ctx.attach(session)
    """

    def __init__(
        self,
        context_id: str,
        *,
        conversation_id: str | None = None,
        caller_session_id: str | None = None,
    ) -> None:
        self._context_id = context_id
        self._conversation_id = conversation_id
        self._caller_session_id = caller_session_id
        self._runner: SessionRunner | None = None

    @property
    def context_id(self) -> str:
        """The context. The same handler run answers every request carrying it."""
        return self._context_id

    @property
    def conversation_id(self) -> str | None:
        """The caller's conversation, whose database this session persists into."""
        return self._conversation_id

    @property
    def caller_session_id(self) -> str | None:
        """The caller's own session in that conversation, which this one is the child of."""
        return self._caller_session_id

    def attach(self, session: AgentSession) -> None:
        """Hand the started session to this context's runner."""
        if self._runner is not None:
            raise RuntimeError("a session is already attached to this context")
        self._runner = SessionRunner(session)


A2ASessionHandler = Callable[[A2ASessionContext], Coroutine[Any, Any, None]]


class _Context:
    """One context id: the handler run that owns its session, and the requests in flight.

    Held until the caller says goodbye or it goes idle. Closing it closes the session, which
    checkpoints a persisted one and lets its lease go, so the next request on the context
    rehydrates it rather than starting over.
    """

    def __init__(self, context_id: str, handler: A2ASessionHandler, first_input: TaskInput) -> None:
        # the first request of a context says where it persists, and the handler runs on it
        self._ctx = A2ASessionContext(
            context_id,
            conversation_id=first_input.conversation_id,
            caller_session_id=first_input.caller_session_id,
        )
        self._handler = handler
        self._ready: asyncio.Task[None] | None = None
        self.runs: dict[str, RequestRun] = {}
        self.touched_at = time.monotonic()

    @property
    def idle_for(self) -> float:
        return 0.0 if self.runs else time.monotonic() - self.touched_at

    async def runner(self) -> SessionRunner:
        if self._ready is None:
            self._ready = asyncio.create_task(self._handler(self._ctx))
        await self._ready
        if self._ctx._runner is None:
            raise RuntimeError(
                "the A2A session handler returned without calling ctx.attach(session)"
            )
        return self._ctx._runner

    async def aclose(self) -> None:
        for run in list(self.runs.values()):
            with contextlib.suppress(Exception):
                await run.aclose()
        if self._ctx._runner is not None:
            with contextlib.suppress(Exception):
                await self._ctx._runner.aclose()
                await self._ctx._runner.session.aclose()
        if self._ready is not None and not self._ready.done():
            self._ready.cancel()


class _SessionExecutor(AgentExecutor):
    """Turns A2A requests into turns of the context's session, and back.

    One context is one handler run, found or created by ``context_id``.
    """

    def __init__(self, handler: A2ASessionHandler, *, idle_timeout: float | None) -> None:
        self._handler = handler
        self._contexts: dict[str, _Context] = {}
        self._by_task: dict[str, RequestRun] = {}
        self._idle_timeout = idle_timeout
        self._sweeper: asyncio.Task[None] | None = None
        self._binding: DefaultRequestHandler | None = None

    def _context(self, context_id: str, task_input: TaskInput) -> _Context:
        if context_id not in self._contexts:
            self._contexts[context_id] = _Context(context_id, self._handler, task_input)
        if self._sweeper is None and self._idle_timeout is not None:
            self._sweeper = asyncio.create_task(self._sweep(), name="a2a_idle_sweep")
        held = self._contexts[context_id]
        held.touched_at = time.monotonic()
        return held

    async def _sweep(self) -> None:
        """Drop contexts nobody came back to.

        The backstop behind ``lk/kind = close``: a caller that crashes says goodbye to
        nobody, and the session it leaves behind holds a model connection open.
        """
        assert self._idle_timeout is not None
        while True:
            await asyncio.sleep(self._idle_timeout / 4)
            await self._drop_idle()

    async def _drop_idle(self) -> None:
        assert self._idle_timeout is not None
        for context_id, held in list(self._contexts.items()):
            if held.idle_for < self._idle_timeout:
                continue
            logger.debug("dropping an idle context", extra={"context_id": context_id})
            self._contexts.pop(context_id, None)
            with contextlib.suppress(Exception):
                await held.aclose()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        context_id = context.context_id or shortuuid("ctx-")
        task_id = context.task_id or shortuuid("task-")

        # the acknowledgment first: a status event before the task has nothing to attach to,
        # and the caller reads its task id from here
        await event_queue.enqueue_event(
            pb.Task(
                id=task_id,
                context_id=context_id,
                status=pb.TaskStatus(state=pb.TaskState.TASK_STATE_SUBMITTED),
            )
        )

        request = pb.SendMessageRequest(message=context.message)
        if context.metadata:
            request.metadata.CopyFrom(struct(dict(context.metadata)))
        task_input = from_a2a_request(request)

        held = self._context(context_id, task_input)
        if task_input.closing:
            # the caller is done, so the context goes now rather than when it times out
            self._contexts.pop(context_id, None)
            await held.aclose()
            await self._emit(event_queue, TaskUpdate(state="completed"), task_id, context_id)
            return

        try:
            runner = await held.runner()
        except Exception as exc:
            logger.exception("the A2A session handler failed", extra={"context_id": context_id})
            failed = TaskUpdate(state="failed", text=str(exc) or type(exc).__name__)
            await self._emit(event_queue, failed, task_id, context_id)
            return

        run = runner.submit(task_input, request_id=task_id)
        held.runs[task_id] = run
        self._by_task[task_id] = run
        try:
            async with run:
                async for update in run:
                    await self._emit(event_queue, update, task_id, context_id)
        finally:
            held.runs.pop(task_id, None)
            self._by_task.pop(task_id, None)
            held.touched_at = time.monotonic()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or ""
        run = self._by_task.get(task_id)
        if run is None:
            return
        # a2a-sdk's ActiveTask.cancel builds this context without the cancel request, so the
        # reason the caller sent arrives empty until the SDK hands the params over
        reason = context.metadata.get(REASON, "")
        logger.debug("cancelling a task", extra={"task_id": task_id, "reason": reason})
        # the binding stops whatever was streaming this task before it asks us, so the end
        # the caller is owed goes out from here rather than from that stream
        ended = await run.cancel()
        await self._emit(event_queue, ended, task_id, context.context_id or "")
        with contextlib.suppress(Exception):
            await run.aclose()

    async def _emit(
        self, event_queue: EventQueue, update: TaskUpdate, task_id: str, context_id: str
    ) -> None:
        for event in to_a2a_events(update, task_id=task_id, context_id=context_id):
            await event_queue.enqueue_event(event)

    async def aclose(self) -> None:
        if self._sweeper is not None:
            await aio.cancel_and_wait(self._sweeper)
            self._sweeper = None
        for held in list(self._contexts.values()):
            await held.aclose()
        self._contexts.clear()
        if self._binding is not None:
            # the binding runs a producer and a consumer per task, and expects to be drained
            await self._binding.aclose()
            self._binding = None


def mount(
    app: FastAPI,
    *,
    endpoint: str,
    handler: A2ASessionHandler,
    description: str,
    name: str | None = None,
    idle_timeout: float | None = None,
) -> _SessionExecutor:
    """Register one A2A endpoint on ``app``, under ``/<endpoint>``.

    The card route goes on before the binding's own routes: the SDK mounts a catch-all that
    would otherwise shadow the well-known path.
    """
    executor = _SessionExecutor(handler, idle_timeout=idle_timeout)
    card_name = name or endpoint
    prefix = f"/{endpoint}"

    async def serve_card(request: Request) -> JSONResponse:
        # the endpoint does not know its own public address until something asks for it
        base = str(request.base_url).rstrip("/")
        card = agent_card(
            url=f"{base}{prefix}{VERSION_PREFIX}", name=card_name, description=description
        )
        return JSONResponse(json_format.MessageToDict(card))

    app.router.routes.append(Route(f"{prefix}{AGENT_CARD_PATH}", serve_card, methods=["GET"]))

    # the handler needs a card too, for what it advertises back to a client
    handler_card = agent_card(
        url=f"{prefix}{VERSION_PREFIX}", name=card_name, description=description
    )
    request_handler = DefaultRequestHandler(
        agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=handler_card
    )
    executor._binding = request_handler
    add_a2a_routes_to_fastapi(
        app,
        rest_routes=create_rest_routes(request_handler, path_prefix=f"{prefix}{VERSION_PREFIX}"),
    )
    logger.debug(
        "serving an agent session over A2A",
        extra={"endpoint": endpoint, "extension": EXTENSION_URI},
    )
    return executor


__all__ = [
    "AGENT_CARD_PATH",
    "A2ASessionContext",
    "A2ASessionHandler",
    "mount",
]
