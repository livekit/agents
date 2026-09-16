"""Serving an ``AgentSession`` over A2A, mounted on the agent server's HTTP app.

The endpoint is a use of ``AgentServer.http``, not a server of its own: the SDK's routes for
the HTTP+JSON binding are registered under the endpoint's prefix with our executor behind
them, and ``cli.run_app(server)`` stays the one start point.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from ..log import logger
from ..utils import shortuuid
from ._codec import from_a2a_request, to_a2a_events
from ._extension import EXTENSION_URI, REASON, agent_card, as_dict, pb, struct
from ._runner import RequestRun, SessionRunner
from ._types import TaskUpdate

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


class TextSessionContext:
    """What a text session handler is given: one conversation, and where to put its session.

    The handler runs once per conversation. Build the session, start it, and hand it over::

        @server.text_session(endpoint="fare-desk", description="Answers fare questions.")
        async def fare_desk(ctx: TextSessionContext) -> None:
            session = AgentSession(llm="openai/gpt-4.1")
            await session.start(agent=FareDesk())
            ctx.attach(session)

    Nothing about the handler changes when the conversation moves into a job process later.
    """

    def __init__(self, context_id: str) -> None:
        self._context_id = context_id
        self._runner: SessionRunner | None = None

    @property
    def context_id(self) -> str:
        """The conversation. The same handler run answers every request carrying it."""
        return self._context_id

    def attach(self, session: AgentSession) -> None:
        """Hand the started session to this conversation's runner."""
        if self._runner is not None:
            raise RuntimeError("a session is already attached to this conversation")
        self._runner = SessionRunner(session)
        self._runner.attach()


TextSessionHandler = Callable[[TextSessionContext], Coroutine[Any, Any, None]]


class _Conversation:
    """One context id: the handler run that owns its session, and the requests in flight."""

    def __init__(self, context_id: str, handler: TextSessionHandler) -> None:
        self._ctx = TextSessionContext(context_id)
        self._handler = handler
        self._ready: asyncio.Task[None] | None = None
        self.runs: dict[str, RequestRun] = {}

    async def runner(self) -> SessionRunner:
        if self._ready is None:
            self._ready = asyncio.create_task(self._handler(self._ctx))
        await self._ready
        if self._ctx._runner is None:
            raise RuntimeError(
                "the text session handler returned without calling ctx.attach(session)"
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
    """Turns A2A requests into turns of the conversation's session, and back.

    One conversation is one handler run, found or created by ``context_id``.
    """

    def __init__(self, handler: TextSessionHandler) -> None:
        self._handler = handler
        self._conversations: dict[str, _Conversation] = {}
        self._by_task: dict[str, RequestRun] = {}

    def _conversation(self, context_id: str) -> _Conversation:
        if context_id not in self._conversations:
            self._conversations[context_id] = _Conversation(context_id, self._handler)
        return self._conversations[context_id]

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

        conversation = self._conversation(context_id)
        try:
            runner = await conversation.runner()
        except Exception as exc:
            logger.exception("text session handler failed", extra={"context_id": context_id})
            failed = TaskUpdate(state="failed", text=str(exc) or type(exc).__name__)
            await self._emit(event_queue, failed, task_id, context_id)
            return

        run = runner.submit(task_input, request_id=task_id)
        conversation.runs[task_id] = run
        self._by_task[task_id] = run
        try:
            async with run:
                async for update in run:
                    await self._emit(event_queue, update, task_id, context_id)
        finally:
            conversation.runs.pop(task_id, None)
            self._by_task.pop(task_id, None)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        run = self._by_task.get(context.task_id or "")
        if run is None:
            return
        reason = as_dict(context.metadata).get(REASON, "") if context.metadata else ""
        logger.debug("cancelling a task", extra={"task_id": context.task_id, "reason": reason})
        # best-effort by contract: work can finish between the decision and the stop, and the
        # run reports what it did either way
        await run.aclose()

    async def _emit(
        self, event_queue: EventQueue, update: TaskUpdate, task_id: str, context_id: str
    ) -> None:
        for event in to_a2a_events(update, task_id=task_id, context_id=context_id):
            await event_queue.enqueue_event(event)

    async def aclose(self) -> None:
        for conversation in list(self._conversations.values()):
            await conversation.aclose()
        self._conversations.clear()


def mount(
    app: FastAPI,
    *,
    endpoint: str,
    handler: TextSessionHandler,
    description: str,
    name: str | None = None,
) -> _SessionExecutor:
    """Register one A2A endpoint on ``app``, under ``/<endpoint>``.

    The card route goes on before the binding's own routes: the SDK mounts a catch-all that
    would otherwise shadow the well-known path.
    """
    executor = _SessionExecutor(handler)
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
    add_a2a_routes_to_fastapi(
        app,
        rest_routes=create_rest_routes(request_handler, path_prefix=f"{prefix}{VERSION_PREFIX}"),
    )
    logger.debug(
        "serving an agent session over A2A",
        extra={"endpoint": endpoint, "extension": EXTENSION_URI},
    )
    return executor


__all__ = ["AGENT_CARD_PATH", "TextSessionContext", "TextSessionHandler", "mount"]
