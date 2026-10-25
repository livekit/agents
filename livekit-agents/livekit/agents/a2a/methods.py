"""The A2A methods of one endpoint (plan 7.3), with no binding in them.

JSON-RPC 2.0 requests in, responses out: one for a unary method, a sequence for a stream.
``message/send`` and ``message/stream`` take the message through ``_core.Turn`` (plan 13.8); the
``tasks/*`` methods read the task table and the answer stream, and need a ``contextId`` in v0
(plan 3.3.8), from the params or the node's preamble. A session is named by its id and nothing
else (docs/sessions.md). Errors map as plan 5.12 lists them.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

from a2a.server.jsonrpc_models import (
    InvalidRequestError as RpcInvalidRequestError,
    MethodNotFoundError as RpcMethodNotFoundError,
)
from a2a.server.request_handlers.response_helpers import build_error_response
from a2a.utils.errors import (
    A2AError,
    InternalError,
    InvalidParamsError,
    TaskNotCancelableError,
    TaskNotFoundError,
    UnsupportedOperationError,
)
from google.protobuf import json_format
from pydantic import JsonValue

from .. import _core
from .._core import ThreadDb
from .._generated.storage import StreamInfo, TaskData, TaskRow
from .._generated.storage_run import TaskRecord
from .._generated.storage_turn import TurnResult
from ..log import logger
from ..storage import hub
from ..storage.agentdb import AgentDBStorage
from ..storage.context import EndHook, TokenHook, TurnHost, UpdateHook
from ..storage.protocol import (
    TERMINAL_TASK_STATES,
    RunRejected,
    StorageError,
    ThreadNotFound,
)
from ..storage.resolve import AgentRegistry, ResolvedThread, Thread, resolve_thread
from ..utils import shortuuid
from .auth import ANONYMOUS, AuthError, Credentials, Flag, Forbidden, Principal, Unauthorized
from .codec import (
    StoredMessage,
    StreamEvent,
    final_event,
    message_from_wire,
    metadata_from_wire,
    submitted_task,
    task_to_pb,
    to_stream_response,
)
from .extension import CALLER_LANE, KIND, KIND_DELEGATION, REASON, Extensions, pb, to_json
from .pump import ResponsePump, WireHost, start

if TYPE_CHECKING:
    from .endpoint import A2AEndpoint

S = TypeVar("S")

JsonRpc = dict[str, JsonValue]
"""One JSON-RPC envelope, as the a2a-sdk's response helpers build it."""

Params = Mapping[str, JsonValue]

# every method under both names: A2A 1.0 (the stock a2a-sdk client) and 0.3 (the plan's text)
METHODS: dict[str, str] = {
    "SendMessage": "send",
    "message/send": "send",
    "SendStreamingMessage": "stream",
    "message/stream": "stream",
    "GetTask": "get",
    "tasks/get": "get",
    "ListTasks": "list",
    "tasks/list": "list",
    "CancelTask": "cancel",
    "tasks/cancel": "cancel",
    "SubscribeToTask": "resubscribe",
    "tasks/resubscribe": "resubscribe",
    "tasks/subscribe": "resubscribe",
}
STREAMING = frozenset({"stream", "resubscribe"})

REASON_THREAD_NOT_FOUND = "thread-not-found"
READY_STREAMS = 2
"""Answer streams created ahead of the requests that will use them."""
CANCEL_WAIT = 10.0
"""How long ``tasks/cancel`` waits for a running task's tools to stop, in seconds."""
LOOKUP_LIMIT = 500
"""The lane's newest task rows a message's routing reads, as the turn's own lookups do."""
REASON_STORAGE_UNAVAILABLE = "storage-unavailable"
"""Plan 5.12: the ``failed`` reason of a message to an id the threads service does not know."""


@dataclass
class CallContext:
    """Who is calling, what they activated, and what the transport already knows."""

    principal: Principal = field(default_factory=lambda: ANONYMOUS)
    extensions: Extensions = field(default_factory=Extensions)
    resume_offset: str | None = None
    """``Last-Event-ID`` or ``params.offset``: where ``tasks/resubscribe`` resumes from."""


@dataclass
class TurnOutcome:
    """What one message produced: the turn, in the session it ended in."""

    result: TurnResult
    context_id: str
    """The thread id."""


class A2AService(Generic[S]):
    """The A2A methods of one endpoint."""

    def __init__(
        self,
        endpoint: A2AEndpoint[S],
        *,
        storage: AgentDBStorage,
        agent_name: Callable[[], str],
        credentials: Credentials | None,
        agents: AgentRegistry | None = None,
        lane: Callable[[], str] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._storage = storage
        self._agent_name = agent_name
        self._lane = lane
        self._credentials = credentials
        self._agents = agents
        self._streams: deque[asyncio.Task[StreamInfo]] = deque()

    @property
    def agent_name(self) -> str:
        """The worker's name, which the door's URL and its callers' tokens carry."""
        name = self._agent_name()
        if not name:
            raise RuntimeError(
                "a session endpoint needs the worker's agent_name: "
                "AgentServer(agent_name=) or LIVEKIT_AGENT_NAME"
            )
        return name

    @property
    def lane(self) -> str:
        """The lane this door writes: the worker's name, or its own when it was included from
        another server (docs/lanes.md)."""
        return (self._lane() if self._lane is not None else "") or self.agent_name

    @property
    def credentials(self) -> Credentials | None:
        return self._credentials

    # -- dispatch --

    async def dispatch(self, request: JsonValue, call: CallContext) -> AsyncIterator[JsonRpc]:
        """One JSON-RPC request to its responses; errors become error responses, auth failures
        raise ``AuthError`` for the binding to answer in its own terms."""
        request_id = request.get("id") if isinstance(request, Mapping) else None
        if not isinstance(request_id, (str, int)):
            request_id = None
        if not isinstance(request, Mapping) or request.get("jsonrpc") != "2.0":
            yield build_error_response(
                request_id, RpcInvalidRequestError(message="'jsonrpc' must be exactly '2.0'")
            )
            return

        name = METHODS.get(str(request.get("method") or ""))
        if name is None:
            yield build_error_response(request_id, RpcMethodNotFoundError())
            return

        params = request.get("params") or {}
        if not isinstance(params, Mapping):
            yield build_error_response(
                request_id, InvalidParamsError(message="params must be an object")
            )
            return

        try:
            async for result in self._results(name, params, call):
                yield {"jsonrpc": "2.0", "id": request_id, "result": result}
        except AuthError:
            raise
        except A2AError as e:
            yield build_error_response(request_id, e)
        except RunRejected as e:
            # plan 5.12: the reject policy refused the message, naming the running task
            running = e.running_task_id
            yield build_error_response(
                request_id,
                UnsupportedOperationError(
                    message="a turn of this lane is running; cancel it first",
                    data={"runningTaskId": running} if running else None,
                ),
            )
        except StorageError as e:
            logger.warning(
                "an agent-store error ended an A2A request",
                extra={"endpoint": self._endpoint.name, "error": str(e)},
            )
            yield build_error_response(request_id, InternalError(message=str(e)))
        except Exception as e:
            logger.exception("an A2A request failed", extra={"endpoint": self._endpoint.name})
            yield build_error_response(
                request_id, InternalError(message=f"{type(e).__name__}: {e}")
            )

    async def _results(
        self, name: str, params: Params, call: CallContext
    ) -> AsyncIterator[JsonValue]:
        """The method's results as JSON: one for a unary method, every frame of a stream."""
        if name == "stream":
            async for event in self.stream(params, call):
                yield to_json(to_stream_response(event))
        elif name == "resubscribe":
            async for frame in self.resubscribe(params, call):
                yield to_json(to_stream_response(frame))
        elif name == "send":
            yield to_json(await self.send(params, call))
        elif name == "get":
            yield to_json(await self.get_task(params, call))
        elif name == "list":
            yield to_json(await self.list_tasks(params, call))
        else:
            yield to_json(await self.cancel_task(params, call))

    # -- message/send and message/stream --

    async def send(self, params: Params, call: CallContext) -> pb.SendMessageResponse:
        """``message/send``: the turn, answered with the ``Task`` when it ends."""
        message = _message_of(params)
        thread_id = await self._resolve(message, call)
        if thread_id is None:
            record, context_id = self._unknown_session(message, call)
        else:
            pump: ResponsePump[TurnOutcome] | None = None
            try:
                pump = await self._start(thread_id, message, call.extensions)
                async for _ in pump.frames():
                    pass
                record, context_id = pump.outcome.result.task, pump.outcome.context_id
            except (RunRejected, ThreadNotFound):
                raise
            except StorageError as e:
                acked = pump.acked if pump is not None else None
                record, context_id = self._unsaved(thread_id, message, e, acked)

        configuration = params.get("configuration")
        length = configuration.get("historyLength") if isinstance(configuration, Mapping) else None
        return pb.SendMessageResponse(
            task=task_to_pb(
                record,
                context_id,
                exts=call.extensions,
                history_length=int(length) if isinstance(length, int) else None,
            )
        )

    async def stream(
        self, params: Params, call: CallContext
    ) -> AsyncIterator[StreamEvent | pb.StreamResponse]:
        """``message/stream``: the acknowledgment, progress, the artifact, the final status."""
        message = _message_of(params)
        received = time.monotonic()
        thread_id = await self._resolve(message, call)
        if thread_id is None:
            record, context_id = self._unknown_session(message, call)
            yield submitted_task(record.id, context_id)
            yield final_event(record, context_id, exts=call.extensions)
            return

        pump: ResponsePump[TurnOutcome] | None = None
        try:
            pump = await self._start(thread_id, message, call.extensions, received)
            async for event in pump.frames():
                yield event
        except (RunRejected, ThreadNotFound):
            raise
        except StorageError as e:
            acked = pump.acked if pump is not None else None
            record, context_id = self._unsaved(thread_id, message, e, acked)
            if acked is None:
                yield submitted_task(record.id, context_id)
            yield final_event(record, context_id, exts=call.extensions)

    async def turn(
        self,
        message: StoredMessage,
        *,
        thread_id: str | None,
        on_update: UpdateHook | None = None,
        on_token: TokenHook | None = None,
        principal: Principal | None = None,
        task_id: str | None = None,
    ) -> TurnOutcome:
        """One message in process, A2A semantics kept, no wire: what
        ``LiveSession(delegate=endpoint)`` calls (plan 7.2). Progress and tokens of its own task
        reach the hooks. It returns when its task ends, as the wire's response does, also when
        its turn made it the lane's holder and the holder goes on with other work."""
        call = CallContext(
            principal=principal or Principal(kind="project"),
            extensions=Extensions(livekit=True),
        )
        if thread_id and not message.get("contextId"):
            message = {**message, "contextId": thread_id}
        resolved = await self._resolve(message, call)
        if resolved is None:
            raise ThreadNotFound(f"thread {thread_id} does not exist")

        task_id = task_id or shortuuid("AST_")
        own = {task_id, str(message.get("taskId") or task_id)}
        outcome: asyncio.Future[TurnOutcome] = asyncio.get_running_loop().create_future()

        async def updated(of: str, text: str) -> None:
            if of in own and on_update is not None:
                await on_update(of, text)

        async def token(of: str, delta: str) -> None:
            if of in own and on_token is not None:
                await on_token(of, delta)

        async def end(record: TaskRecord, output: JsonValue) -> None:
            if record.id in own and record.status != "working" and not outcome.done():
                result = TurnResult(
                    task=record,
                    output=output,
                    thread_id=resolved,
                    answered_by=record.id,
                    run_id=None,
                )
                outcome.set_result(TurnOutcome(result=result, context_id=resolved))

        host = self._host(resolved, on_update=updated, on_token=token, on_end=end)
        db = await self._storage.open(resolved)
        core, turn = self._turn_on(db, host)

        async def run() -> None:
            # the holder's work goes on after this task ended, whatever this caller does
            try:
                ran = await self._run(resolved, message, core, turn, task_id)
            except Exception as e:
                if not outcome.done():
                    outcome.set_exception(e)
                else:
                    logger.warning("the lane's holder failed", extra={"error": str(e)})
                return
            if not outcome.done():
                outcome.set_result(ran)

        start(run())
        return await outcome

    def _next_stream(self) -> asyncio.Task[StreamInfo]:
        """An answer stream created ahead of the request, so a turn does not wait for one;
        the next is made as this one is taken."""
        while len(self._streams) < READY_STREAMS:
            self._streams.append(asyncio.ensure_future(self._storage.streams.create()))
        taken = self._streams.popleft()
        self._streams.append(asyncio.ensure_future(self._storage.streams.create()))
        return taken

    async def _start(
        self,
        thread_id: str,
        message: StoredMessage,
        exts: Extensions,
        received: float | None = None,
    ) -> ResponsePump[TurnOutcome]:
        """The turn in a task of its own, its frames through the pump (plan 6)."""
        self._refuse_own_lane(message)
        began = time.monotonic()
        opening = asyncio.ensure_future(self._storage.open(thread_id))
        try:
            db, created = await asyncio.gather(opening, self._next_stream())
            connected = time.monotonic()
            # the routing reads on the connection the turn runs on: a second one per request
            # is dialed while the lane's holder keeps its own
            continued = await self._continued(thread_id, db, message)
        except BaseException:
            # a request that fails before its turn leaves no connection open behind it
            def close(f: asyncio.Future[ThreadDb]) -> None:
                if not f.cancelled() and f.exception() is None:
                    f.result().close()

            opening.cancel()
            opening.add_done_callback(close)
            raise
        logger.debug(
            "a turn started",
            extra={
                "threadId": thread_id,
                "openMs": _ms(received if received is not None else began),
                "resolveMs": round((began - received) * 1000) if received is not None else None,
                "connectMs": round((connected - began) * 1000),
                "routeMs": _ms(connected),
            },
        )
        # a continued task answers on its own stream, which stayed open while it waited
        stream_id = continued or created.stream_id
        if continued:
            message = {**message, "taskId": continued}
        pump: ResponsePump[TurnOutcome] = ResponsePump(self._storage.streams, stream_id)
        wire = WireHost(
            lambda on_update, on_token, on_end: self._host(
                thread_id, on_update=on_update, on_token=on_token, on_end=on_end
            ),
            pump,
            task_id=stream_id,
            context_id=thread_id,
            lane=self.lane,
            exts=exts,
        )
        core, turn = self._turn_on(db, wire)
        if turn.policy_for(_channel_of(message)) == "live":
            # acknowledged once the message is in the lane's inbox, under the task the turn
            # found it is (``WireHost.accepted``); a continued task's stream is relayed from
            # where it ends now
            wire.live = True
            if continued:
                wire.follow_after = asyncio.ensure_future(_tail(self._storage, stream_id))

        async def produce() -> None:
            try:
                outcome = await self._run(thread_id, message, core, turn, created.stream_id)
            except Exception as e:
                if wire.holds:
                    hub.drop(thread_id, self.lane)  # the holder failed: its lane is not held here
                await pump.fail(e)
                return

            await wire.finish(outcome)
            # under live the holder that ended the task closed its stream
            ended = outcome.result.task.status in TERMINAL_TASK_STATES
            pump.finish(outcome, close=ended and not wire.live)

        start(produce())
        return pump

    def _host(
        self,
        thread_id: str,
        *,
        on_update: UpdateHook | None = None,
        on_token: TokenHook | None = None,
        on_end: EndHook | None = None,
    ) -> TurnHost[S]:
        thread = Thread(self._storage, thread_id, agent_name=self.lane)
        return TurnHost(
            self._endpoint.handler,
            thread=thread,
            state_type=self._endpoint.state_type,
            agents=self._agents,
            on_update=on_update,
            on_token=on_token,
            on_end=on_end,
        )

    def _turn_on(
        self, db: ThreadDb, host: _core.TurnHostProtocol
    ) -> tuple[_core.Thread, _core.Turn]:
        """``_core.Turn`` of this door on ``db``, the thread's database, with ``host``."""
        core = _core.Thread(db, self._storage.process_holder)
        return core, _core.Turn(core, self.lane, host, self._endpoint.turn_options)

    async def _run(
        self,
        thread_id: str,
        message: StoredMessage,
        core: _core.Thread,
        turn: _core.Turn,
        task_id: str | None,
    ) -> TurnOutcome:
        """One message through the turn; the thread's database is closed after."""
        try:
            result = await turn.run(message, task_id)
        finally:
            core.close()

        return TurnOutcome(result=result, context_id=thread_id)

    def _refuse_own_lane(self, message: StoredMessage) -> None:
        """A delegation that names this door's own lane as its caller's would wait for the lane
        its caller holds, so it can never run: refused at once."""
        caller = metadata_from_wire(message.get("metadata")).get(CALLER_LANE)
        if caller is not None and caller == self.lane:
            raise UnsupportedOperationError(
                message=f"a delegation to the caller's own lane {self.lane} cannot run",
                data={"lane": self.lane},
            )

    async def _continued(self, thread_id: str, db: ThreadDb, message: StoredMessage) -> str | None:
        """The task waiting on a question that this message continues (docs/delegation.md):
        the one its ``taskId`` names, or, for a message from the person with no ``taskId``,
        the question in front of them. A delegation without a ``taskId`` is a new task."""
        named = str(message.get("taskId") or "")
        if named:
            rows = await db.tasks(self.lane, limit=LOOKUP_LIMIT)
            found = next((r for r in rows if r.task_id == named), None)
            if found is None:
                raise TaskNotFoundError(message=f"task {named} is not in session {thread_id}")
            history = (found.data or {}).get("history") or []
            resent = any(m.get("messageId") == message.get("messageId") for m in history)
            if found.status in TERMINAL_TASK_STATES and not resent:
                raise UnsupportedOperationError(
                    message=f"task {named} is {found.status}; a follow-up is a new task",
                    data={"taskId": named, "status": found.status},
                )
            # a resend is answered by the turn with what the task holds
            return named if found.status == "input-required" and not resent else None

        metadata = metadata_from_wire(message.get("metadata"))
        if metadata.get(KIND) == KIND_DELEGATION:
            return None
        return front_question(
            await db.tasks(self.lane, status="input-required", limit=LOOKUP_LIMIT)
        )

    def _unknown_session(self, message: StoredMessage, call: CallContext) -> tuple[TaskRecord, str]:
        """Plan 5.12: an id the threads service does not know fails the task ``thread-not-found``."""
        context_id = str(message.get("contextId") or "")
        record = _failed_record(shortuuid("AST_"), self.lane, message, REASON_THREAD_NOT_FOUND)
        return record, context_id

    def _unsaved(
        self, thread_id: str, message: StoredMessage, error: StorageError, task_id: str | None
    ) -> tuple[TaskRecord, str]:
        """The turn's save could not be written: the task fails ``storage-unavailable`` in the
        answer, under the id the client was acknowledged with, and the client resends
        (docs/a2a.md)."""
        logger.warning(
            "an agent-store error ended a turn",
            extra={"endpoint": self._endpoint.name, "threadId": thread_id, "error": str(error)},
        )
        record = _failed_record(
            task_id or shortuuid("AST_"), self.lane, message, REASON_STORAGE_UNAVAILABLE
        )
        return record, thread_id

    # -- tasks/* --

    async def get_task(self, params: Params, call: CallContext) -> pb.Task:
        thread_id = await self._context_for_tasks(params, call, flag="read")
        task_id = str(params.get("id") or "")
        record = await self._find_task(thread_id, task_id)
        if record is None:
            raise TaskNotFoundError(message=f"task {task_id} is not in session {thread_id}")

        length = params.get("historyLength")
        return task_to_pb(
            record,
            thread_id,
            exts=call.extensions,
            history_length=length if isinstance(length, int) else None,
        )

    async def list_tasks(self, params: Params, call: CallContext) -> pb.ListTasksResponse:
        thread_id = await self._context_for_tasks(params, call, flag="read")
        records = await self._tasks(thread_id)

        wanted = str(params.get("status") or "")
        if wanted and wanted != "TASK_STATE_UNSPECIFIED":
            records = [
                r
                for r in records
                if wanted in (r.status, "TASK_STATE_" + r.status.upper().replace("-", "_"))
            ]

        page_size = int(str(params.get("pageSize") or 50))
        page_token = str(params.get("pageToken") or "")
        start_at = int(page_token) if page_token.isdigit() else 0
        length = params.get("historyLength")
        response = pb.ListTasksResponse(
            tasks=[
                task_to_pb(
                    r,
                    thread_id,
                    exts=call.extensions,
                    history_length=length if isinstance(length, int) else 0,
                    include_artifacts=bool(params.get("includeArtifacts", False)),
                )
                for r in records[start_at : start_at + page_size]
            ],
            page_size=page_size,
            total_size=len(records),
        )
        if start_at + page_size < len(records):
            response.next_page_token = str(start_at + page_size)
        return response

    async def cancel_task(self, params: Params, call: CallContext) -> pb.Task:
        """``tasks/cancel`` (docs/delegation.md): a task that runs gets a cancel request, its
        tools stop at their next wait point, and its holder ends it ``canceled`` with what they
        did, which the answer waits for, up to ``CANCEL_WAIT``; a task that runs nothing (it
        waits on a question, or for the lane) goes ``canceled`` now."""
        thread_id = await self._context_for_tasks(params, call, flag="write")
        task_id = str(params.get("id") or "")
        record = await self._find_task(thread_id, task_id)
        if record is None:
            raise TaskNotFoundError(message=f"task {task_id} is not in session {thread_id}")
        if record.status in TERMINAL_TASK_STATES:
            raise TaskNotCancelableError(message=f"task {task_id} is {record.status}")

        metadata = dict(record.data.metadata)
        reason = metadata_from_wire(params.get("metadata")).get(REASON)
        if reason is not None:
            metadata[REASON] = reason
        thread = Thread(self._storage, thread_id, agent_name=record.lane)
        requested = record.status == "working" and await thread.request_cancel(
            record, str(reason) if reason is not None else None
        )
        updated: TaskRecord | None = None
        if requested:
            # the holder hears it at once when it is in this process or a job of this worker,
            # and says so when the task ends; a holder on another worker is polled
            hub.signal(thread_id, record.lane, "cancel")
            deadline = time.monotonic() + CANCEL_WAIT
            while time.monotonic() < deadline:
                if (
                    hub.held_here(thread_id, record.lane) is not None
                    or hub.job_holding(thread_id, record.lane) is not None
                ):
                    await hub.settled(task_id, deadline - time.monotonic())
                else:
                    await asyncio.sleep(0.2)
                # the task's own lane, one read
                updated = next((r for r in await thread.tasks() if r.id == task_id), None)
                if updated is not None and updated.status in TERMINAL_TASK_STATES:
                    break
        else:
            await thread.cancel(record, metadata)
            updated = next((r for r in await thread.tasks() if r.id == task_id), None)

        logger.info(
            "a client canceled a task",
            extra={"taskId": task_id, "threadId": thread_id, "reason": reason},
        )
        return task_to_pb(updated or record, thread_id, exts=call.extensions)

    async def resubscribe(
        self, params: Params, call: CallContext
    ) -> AsyncIterator[StreamEvent | pb.StreamResponse]:
        """``tasks/resubscribe``: the current ``Task`` object, then the stream's frames from the
        offset, following live until the terminal one (plan 6)."""
        thread_id = await self._context_for_tasks(params, call, flag="read")
        task_id = str(params.get("id") or "")
        record = await self._find_task(thread_id, task_id)
        if record is None:
            raise TaskNotFoundError(message=f"task {task_id} is not in session {thread_id}")

        yield task_to_pb(record, thread_id, exts=call.extensions)
        offset = params.get("offset")
        after = str(offset) if offset is not None else call.resume_offset
        # a task waiting on a question keeps its stream open until it is answered; a reader
        # gets what is there, and follows only a task that runs
        live = record.status in ("submitted", "working")
        try:
            reader = self._storage.streams.read(task_id, after=after, live=live)
            while (frame := await reader.next()) is not None:
                yield json_format.ParseDict(
                    json.loads(frame), pb.StreamResponse(), ignore_unknown_fields=True
                )
        except StorageError:
            # a task older than its stream's TTL, or one written by another path: the Task
            # object above is everything there is
            logger.info(
                "no answer stream to replay for a task",
                extra={"taskId": task_id, "threadId": thread_id},
            )

    # -- the session and its tasks --

    async def _resolve(self, message: StoredMessage, call: CallContext) -> str | None:
        """The session of a message (plan 4.4) under the grant rules of 4.3.2; ``None`` for an
        id the threads service does not know."""
        grant = call.principal.grant
        context_id = str(message.get("contextId") or "")
        scope = grant.scope if grant is not None and not context_id else ""
        if grant is not None and not context_id and not grant.can_create(scope):
            raise Forbidden("the grant does not allow create")

        try:
            if context_id:
                # the thread as this process last saw it: a message a turn needs no fresh read
                info = await self._storage.get_thread(context_id)
                resolved = ResolvedThread(thread=info, created=False)
            else:
                resolved = await resolve_thread(self._storage.threads, scope=scope)
        except ThreadNotFound:
            return None

        # A grant that just created the thread sends its first message into it; later
        # messages need the thread in the grant, as protocol checks it.
        info = resolved.thread
        if grant is not None:
            if not resolved.created and not grant.can_write(info):
                raise Forbidden("the grant does not write this thread")
        elif call.principal.anonymous and not self._endpoint.public:
            raise Unauthorized("this endpoint is not public: a project token is required")
        return info.thread_id

    async def _context_for_tasks(self, params: Params, call: CallContext, *, flag: Flag) -> str:
        """The ``contextId`` every ``tasks/*`` call needs in v0 (plan 3.3.8), checked against
        the grant."""
        thread_id = str(params.get("contextId") or params.get("context_id") or "")
        if not thread_id:
            raise TaskNotFoundError(
                message="tasks/* need a contextId in v0: a task id alone names no session"
            )
        if call.principal.anonymous and not self._endpoint.public:
            raise Unauthorized("this endpoint is not public: a project token is required")

        grant = call.principal.grant
        if grant is not None:
            try:
                info = await self._storage.get_thread(thread_id)
            except ThreadNotFound:
                raise TaskNotFoundError(message=f"unknown session {thread_id}") from None
            allowed = grant.can_write(info) if flag == "write" else grant.can_read(info)
            if not allowed:
                raise Forbidden(f"the grant does not {flag} this thread")
        return thread_id

    async def _tasks(self, thread_id: str) -> list[TaskRecord]:
        return await Thread(self._storage, thread_id, agent_name=self.lane).all_tasks()

    async def _find_task(self, thread_id: str, task_id: str) -> TaskRecord | None:
        for record in await self._tasks(thread_id):
            if record.id == task_id:
                return record

        return None


def _channel_of(message: StoredMessage) -> str:
    """``livekit/channel`` of a message; a delegation's is ``delegation``, none is ``a2a``."""
    metadata = metadata_from_wire(message.get("metadata"))
    if metadata.get(KIND) == KIND_DELEGATION:
        return "delegation"
    channel = metadata.get("livekit/channel")
    return str(channel) if channel else "a2a"


async def _tail(storage: AgentDBStorage, stream_id: str) -> str | None:
    """Where a stream ends now: a continued task's frames are relayed from there."""
    try:
        reader = storage.streams.read(stream_id, after=None, live=False)
        while await reader.next() is not None:
            pass
        return str(await reader.next_offset())
    except StorageError:
        return None


def front_question(rows: list[TaskRow]) -> str | None:
    """The question in front of the person: the oldest task of the lane waiting on one, among
    those not queued behind another (docs/delegation.md). ``rows`` newest first."""
    for row in reversed(rows):
        if row.status != "input-required":
            continue
        ask = (row.internal or {}).get("ask")
        if isinstance(ask, Mapping) and ask.get("queued"):
            continue
        return row.task_id
    return None


def _ms(since: float) -> int:
    return round((time.monotonic() - since) * 1000)


def _message_of(params: Params) -> StoredMessage:
    raw = params.get("message")
    if not isinstance(raw, Mapping):
        raise InvalidParamsError(message="params.message is required")

    message = message_from_wire(raw)
    if not message["messageId"]:
        raise InvalidParamsError(message="a message needs a messageId")
    return message


def _failed_record(task_id: str, lane: str, message: StoredMessage, reason: str) -> TaskRecord:
    """A ``failed`` task the worker could not write, with ``reason`` on its status message."""
    return TaskRecord(
        id=task_id,
        lane=lane,
        seq=0,
        name=None,
        message_id=str(message.get("messageId") or ""),
        status="failed",
        status_message={
            "messageId": shortuuid("msg_"),
            "role": "agent",
            "parts": [{"kind": "text", "text": f"the task failed: {reason}"}],
            "metadata": {REASON: reason},
        },
        updated_at=int(time.time()),
        data=TaskData(history=[dict(message)], metadata={REASON: reason}),
        internal=None,
    )


__all__ = [
    "METHODS",
    "REASON_THREAD_NOT_FOUND",
    "REASON_STORAGE_UNAVAILABLE",
    "STREAMING",
    "A2AService",
    "CallContext",
    "TurnOutcome",
    "front_question",
]
