"""What an ``a2a_session`` handler receives (plan 7.3, 7.6.1), and the host the Rust turn calls.

``_core.Turn`` runs the policies and the commit; what it hands over is the ``TurnContext`` model
and a ``Boundary``, and what it reads back is a ``HandlerOutput``. ``TurnHost`` is the one object
it calls into (plan 13.8): ``handle`` builds the ``A2ASessionContext`` the developer's handler
takes, runs it with the run hooks of ``Agent.run()`` installed, and reports the output;
``on_update`` and ``on_token`` are the door's wire hooks, forwarded as they are.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Any, Generic, cast

from pydantic import JsonValue
from typing_extensions import TypeVar

from .._core import Boundary
from .._generated.storage_run import RunUsage, TaskRecord
from .._generated.storage_turn import (
    Background,
    HandlerOutput,
    HostEvent,
    Message,
    PendingCall,
    TurnContext,
    WaitingAsk,
)
from ..llm import ChatContext, ChatMessage, FunctionCall, FunctionCallOutput
from ..llm.tool_context import TASK_KEY
from ..log import logger
from ..voice.agent_run import RunHooks, reset_run_hooks, set_run_hooks
from ..voice.events import RunContext
from ..voice.run_result import RunResult
from .hub import (
    LaneHub,
    held_here,
    job_holding,
    lane_hub,
    polling,
    settled,
    signal,
    task_moved,
)
from .protocol import (
    TERMINAL_TASK_STATES,
    StorageError,
    dump_state,
    history_of,
    item_to_payload,
    load_state,
)
from .resolve import AgentRegistry, Thread

S = TypeVar("S", default=None)  # a door declared with no state: ``A2ASessionContext``

Handler = Callable[["A2ASessionContext[S]"], "Awaitable[object] | AsyncIterable[str]"]
"""The developer's handler: the context in, the answer out, either returned whole or yielded as
it is written (an async generator), in which case the answer is the last run's result."""

UpdateHook = Callable[[str, str], Awaitable[None]]
"""``(task_id, text)``: a progress message to the door's live response."""

TokenHook = Callable[[str, str], Awaitable[None]]
"""``(task_id, delta)``: one streamed token to the door's live response."""

EndHook = Callable[[TaskRecord, JsonValue], Awaitable[None]]
"""``(record, output)``: under ``live``, a task stopped running in a turn; its last frames."""

LOCAL_BACKSTOP = 10.0
"""A request whose lane is held in its own process is told when its task moves; it reads the
row anyway after this many seconds."""


class A2ASessionContext(Generic[S]):
    """What an ``@server.a2a_session`` handler receives: the turn as loaded, and the ways to
    report back. ``message`` is the batch as one user message, ``messages`` each message with its
    channel and time; ``message_history`` the lane's history, ``state`` the lane's state, which
    the handler mutates in place; ``caller_history`` a delegating caller's conversation when it
    sent one (plan 7.5), ``caller_lane`` its lane when it delegates in this thread."""

    def __init__(
        self,
        turn: TurnContext,
        boundary: Boundary,
        *,
        state: S,
        thread: Thread,
        on_update: UpdateHook | None,
        on_token: TokenHook | None,
    ) -> None:
        self.message = ChatMessage.model_validate(turn.message)
        self.message.extra.setdefault(TASK_KEY, turn.task_id)
        self.messages: list[Message] = list(turn.batch)
        self.message_history: ChatContext = history_of(turn.history)
        self.state = state
        self.thread = thread
        self.task_id = turn.task_id
        self.tasks: list[TaskRecord] = list(turn.tasks)
        self.caller_history: ChatContext | None = (
            ChatContext.from_dict(turn.caller_history) if turn.caller_history is not None else None
        )
        lane = turn.batch[-1].metadata.get("livekit/callerLane") if turn.batch else None
        self.caller_lane: str | None = lane if isinstance(lane, str) else None
        """The delegating caller's lane of this thread, when the caller named it: its history
        holds the person's own words, ``await ctx.thread.chat_history(ctx.caller_lane)``."""
        self.metadata: dict[str, JsonValue] = {}
        """What goes on the task's terminal row and final status event, under short keys."""
        self.events: list[HostEvent] = []
        """What the turn appends to the journal (docs/journal.md): each ``update`` here, the
        tool calls from the runs' results."""
        self.interrupted = False
        self._boundary = boundary
        self._on_update = on_update
        self._on_token = on_token

    async def update(self, text: str) -> None:
        """A ``working`` status frame to the client: progress, as the person hears it."""
        self.events.append(_tool_event("update", text=text))
        if self._on_update is not None:
            await self._on_update(self.task_id, text)

    async def token(self, delta: str) -> None:
        """One piece of the reply as the model writes it, shown before the turn ends."""
        if self._on_token is not None:
            await self._on_token(self.task_id, delta)

    async def stream(self, source: AsyncIterable[str]) -> str:
        """Forward tokens as they arrive; the boundary is checked once before the first one."""
        if await self._boundary.check():
            self.interrupted = True
            return ""

        parts: list[str] = []
        async for text in source:
            parts.append(text)
            if self._on_token is not None:
                await self._on_token(self.task_id, text)

        return "".join(parts)

    async def task(self, task_id: str) -> TaskRecord | None:
        """One task row of the lane, as it is now."""
        for record in await self.thread.tasks():
            if record.id == task_id:
                return record

        return None


class TurnHost(Generic[S]):
    """What ``_core.Turn`` calls back into for one door: the handler, and the wire hooks."""

    def __init__(
        self,
        handler: Handler[S],
        *,
        thread: Thread,
        state_type: type[S] | None,
        agents: AgentRegistry | None = None,
        on_update: UpdateHook | None = None,
        on_token: TokenHook | None = None,
        on_end: EndHook | None = None,
    ) -> None:
        self._handler = handler
        self._thread = thread
        self._state_type = state_type
        self._agents = agents if agents is not None else AgentRegistry()
        self._on_update = on_update
        self._on_token = on_token
        self._on_end = on_end

    async def handle(self, ctx: TurnContext, boundary: Boundary) -> HandlerOutput:
        """The developer's handler on one turn, the run hooks installed around it. Under
        ``live`` this host holds the lane: its turns share one live state object, which a tool
        in the background still writes, and its work is on the lane's hub."""
        hub: LaneHub | None = None
        if ctx.live:
            hub = lane_hub(self._thread.id, self._thread.agent_name)
            hub.held = True
            if hub.state is None:
                hub.state = self._load_state(ctx.state)
            state = hub.state
        else:
            state = self._load_state(ctx.state)
        context = A2ASessionContext(
            ctx,
            boundary,
            state=state,
            thread=self._thread,
            on_update=self._on_update,
            on_token=self._on_token,
        )
        results: list[RunResult[object]] = []
        resumed = bool(ctx.batch) and all(
            m.metadata.get("livekit/kind") == "background" for m in ctx.batch
        )
        # each finished piece of work with the task it belongs to
        owned = [
            (o, m.task_id or ctx.task_id)
            for m in ctx.batch
            for o in (hub.take(m.task_id or ctx.task_id) if resumed and hub is not None else [])
        ]
        for o, task in owned:
            o.fnc_call.extra[TASK_KEY] = task
        own = {ctx.task_id, *(task for _, task in owned)}

        def background(
            fnc_call: FunctionCall, work: asyncio.Task[object], run_ctx: RunContext[Any, Any]
        ) -> None:
            assert hub is not None
            hub.start(ctx.task_id, fnc_call, work, run_ctx)

        recorded = False

        async def record_message() -> None:
            # once per turn: the commit writes the message again, with the rest
            nonlocal recorded
            if recorded or resumed:
                return
            recorded = True
            message_id = ctx.batch[-1].message_id if ctx.batch else None
            await boundary.record(item_to_payload(context.message), message_id)

        hooks = RunHooks(
            boundary=boundary.check,
            on_result=results.append,
            on_update=context.update,
            on_token=context.token,
            thread_id=self._thread.id,
            agent_stack=ctx.agent_stack,
            rebuild=self._agents.rebuild,
            answer_call_id=ctx.answer_call_id,
            cut_off=ctx.cut_off if isinstance(ctx.cut_off, dict) else None,
            delegated=bool(ctx.batch) and all(m.channel == "delegation" for m in ctx.batch),
            front_question=ctx.front_question,
            relay=ctx.relay if isinstance(ctx.relay, dict) else None,
            background=background if hub is not None else None,
            # the tasks whose work ended together are resumed in one turn, one reply for all
            resume=[o for o, _ in owned] if resumed else None,
            canceled=lambda: boundary.client_canceled,
            lane=ctx.lane,
            record_message=record_message,
            running=hub.calls if hub is not None else None,
            hub=hub,
            task_id=ctx.task_id,
            open_tasks=frozenset(
                t.id for t in ctx.tasks if t.status not in TERMINAL_TASK_STATES and t.id not in own
            ),
        )

        token = set_run_hooks(hooks)
        try:
            handled = self._handler(context)
            if isinstance(handled, AsyncIterable):
                # yielded as it is written: each piece reaches the client, the answer is the
                # last run's, or the text itself when no run was made
                parts: list[str] = []
                async for delta in handled:
                    parts.append(delta)
                    await context.token(delta)
                output: object = results[-1] if results else "".join(parts)
            else:
                output = await handled
        except StorageError:
            raise
        except Exception as e:
            logger.exception(
                "the a2a_session handler raised; the task fails",
                extra={"lane": ctx.lane, "taskId": ctx.task_id},
            )
            return HandlerOutput(error=f"{type(e).__name__}: {e}")
        finally:
            reset_run_hooks(token)

        owners = {o.fnc_call.call_id: task for o, task in owned}
        return self._output(context, output, results, hub, owners)

    async def on_update(self, task_id: str, text: str) -> None:
        if self._on_update is not None:
            await self._on_update(task_id, text)

    async def on_token(self, task_id: str, delta: str) -> None:
        if self._on_token is not None:
            await self._on_token(task_id, delta)

    async def on_end(self, record: TaskRecord, output: JsonValue) -> None:
        if self._on_end is not None:
            await self._on_end(record, output)
        task_moved(record.id)

    async def discard(self, task_id: str) -> None:
        """The task ended before its background work: what the work left is dropped."""
        if (hub := held_here(self._thread.id, self._thread.agent_name)) is not None:
            hub.take(task_id)

    async def background(self, wait: float, cancel: list[str]) -> Background:
        """The lane's tools in the background, on its hub (``storage/hub.py``); the work of the
        ``cancel`` tasks stops first. A message or a cancel signaled in this process ends the
        wait at once."""
        hub = lane_hub(self._thread.id, self._thread.agent_name)
        hub.cancel(cancel)
        return await hub.background(wait)

    async def arrived(self, task_id: str) -> None:
        """This request's message is in the lane's inbox: the holder hears it now when it is in
        this process or one of this worker's jobs."""
        signal(self._thread.id, self._thread.agent_name)

    async def settled(self, task_id: str, wait: float) -> bool:
        """Waits for a turn of the task to end: told by a holder in this process or in a job
        of this worker, with a long backstop, else after ``wait``, polling agent-db."""
        thread_id, lane = self._thread.id, self._thread.agent_name
        if held_here(thread_id, lane) is not None or job_holding(thread_id, lane) is not None:
            return await settled(task_id, max(wait, LOCAL_BACKSTOP))
        polling(self._thread.id, self._thread.agent_name)
        await asyncio.sleep(wait)
        return False

    def _load_state(self, stored: JsonValue) -> S:
        if self._state_type is None:
            return cast(S, None)  # a door declared with no state: ``A2ASessionContext[None]``
        return load_state(stored, self._state_type)

    def _output(
        self,
        ctx: A2ASessionContext[S],
        output: object,
        results: list[RunResult[object]],
        hub: LaneHub | None,
        owners: dict[str, str],
    ) -> HandlerOutput:
        """What the turn commits: the answer, what the runs added, the state, the ask, the
        agent stack when a run moved it. ``owners`` maps a resumed piece of work's call to its
        task."""
        if isinstance(output, RunResult):
            # a run returned as is answers with its question while it waits, else its output;
            # one that asked while it resumed other tasks too answers those with its reply
            if output._answered is not None:
                output = output._answered
            elif output.input_required is not None:
                output = output.input_required
            else:
                output = output.output
        from ..a2a.extension import VERBATIM

        asked = next((r for r in reversed(results) if r.input_required is not None), None)
        interrupted = ctx.interrupted or any(r.interrupted for r in results)
        if interrupted:
            # an interrupted turn answers only with the reply it showed before it stopped
            output = next((r._run_output for r in reversed(results) if r.interrupted), None)
        metadata = dict(ctx.metadata)
        last = results[-1] if results else None
        if last is not None and last.verbatim is not None:
            # what must reach the person as written: the delegate relays it unchanged
            metadata.setdefault(VERBATIM, last.verbatim)
        pending = None
        if asked is not None and (call := asked.pending_call) is not None:
            pending = PendingCall(
                name=call.name, call_id=call.call_id, answer_tool=asked._answer_tool
            )
        return HandlerOutput(
            output=output,
            new_items=[item_to_payload(item) for r in results for item in r.new_messages()],
            state=dump_state(ctx.state, self._state_type) if self._state_type is not None else None,
            usage=_usage(results),
            input_required=asked.input_required if asked is not None else None,
            pending_call=pending,
            # a resumed task's question may join one its task already waited on
            asked_task=owners.get(asked.pending_call.call_id)
            or asked.pending_call.extra.get(TASK_KEY)
            if owners and asked is not None and asked.pending_call is not None
            else None,
            agent_stack=next(
                (r._agent_stack for r in reversed(results) if r._agent_stack is not None), None
            ),
            interrupted=interrupted,
            # the task waits for all its work: a tool still running answers it later
            background=any(r._background for r in results)
            or (hub is not None and hub.runs_for(ctx.task_id)),
            queued=asked is not None and asked._queued,
            relayed=next((r._relayed for r in reversed(results) if r._relayed), None),
            heard=asked._heard if asked is not None else None,
            waiting=[
                WaitingAsk(
                    task_id=owners.get(call.call_id, task),
                    question=question,
                    pending_call=PendingCall(
                        name=call.name, call_id=call.call_id, answer_tool=tool
                    ),
                    verbatim=verbatim,
                )
                for r in results
                for task, question, call, tool, verbatim in r._waiting
            ],
            metadata=metadata,
            events=[*_tool_events(results), *ctx.events],
        )


def _tool_events(results: list[RunResult[object]]) -> list[HostEvent]:
    """Each call's start and end as the runs' results hold them (docs/journal.md)."""
    events: list[HostEvent] = []
    for result in results:
        for item in result.new_messages():
            if isinstance(item, FunctionCall):
                events.append(_tool_event("started", callId=item.call_id, name=item.name))
            elif isinstance(item, FunctionCallOutput):
                events.append(
                    _tool_event("ended", callId=item.call_id, name=item.name, isError=item.is_error)
                )
    return events


def _tool_event(phase: str, **fields: JsonValue) -> HostEvent:
    return HostEvent(type="tool_execution_updated", payload={"phase": phase, **fields})


def _usage(results: list[RunResult[object]]) -> RunUsage:
    usage = RunUsage()
    for result in results:
        usage.requests += result.usage.requests
        usage.input_tokens += result.usage.input_tokens
        usage.output_tokens += result.usage.output_tokens
        usage.tool_calls += result.usage.tool_calls

    return usage


__all__ = ["A2ASessionContext", "EndHook", "Handler", "TokenHook", "TurnHost", "UpdateHook"]
