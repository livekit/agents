"""What an ``a2a_session`` handler receives, and the host the Rust turn drives.

``_core.Turn`` runs the policies and the commit. It opens the handler through
``TurnHost._core_handle`` on the ``TurnContext`` and reads the turn's events back: each
checkpoint the run reaches, which ``_core`` answers ``continue`` or ``yield``, then the
``HandlerOutput``. The handler's progress and tokens go straight to the door's live response.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Any, Generic, cast

from pydantic import JsonValue
from typing_extensions import TypeVar

from .._generated.storage_run import RunUsage, TaskRow
from .._generated.storage_turn import HandlerOutput, HostEvent, Message, PendingCall, TurnContext
from ..llm import ChatContext, ChatMessage, FunctionCall, FunctionCallOutput
from ..log import logger
from ..voice.agent_run import RunHooks, reset_run_hooks, set_run_hooks
from ..voice.run_result import RunResult
from .protocol import (
    StorageError,
    dump_state,
    history_of,
    item_to_payload,
    load_state,
    payload_to_item,
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


class A2ASessionContext(Generic[S]):
    """What an ``@server.a2a_session`` handler receives: the turn as loaded, and the ways to
    report back. ``message`` is the batch as one user message, ``messages`` each message with its
    channel and time; ``message_history`` the lane's history, ``state`` the lane's state, which
    the handler mutates in place; ``caller_history`` a delegating caller's conversation when it
    sent one (plan 7.5)."""

    def __init__(
        self,
        turn: TurnContext,
        checkpoint: Callable[[], Awaitable[bool]],
        *,
        state: S,
        thread: Thread,
        on_update: UpdateHook | None,
        on_token: TokenHook | None,
    ) -> None:
        self.message = cast(ChatMessage, payload_to_item(turn.message))
        self.messages: list[Message] = list(turn.batch)
        self.message_history: ChatContext = history_of(turn.history)
        self.state = state
        self.thread = thread
        self.task_id = turn.task_id
        self.tasks: list[TaskRow] = list(turn.tasks)
        self.caller_history: ChatContext | None = (
            ChatContext.from_dict(turn.caller_history) if turn.caller_history is not None else None
        )
        self.metadata: dict[str, JsonValue] = {}
        """What goes on the task's terminal row and final status event, under short keys."""
        self.events: list[HostEvent] = []
        """What the turn appends to the journal (docs/journal.md): each ``update`` here, the
        tool calls from the runs' results."""
        self.interrupted = False
        self._checkpoint = checkpoint
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
        """Forward tokens as they arrive, after a checkpoint before the first one."""
        if await self._checkpoint():
            self.interrupted = True
            return ""

        parts: list[str] = []
        async for text in source:
            parts.append(text)
            if self._on_token is not None:
                await self._on_token(self.task_id, text)

        return "".join(parts)

    async def task(self, task_id: str) -> TaskRow | None:
        """One task row of the lane, as it is now."""
        for record in await self.thread.tasks():
            if record.task_id == task_id:
                return record

        return None


class TurnHost(Generic[S]):
    """What ``_core.Turn`` drives for one door: the handler, its progress and tokens to the
    door's wire hooks."""

    def __init__(
        self,
        handler: Handler[S],
        *,
        thread: Thread,
        state_type: type[S] | None,
        agents: AgentRegistry | None = None,
        on_update: UpdateHook | None = None,
        on_token: TokenHook | None = None,
    ) -> None:
        self._handler = handler
        self._thread = thread
        self._state_type = state_type
        self._agents = agents if agents is not None else AgentRegistry()
        self._on_update = on_update
        self._on_token = on_token

    def _core_handle(self, ctx: dict[str, Any]) -> TurnStream:
        """The handler opened on one turn, as the stream of its events."""
        turn = TurnContext.model_validate(ctx)
        return TurnStream(lambda checkpoint: self._run(turn, checkpoint))

    async def _run(
        self, ctx: TurnContext, checkpoint: Callable[[], Awaitable[bool]]
    ) -> HandlerOutput:
        """The developer's handler on one turn, the run hooks installed around it."""
        context = A2ASessionContext(
            ctx,
            checkpoint,
            state=self._load_state(ctx.state),
            thread=self._thread,
            on_update=self._on_update,
            on_token=self._on_token,
        )
        results: list[RunResult[object]] = []
        hooks = RunHooks(
            checkpoint=checkpoint,
            on_result=results.append,
            on_update=context.update,
            on_token=context.token,
            thread_id=self._thread.id,
            agent_stack=ctx.agent_stack,
            rebuild=self._agents.rebuild,
            answer_tool=ctx.answer_tool,
            delegated=bool(ctx.batch) and all(m.channel == "delegation" for m in ctx.batch),
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

        return self._output(context, output, results)

    def _load_state(self, stored: JsonValue) -> S:
        if self._state_type is None:
            return cast(S, None)  # a door declared with no state: ``A2ASessionContext[None]``
        return load_state(stored, self._state_type)

    def _output(
        self, ctx: A2ASessionContext[S], output: object, results: list[RunResult[object]]
    ) -> HandlerOutput:
        """What the turn commits: the answer, what the runs added, the state, the ask, the
        agent stack when a run moved it."""
        if isinstance(output, RunResult):
            # a run returned as is answers with its question while it waits, else its output
            output = output.input_required if output.input_required is not None else output.output
        from ..a2a.extension import VERBATIM

        asked = next((r for r in reversed(results) if r.input_required is not None), None)
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
            agent_stack=next(
                (r._agent_stack for r in reversed(results) if r._agent_stack is not None), None
            ),
            interrupted=ctx.interrupted or any(r.interrupted for r in results),
            metadata=metadata,
            events=[*_tool_events(results), *ctx.events],
        )


class TurnStream:
    """One turn's events to ``_core``: ``{"kind": "checkpoint"}`` each time the run reaches one,
    answered through ``_core_push`` (``"continue"`` or ``"yield"``), then ``{"kind": "output"}``
    with the ``HandlerOutput``. The run starts on the first ``__anext__``; ``aclose`` cancels it."""

    def __init__(
        self, run: Callable[[Callable[[], Awaitable[bool]]], Awaitable[HandlerOutput]]
    ) -> None:
        self._run = run
        self._events: asyncio.Queue[dict[str, Any] | BaseException | None] = asyncio.Queue()
        self._verdict: asyncio.Future[str] | None = None
        self._task: asyncio.Task[None] | None = None

    def __aiter__(self) -> TurnStream:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._task is None:
            self._task = asyncio.ensure_future(self._drive())
        event = await self._events.get()
        if event is None:
            raise StopAsyncIteration
        if isinstance(event, BaseException):
            raise event
        return event

    def _core_push(self, verdict: str) -> None:
        if self._verdict is not None and not self._verdict.done():
            self._verdict.set_result(verdict)

    async def aclose(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _checkpoint(self) -> bool:
        self._verdict = asyncio.get_running_loop().create_future()
        self._events.put_nowait({"kind": "checkpoint"})
        return await self._verdict == "yield"

    async def _drive(self) -> None:
        try:
            output = await self._run(self._checkpoint)
        except asyncio.CancelledError as e:
            self._events.put_nowait(e)
            raise
        except Exception as e:
            # a storage error the handler met; its own errors are already the output's
            self._events.put_nowait(e)
            return
        self._events.put_nowait({"kind": "output", "output": output.model_dump(mode="json")})
        self._events.put_nowait(None)


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


__all__ = ["A2ASessionContext", "Handler", "TokenHook", "TurnHost", "TurnStream", "UpdateHook"]
