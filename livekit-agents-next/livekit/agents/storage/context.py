"""What an ``a2a_session`` handler receives, and the host the Rust turn drives.

``_core.Turn`` runs the policies and the commit. It opens the handler through
``_TurnHost._core_handle`` on the ``TurnContext`` and reads the turn's events back: each
checkpoint the run reaches, which ``_core`` answers ``continue`` or ``yield``, then the
``HandlerOutput``. The handler's progress and tokens go straight to the door's live response.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import cast

from pydantic_core import to_jsonable_python

from .. import _core
from .._generated.a2a_client import Message
from .._generated.storage_class import TaskRecord
from .._generated.storage_handler import (
    HandlerOutcome,
    HandlerOutcomeCompleted,
    HandlerOutcomeFailed,
    HandlerOutcomeInputRequired,
    HandlerOutcomeRejected,
    HandlerOutcomeValue,
    HandlerOutcomeWorking,
    HandlerRun,
)
from .._generated.storage_turn import (
    Checkpoint,
    HandlerOutput,
    Output,
    ToolExecution,
    TurnContext,
    TurnEvent,
)
from ..llm import ChatContext, ChatMessage, InputRequired
from ..llm.tool_context import Completed, Failed, Rejected, Working
from ..log import logger
from ..voice.agent_run import RunHooks, reset_run_hooks, set_run_hooks
from ..voice.run_result import RunCompleted, RunInputRequired, RunNoOutput, RunResult
from ._protocol import history_of, payload_to_item
from .context_store import ContextStore
from .errors import StorageError
from .resolve import Thread

_Handler = Callable[["A2ASessionContext"], Awaitable[object]]
"""The developer's handler: the context in, the answer out (docs/server.md)."""

_UpdateHook = Callable[[str, str], Awaitable[None]]
"""``(task_id, text)``: a progress message to the door's live response."""

_TokenHook = Callable[[str, str], Awaitable[None]]
"""``(task_id, delta)``: one streamed token to the door's live response."""


class A2ASessionContext:
    """What an ``@server.a2a_session`` handler receives: the turn as loaded, and the ways to
    report back. ``message`` is the batch as one user message, ``messages`` each message with its
    channel and time; ``message_history`` the lane's history; ``store`` supports explicit
    context persistence; ``caller_history`` a delegating caller's conversation when it
    sent one (plan 7.5)."""

    def __init__(
        self,
        turn: TurnContext,
        checkpoint: Callable[[], Awaitable[bool]],
        *,
        store: ContextStore,
        thread: Thread,
        on_update: _UpdateHook | None,
        on_token: _TokenHook | None,
    ) -> None:
        self.message = cast(ChatMessage, payload_to_item(turn.message))
        self.messages: list[Message] = list(turn.batch)
        self.message_history: ChatContext = history_of(turn.history)
        self.store = store
        self.thread = thread
        self.task_id = turn.task_id
        self.tasks: list[TaskRecord] = list(turn.tasks)
        self.caller_history: ChatContext | None = (
            ChatContext.from_dict(turn.caller_history) if turn.caller_history is not None else None
        )
        self._events: list[ToolExecution] = []
        """What the turn appends to the journal (docs/internals/storage.md#journal): each ``update`` here, the
        tool calls from the runs' results."""
        self._interrupted = False
        self._checkpoint = checkpoint
        self._on_update = on_update
        self._on_token = on_token

    async def update(self, text: str) -> None:
        """A ``working`` status frame to the client: progress, as the person hears it."""
        self._events.append(ToolExecution(phase="update", text=text))
        if self._on_update is not None:
            await self._on_update(self.task_id, text)

    async def _token(self, delta: str) -> None:
        """One piece of the reply as the model writes it, shown before the turn ends."""
        if self._on_token is not None:
            await self._on_token(self.task_id, delta)

    async def stream(self, source: AsyncIterable[str]) -> str:
        """Forward tokens as they arrive, after a checkpoint before the first one."""
        if await self._checkpoint():
            self._interrupted = True
            return ""

        parts: list[str] = []
        async for text in source:
            parts.append(text)
            if self._on_token is not None:
                await self._on_token(self.task_id, text)

        return "".join(parts)


class _TurnHost:
    """What ``_core.Turn`` drives for one door: the handler, its progress and tokens to the
    door's wire hooks."""

    def __init__(
        self,
        handler: _Handler,
        *,
        thread: Thread,
        on_update: _UpdateHook | None = None,
        on_token: _TokenHook | None = None,
    ) -> None:
        self._handler = handler
        self._thread = thread
        self._on_update = on_update
        self._on_token = on_token

    async def _core_handle(self, ctx: TurnContext) -> _TurnStream:
        """The handler opened on one turn, as the stream of its events."""

        async def update(task_id: str, text: str) -> None:
            await stream.progress(task_id, text)
            if self._on_update is not None:
                await self._on_update(task_id, text)

        async def token(task_id: str, delta: str) -> None:
            await stream.token(task_id, delta)
            if self._on_token is not None:
                await self._on_token(task_id, delta)

        stream = _TurnStream(lambda checkpoint: self._run(ctx, checkpoint, update, token))
        return stream

    async def _run(
        self,
        ctx: TurnContext,
        checkpoint: Callable[[], Awaitable[bool]],
        on_update: _UpdateHook,
        on_token: _TokenHook,
    ) -> HandlerOutput:
        """The developer's handler on one turn, the run hooks installed around it."""
        thread = self._thread
        if thread.id != ctx.thread_id:
            thread = Thread(thread._storage, ctx.thread_id, agent_name=ctx.lane)
        activation = await thread._storage.threads._core_turn_store(
            ctx.thread_id,
            ctx.lane,
            ctx.context_scopes,
            routing_state=ctx.state,
            read_only=ctx.read_only,
        )
        context = A2ASessionContext(
            ctx,
            checkpoint,
            store=ContextStore(activation=activation, scope_id=ctx.context_scope_id),
            thread=thread,
            on_update=on_update,
            on_token=on_token,
        )
        results: list[RunResult[object]] = []
        hooks = RunHooks(
            checkpoint=checkpoint,
            on_result=results.append,
            on_update=context.update,
            on_token=context._token,
            thread=thread,
            task_id=ctx.task_id,
            store=context.store,
            answer_tool=ctx.answer_tool,
            delegated=bool(ctx.batch) and all(m.channel == "delegation" for m in ctx.batch),
        )

        token = set_run_hooks(hooks)
        try:
            output = await self._handler(context)
        except StorageError:
            raise
        except Exception as e:
            logger.exception(
                "the a2a_session handler raised; the task fails",
                extra={"lane": ctx.lane, "taskId": ctx.task_id},
            )
            return self._output(
                context,
                None,
                results,
                interrupted=hooks.interrupted,
                error=f"{type(e).__name__}: {e}",
            )
        finally:
            reset_run_hooks(token)
            context.store._close()

        return self._output(context, output, results, interrupted=hooks.interrupted)

    def _output(
        self,
        ctx: A2ASessionContext,
        output: object,
        results: list[RunResult[object]],
        *,
        interrupted: bool,
        error: str | None = None,
    ) -> HandlerOutput:
        """The answer, new history, explicitly saved fields and any pending question."""
        return _core.normalize_handler_output(
            HandlerOutput(
                output=None,
                new_items=[],
                state=None,
                scope_writes=ctx.store._scope_writes,
                app_writes=ctx.store._statements,
                usage=[],
                input_required=None,
                pending_call=None,
                interrupted=ctx._interrupted or interrupted,
                metadata={},
                tool_executions=list(ctx._events),
                error=error,
                status=None,
            ),
            _handler_outcome(output),
            [
                HandlerRun(
                    items=result.new_messages(),
                    usage=result.usage,
                    verbatim=result.verbatim,
                    question=result.question if isinstance(result, RunInputRequired) else None,
                    pending_call=result._pending_call,
                )
                for result in results
            ],
        )


class _TurnStream:
    """One turn's events to ``_core``: ``{"kind": "checkpoint"}`` each time the run reaches one,
    answered through ``_core_push`` (``"continue"`` or ``"yield"``), then ``{"kind": "output"}``
    with the ``HandlerOutput``. The run starts on the first ``__anext__``; ``aclose`` cancels it."""

    def __init__(
        self, run: Callable[[Callable[[], Awaitable[bool]]], Awaitable[HandlerOutput]]
    ) -> None:
        self._run = run
        self._events: asyncio.Queue[TurnEvent | BaseException | None] = asyncio.Queue(maxsize=64)
        self._closed = False
        self._verdict: asyncio.Future[str] | None = None
        self._task: asyncio.Task[None] | None = None

    def __aiter__(self) -> _TurnStream:
        return self

    async def __anext__(self) -> TurnEvent:
        if self._task is None:
            self._task = asyncio.ensure_future(self._drive())
        try:
            event = await self._events.get()
        except asyncio.CancelledError:
            await self.aclose()
            raise
        if event is None:
            raise StopAsyncIteration
        if isinstance(event, BaseException):
            raise event
        return event

    def _core_push(self, verdict: str) -> None:
        if self._verdict is not None and not self._verdict.done():
            self._verdict.set_result(verdict)

    async def aclose(self) -> None:
        self._closed = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def progress(self, task_id: str, text: str) -> None:
        from .._generated.storage_turn import Progress

        await self._events.put(Progress(task_id=task_id, text=text))

    async def token(self, task_id: str, delta: str) -> None:
        from .._generated.storage_turn import Token

        await self._events.put(Token(task_id=task_id, delta=delta))

    async def _checkpoint(self) -> bool:
        self._verdict = asyncio.get_running_loop().create_future()
        await self._events.put(Checkpoint())
        return await self._verdict == "yield"

    async def _drive(self) -> None:
        try:
            output = await self._run(self._checkpoint)
        except asyncio.CancelledError as e:
            if not self._closed:
                await self._events.put(e)
            raise
        except Exception as e:
            # a storage error the handler met; its own errors are already the output's
            await self._events.put(e)
            return
        await self._events.put(Output(output=output))
        await self._events.put(None)


def _handler_outcome(output: object) -> HandlerOutcome:
    """Classify Python return types; Rust owns their transport meaning."""
    if isinstance(output, RunInputRequired):
        return HandlerOutcomeInputRequired(question=output.question)
    if isinstance(output, RunCompleted):
        return HandlerOutcomeValue(value=to_jsonable_python(output.output))
    if isinstance(output, RunNoOutput):
        return HandlerOutcomeValue(value=None)
    if isinstance(output, Completed):
        return HandlerOutcomeCompleted(
            value=to_jsonable_python(output.value), verbatim=output.verbatim
        )
    if isinstance(output, Failed):
        return HandlerOutcomeFailed(error=output.error)
    if isinstance(output, Rejected):
        return HandlerOutcomeRejected(reason=output.reason)
    if isinstance(output, Working):
        return HandlerOutcomeWorking(message=output.message)
    if isinstance(output, InputRequired):
        return HandlerOutcomeInputRequired(question=output.question)
    return HandlerOutcomeValue(value=to_jsonable_python(output))


__all__ = ["A2ASessionContext"]
