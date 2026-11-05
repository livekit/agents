"""What an ``a2a_session`` handler receives, and the host the Rust turn drives.

``_core.Turn`` runs the policies and the commit. It runs the handler through one awaited
``_TurnHost._core_handle`` call on the ``TurnContext`` and a native ``_core_HandlerTurn``, which
carries the handler's checkpoints, progress and tokens back to the turn; the call returns the
``HandlerOutput``.
"""

from __future__ import annotations

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
from .._generated.storage_turn import HandlerOutput, ToolExecution, TurnContext
from .._proto import payload_to_item
from ..llm import ChatContext, ChatMessage, InputRequired
from ..llm.tool_context import Completed, Failed, Rejected, Working
from ..log import logger
from ..voice.agent_run import RunHooks, reset_run_hooks, set_run_hooks
from ..voice.run_result import RunCompleted, RunInputRequired, RunNoOutput, RunResult
from . import StorageError, Thread, history_of
from .context import ContextStore

_Handler = Callable[["A2ASessionContext"], Awaitable[object]]
"""The developer's handler: the context in, the answer out (docs/server.md)."""


class A2ASessionContext:
    """What an ``@server.a2a_session`` handler receives: the turn as loaded, and the ways to
    report back. ``message`` is the batch as one user message, ``messages`` each message with its
    channel and time; ``message_history`` the lane's history; ``store`` supports explicit
    context persistence; ``caller_history`` a delegating caller's conversation when it
    sent one (plan 7.5)."""

    def __init__(
        self,
        turn: TurnContext,
        reports: _core._core_HandlerTurn,
        *,
        store: ContextStore,
        thread: Thread,
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
        self._reports = reports

    async def update(self, text: str) -> None:
        """A ``working`` status frame to the client: progress, as the person hears it. Waits
        while the turn holds 64 unread reports."""
        self._events.append(ToolExecution(phase="update", text=text))
        await self._reports.progress(text)

    async def stream(self, source: AsyncIterable[str]) -> str:
        """Forward tokens as they arrive, after a checkpoint before the first one."""
        if await self._reports.checkpoint():
            self._interrupted = True
            return ""

        parts: list[str] = []
        async for text in source:
            parts.append(text)
            await self._reports.token(text)

        return "".join(parts)


class _TurnHost:
    """What ``_core.Turn`` runs for one door: the developer's handler, its run hooks installed
    around it."""

    def __init__(self, handler: _Handler, *, thread: Thread) -> None:
        self._handler = handler
        self._thread = thread

    async def _core_handle(
        self, ctx: TurnContext, reports: _core._core_HandlerTurn
    ) -> HandlerOutput:
        """The handler on one turn, reporting through ``reports``; its output."""
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
            reports,
            store=ContextStore(activation=activation, scope_id=ctx.context_scope_id),
            thread=thread,
        )
        results: list[RunResult[object]] = []
        hooks = RunHooks(
            checkpoint=reports.checkpoint,
            on_result=results.append,
            on_update=context.update,
            on_token=reports.token,
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


__all__ = [
    "A2ASessionContext",
]
