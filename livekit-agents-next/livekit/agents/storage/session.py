from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable
from typing import cast

from .. import _core
from .._generated.a2a_client import Message
from .._generated.run_result import _Settled
from .._generated.storage_class import TaskRecord
from .._generated.storage_turn import HandlerOutput, ToolExecution, TurnContext
from ..llm import ChatContext, ChatMessage
from ..log import logger
from ..voice.agent_run import RunHooks, reset_run_hooks, set_run_hooks
from . import StorageError, Thread
from .context import ContextStore

_Handler = Callable[["A2ASessionContext"], Awaitable[object]]


class A2ASessionContext:
    """What an ``@server.a2a_session`` handler receives: the turn as loaded, and the ways to
    report back. ``message`` is the batch as one user message, ``messages`` each message with its
    channel and time; ``message_history`` the lane's history; ``store`` supports explicit
    context persistence; ``caller_history`` a delegating caller's conversation when it
    sent one."""

    def __init__(
        self,
        turn: TurnContext,
        reports: _core._core_HandlerTurn,
        *,
        store: ContextStore,
        thread: Thread,
    ) -> None:
        self.message = cast(ChatMessage, turn.message)
        self.messages: list[Message] = list(turn.batch)
        self.message_history: ChatContext = ChatContext([row.payload for row in turn.history])
        self.store = store
        self.thread = thread
        self.task_id = turn.task_id
        self.tasks: list[TaskRecord] = list(turn.tasks)
        self.caller_history: ChatContext | None = (
            ChatContext.from_dict(turn.caller_history) if turn.caller_history is not None else None
        )
        self._events: list[ToolExecution] = []
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
    def __init__(self, handler: _Handler, *, thread: Thread) -> None:
        self._handler = handler
        self._thread = thread

    async def _core_handle(
        self, ctx: TurnContext, reports: _core._core_HandlerTurn
    ) -> HandlerOutput:
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
        results: list[_Settled] = []
        hooks = RunHooks(
            checkpoint=reports.checkpoint,
            on_result=results.append,
            on_update=context.update,
            on_token=reports.token,
            thread=thread,
            task_id=ctx.task_id,
            store=context.store,
            answer_tool=ctx.answer_tool,
            delegated=ctx.delegated,
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
        results: list[_Settled],
        *,
        interrupted: bool,
        error: str | None = None,
    ) -> HandlerOutput:
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
            output,
            results,
        )


__all__ = [
    "A2ASessionContext",
]
