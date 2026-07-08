from __future__ import annotations

import asyncio
import functools
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from typing_extensions import TypedDict

from .. import utils
from ..llm.chat_context import ChatContext, ChatItem
from ..llm.tool_context import (
    CONFIRM_DUPLICATE_PARAM,
    DuplicateMode,
    FunctionTool,
    RawFunctionTool,
    StopResponse,
    Tool,
    ToolError,
    ToolFlag,
    Toolset,
    function_tool,
)
from ..llm.utils import prepare_function_arguments
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from .events import (
    RunContext,
    ToolCallEnded,
    ToolCallStarted,
    ToolExecutionUpdatedEvent,
    ToolReplyUpdated,
)

if TYPE_CHECKING:
    from ..durable_scheduler import DurableScheduler
    from .agent import Agent
    from .agent_activity import AgentActivity
    from .agent_session import AgentSession
    from .speech_handle import SpeechHandle


UPDATE_TEMPLATE = """The tool `{function_name}` has updated, message: {message}
The task is still running, so DON'T make up or give information not included in the message above."""

DUPLICATE_REJECT = """Same tool `{function_name}` is already running:
{fnc_calls_text}
If you want to cancel the existing one, call `lk_agents_cancel_task` with call_id.
Only do this when user explicitly requests it."""

DUPLICATE_CONFIRM = """Same tool `{function_name}` is already running:
{fnc_calls_text}
Re-call with confirm duplicate True to run a duplicate if needed,
or if you want to cancel the existing one, call `lk_agents_cancel_task` with call_id.
Only run duplicate or cancel the existing one when user explicitly requests it."""

# used when the pending update is the most recent item in chat_ctx — the agent
# can't have already talked about it.
REPLY_INSTRUCTIONS_AT_TAIL = """New results arrived from background tool calls (call_ids: {call_ids}).
Summarize the results naturally. Do NOT repeat information you have already told the user."""

# used when newer items have been appended after the pending update — the agent
# may have already verbalized the result in its most recent turn.
REPLY_INSTRUCTIONS_MAYBE_COVERED = """New results arrived from background tool calls (call_ids: {call_ids}).
You may have already mentioned them in your most recent replies.
If you already told the user everything in these results, reply with an empty response (no text at all).
Otherwise, summarize only what you have not said yet, with a natural transition.
Never repeat information you have already told the user."""


class UpdatePromptArgs(TypedDict):
    """Args for the ``update`` template."""

    function_name: str
    call_id: str
    message: str


class DuplicatePromptArgs(TypedDict):
    """Args for the ``duplicate_reject`` / ``duplicate_confirm`` templates."""

    function_name: str
    fnc_calls_json: list[str]
    """JSON dump per in-flight FunctionCall — use this from callable templates."""
    fnc_calls_text: str
    """``fnc_calls_json`` joined by newlines — what the default string templates use."""


class ReplyPromptArgs(TypedDict):
    """Args for the ``reply_at_tail`` / ``reply_maybe_covered`` templates."""

    call_ids: list[str]


class AsyncToolOptions(TypedDict, total=False):
    """System-message templates injected around async tool dispatch.

    Each field is either a ``str.format()`` template or a callable returning a string,
    with the args typed as ``UpdatePromptArgs`` / ``DuplicatePromptArgs`` /
    ``ReplyPromptArgs``. Unmentioned keys keep their defaults.
    """

    update_template: str | Callable[[UpdatePromptArgs], str]
    """Wraps a user-provided ``ctx.update(message)`` string before it lands in chat_ctx."""
    duplicate_reject_template: str | Callable[[DuplicatePromptArgs], str]
    """Sent to the LLM when ``on_duplicate='reject'`` blocks a duplicate call."""
    duplicate_confirm_template: str | Callable[[DuplicatePromptArgs], str]
    """Sent to the LLM when ``on_duplicate='confirm'`` requires re-call with confirmation."""
    reply_at_tail_template: str | Callable[[ReplyPromptArgs], str]
    """Instruction for the deferred reply when the pending update is still the tail of chat_ctx."""
    reply_maybe_covered_template: str | Callable[[ReplyPromptArgs], str]
    """Instruction for the deferred reply when newer items came after the pending update."""


class ToolHandlingOptions(TypedDict, total=False):
    """Configuration for the tool handling system.

    Can be passed as a plain dict::

        AgentSession(
            tool_handling={
                "async_options": {"update_template": "..."},
            },
        )

    Set on ``AgentSession``, ``Agent``, or ``AsyncToolset`` (most specific wins).
    """

    async_options: AsyncToolOptions
    """Templates injected around async tool dispatch (``ctx.update()``, duplicate
    handling, coalesced replies). Unmentioned keys keep their defaults."""


def _render(template: str | Callable[[Any], str], args: dict[str, Any]) -> str:
    """Render a template: callables receive ``args``; strings use ``str.format(**args)``."""
    if callable(template):
        return template(args)
    return template.format(**args)


_ASYNC_TOOL_OPTIONS_DEFAULTS: AsyncToolOptions = {
    "update_template": UPDATE_TEMPLATE,
    "duplicate_reject_template": DUPLICATE_REJECT,
    "duplicate_confirm_template": DUPLICATE_CONFIRM,
    "reply_at_tail_template": REPLY_INSTRUCTIONS_AT_TAIL,
    "reply_maybe_covered_template": REPLY_INSTRUCTIONS_MAYBE_COVERED,
}


def _resolve_async_tool_options(
    config: AsyncToolOptions | None = None,
) -> AsyncToolOptions:
    """Return a fully-populated ``AsyncToolOptions`` with defaults filled in for absent keys."""
    if config is None:
        return AsyncToolOptions(**_ASYNC_TOOL_OPTIONS_DEFAULTS)
    return AsyncToolOptions(**{**_ASYNC_TOOL_OPTIONS_DEFAULTS, **config})


# session-scoped view shared across executors, so cancel_task / get_running_tasks
# see all tasks of their session but never a nested session's. weak-keyed so a
# dropped session can't leak its tasks.
_RunningTasks: weakref.WeakKeyDictionary[AgentSession, dict[str, _RunningTask]] = (
    weakref.WeakKeyDictionary()
)


@function_tool(name="lk_agents_get_running_tasks")
async def get_running_tasks(ctx: RunContext) -> list[dict]:
    """Get the list of running tool calls that are cancellable."""
    return [
        task.ctx.function_call.model_dump()
        for task in _RunningTasks.get(ctx.session, {}).values()
        if task.allow_cancellation
    ]


@function_tool(name="lk_agents_cancel_task")
async def cancel_task(ctx: RunContext, call_id: str) -> str:
    """Cancel a running tool call by call_id."""
    task = _RunningTasks.get(ctx.session, {}).get(call_id)
    if task is None:
        raise ToolError(f"Task {call_id} not found")

    if not await task.executor.cancel(call_id):
        raise ToolError(f"Task {call_id} not found or already completed")
    return f"Task {call_id} cancelled successfully."


def has_cancellable_tool(tools: Sequence[Tool | Toolset]) -> bool:
    """Return True if any tool (or nested toolset tool) has ``ToolFlag.CANCELLABLE``."""
    for tool in tools:
        if isinstance(tool, (FunctionTool, RawFunctionTool)):
            if ToolFlag.CANCELLABLE in tool.info.flags:
                return True
        elif isinstance(tool, Toolset):
            if has_cancellable_tool(tool.tools):
                return True
    return False


@dataclass
class _RunningTask:
    ctx: RunContext
    exe_task: asyncio.Task[Any]
    executor: _ToolExecutor
    allow_cancellation: bool


@dataclass
class _PendingUpdate:
    ctx: RunContext
    items: list[ChatItem]
    target: Agent  # agent that received the eager chat_ctx insert


class _ToolExecutor:
    """Lifecycle manager for in-flight tool calls.

    Activity-scoped (``owning_activity`` set): tasks belong to one AgentActivity
    and are cancelled or awaited on drain depending on ``allow_cancellation``.

    Session-scoped (``owning_activity=None``): tasks survive agent handoff; replies
    are delivered to whichever agent is current at delivery time.
    """

    def __init__(
        self,
        *,
        owning_activity: AgentActivity | None = None,
        async_tool_options: AsyncToolOptions | None = None,
    ) -> None:
        self._running_tasks: dict[str, _RunningTask] = {}
        self._duplicate_check_lock = asyncio.Lock()

        self._pending_updates: list[_PendingUpdate] = []
        self._reply_task: asyncio.Task[None] | None = None

        self._owning_activity: AgentActivity | None = owning_activity
        self._tool_options: AsyncToolOptions = _resolve_async_tool_options(async_tool_options)

    def set_owning_activity(self, activity: AgentActivity | None) -> None:
        self._owning_activity = activity

    def set_tool_options(self, options: AsyncToolOptions) -> None:
        """Replace the async tool templates. Caller must pre-resolve defaults."""
        self._tool_options = options

    @property
    def has_running_tasks(self) -> bool:
        return bool(self._running_tasks)

    @property
    def has_cancellable_running_tasks(self) -> bool:
        return any(t.allow_cancellation for t in self._running_tasks.values())

    async def execute(
        self,
        *,
        tool: FunctionTool | RawFunctionTool,
        run_ctx: RunContext,
        raw_arguments: dict[str, Any],
        mock: Callable[..., Any] | None = None,
        durable_scheduler: DurableScheduler | None = None,
    ) -> Any:
        """Run ``tool``. Returns when the first ``ctx.update()`` lands or the tool returns."""
        info = tool.info

        if ToolFlag.DURABLE in info.flags:
            if durable_scheduler is None:
                raise RuntimeError("a durable tool requires a durable scheduler")
            return await self._execute_durable(
                tool=tool,
                run_ctx=run_ctx,
                raw_arguments=raw_arguments,
                mock=mock,
                durable_scheduler=durable_scheduler,
            )

        call_id = run_ctx.function_call.call_id
        fnc_name = run_ctx.function_call.name
        on_duplicate: DuplicateMode = info.on_duplicate
        allow_cancellation: bool = ToolFlag.CANCELLABLE in info.flags

        confirm_duplicate: bool | None = None
        if on_duplicate == "confirm":
            confirm_duplicate = bool(raw_arguments.pop(CONFIRM_DUPLICATE_PARAM, False))

        duplicate_result = await self._check_duplicate(
            fnc_name, on_duplicate=on_duplicate, confirm_duplicate=confirm_duplicate
        )
        if duplicate_result is not None:
            logger.debug(
                "duplicate tool call rejected",
                extra={"call_id": call_id, "function": fnc_name},
            )
            return duplicate_result

        if call_id in self._running_tasks:
            raise ValueError(f"Task already running for call_id: {call_id}")

        # the future is how RunContext.update() talks back to dispatch
        first_update_fut = asyncio.Future[Any]()
        run_ctx._attach_executor(self, first_update_fut)

        # run the tool and return its raw output (or the caught exception); _on_done
        # derives the call's single terminal entry from how the task ended
        async def _execute_tool() -> Any:
            try:
                fnc_args, fnc_kwargs = prepare_function_arguments(
                    fnc=tool, json_arguments=raw_arguments, call_ctx=run_ctx
                )
                if mock is not None:
                    from .run_result import _run_mock

                    output = await _run_mock(mock, *fnc_args, **fnc_kwargs)
                else:
                    output = await tool(*fnc_args, **fnc_kwargs)
            except asyncio.CancelledError:
                logger.debug("tool cancelled", extra={"call_id": call_id, "function": fnc_name})
                if not first_update_fut.done():
                    first_update_fut.set_result(None)
                raise  # _on_done emits the cancelled terminal
            except Exception as e:
                output = e

            if not first_update_fut.done():
                # tool returned without ctx.update() — surface the result to dispatch
                if isinstance(output, BaseException):
                    first_update_fut.set_exception(output)
                else:
                    first_update_fut.set_result(output)
                return output

            if output is None or isinstance(output, StopResponse):
                return output

            # the first update has already been returned to dispatch, so an Agent
            # return now has no surface to carry an agent_task back
            from .agent import Agent

            if isinstance(output, Agent):
                logger.error(
                    f"tool `{fnc_name}` returned an Agent after ctx.update(); "
                    "agent handoff after a progress update is not supported",
                    extra={"call_id": call_id, "function": fnc_name},
                )
                raise RuntimeError("agent handoff after a progress update is not supported")

            if isinstance(output, BaseException):
                if isinstance(output, ToolError):
                    logger.warning(
                        "ToolError while executing tool: %s",
                        output.message,
                        extra={"function": fnc_name, "call_id": call_id},
                    )
                else:
                    logger.error(
                        "exception occurred while executing tool",
                        extra={"function": fnc_name, "call_id": call_id},
                        exc_info=output,
                    )

            # final return goes through the coalescer as a synthetic output
            pair = run_ctx._make_update_pair(output, call_id_suffix="_final")
            run_ctx._updates.append(pair)
            await self._enqueue_reply(run_ctx, [pair[0], pair[1]])
            return output

        exe_task = asyncio.create_task(_execute_tool(), name=f"tool_exec_{fnc_name}")
        from .agent import _pass_through_activity_task_info

        _pass_through_activity_task_info(exe_task)

        running_task = _RunningTask(
            ctx=run_ctx,
            exe_task=exe_task,
            executor=self,
            allow_cancellation=allow_cancellation,
        )
        self._running_tasks[call_id] = running_task

        session = run_ctx.session
        _RunningTasks.setdefault(session, {})[call_id] = running_task

        session.emit(
            "tool_execution_updated",
            ToolExecutionUpdatedEvent(update=ToolCallStarted(function_call=run_ctx.function_call)),
        )

        def _on_done(task: asyncio.Task[Any]) -> None:
            self._running_tasks.pop(call_id, None)
            if (session_tasks := _RunningTasks.get(session)) is not None:
                session_tasks.pop(call_id, None)
            # detach so a stashed RunContext can't drive the executor post-completion
            run_ctx._detach_executor()

            # how the task ended: a returned value, a raised exception, or cancellation
            try:
                output = task.result()
            except BaseException as e:
                output = e

            if not first_update_fut.done():
                first_update_fut.set_result(None)  # cancelled before the first update

            # one terminal entry per call; deferred entries use the _final id
            from .agent import Agent

            status: Literal["done", "error", "cancelled"]
            message: str | None
            if task.cancelled() or isinstance(output, asyncio.CancelledError):
                status, message = "cancelled", None
            elif isinstance(output, BaseException) and not isinstance(output, StopResponse):
                status, message = "error", str(output)
            elif output is None or isinstance(output, (StopResponse, Agent)):
                status, message = "done", None
            else:
                status, message = "done", str(output)

            entry_id = call_id + "_final" if run_ctx._updates else call_id
            session.emit(
                "tool_execution_updated",
                ToolExecutionUpdatedEvent(
                    update=ToolCallEnded(
                        id=entry_id, call_id=call_id, message=message, status=status
                    )
                ),
            )

        exe_task.add_done_callback(_on_done)

        return await first_update_fut

    async def _execute_durable(
        self,
        *,
        tool: FunctionTool | RawFunctionTool,
        run_ctx: RunContext,
        raw_arguments: dict[str, Any],
        mock: Callable[..., Any] | None,
        durable_scheduler: DurableScheduler,
    ) -> Any:
        """Drive a durable tool through the durable scheduler.

        The scheduler replays the tool step by step, checkpointing each awaited durable
        op (e.g. ``EffectCall``), so it must drive the tool coroutine directly — the normal
        executor path (detached task + first-update future) can't be checkpointed/replayed.
        The executor's interactive features (``ctx.update()``, filler, cancellation) aren't
        available here; ``ctx.update()`` raises (see ``RunContext.update``).
        """
        from .generation import _DurableExecutionMetadata

        run_ctx._durable = True
        fnc_args, fnc_kwargs = prepare_function_arguments(
            fnc=tool, json_arguments=raw_arguments, call_ctx=run_ctx
        )
        fnc_callable: Callable[[], Any]
        if mock is not None:
            from .run_result import _run_mock

            fnc_callable = functools.partial(_run_mock, mock, *fnc_args, **fnc_kwargs)
        else:
            fnc_callable = functools.partial(tool, *fnc_args, **fnc_kwargs)

        speech_handle = run_ctx.speech_handle
        return await durable_scheduler.execute(
            cast("Callable[[], Any]", fnc_callable),
            metadata=_DurableExecutionMetadata(
                num_steps=speech_handle.num_steps,
                function_call=run_ctx.function_call.model_dump_json(),
                allow_interruptions=speech_handle.allow_interruptions,
                input_details=speech_handle.input_details,
            ),
        )

    async def cancel(self, call_id: str) -> bool:
        task = self._running_tasks.get(call_id)
        if task is None:
            return False

        if not task.allow_cancellation:
            raise ToolError(f"Tool call {call_id} is not cancellable")

        if not task.ctx.speech_handle.allow_interruptions:
            raise ToolError(
                f"Tool call {call_id} is not cancellable because interruptions are disallowed"
            )
        await utils.aio.cancel_and_wait(task.exe_task)
        return True

    async def cancel_all(self, *, cancellable_only: bool = False) -> None:
        """Cancel all running tasks. When ``cancellable_only=True``, tasks with
        ``allow_cancellation=False`` are awaited to completion instead (used by drain)."""
        if cancellable_only:
            to_cancel = [t.exe_task for t in self._running_tasks.values() if t.allow_cancellation]
            to_wait = [t.exe_task for t in self._running_tasks.values() if not t.allow_cancellation]
        else:
            to_cancel = [t.exe_task for t in self._running_tasks.values()]
            to_wait = []

        if to_cancel:
            await utils.aio.cancel_and_wait(*to_cancel)
        if to_wait:
            await asyncio.gather(*to_wait, return_exceptions=True)

    async def aclose(self) -> None:
        """Cancel everything and drop any buffered replies."""
        self._pending_updates.clear()
        tasks = [task.exe_task for task in self._running_tasks.values()]
        if self._reply_task is not None:
            tasks.append(self._reply_task)
        if tasks:
            await utils.aio.cancel_and_wait(*tasks)
        self._running_tasks.clear()

    async def drain(self) -> None:
        """Cancel cancellable tools, await the rest. Reply delivery is left running;
        ``_deliver_reply`` drops itself when its target activity closes."""
        await self.cancel_all(cancellable_only=True)

    async def _enqueue_reply(self, ctx: RunContext, items: list[ChatItem]) -> None:
        # eager insert so a reply firing before delivery sees the items
        target = (
            self._owning_activity.agent
            if self._owning_activity is not None
            else ctx.session.current_agent
        )
        chat_ctx = target.chat_ctx.copy()
        chat_ctx.insert(items)
        await target.update_chat_ctx(chat_ctx)
        ctx.session.history.insert(items)

        self._pending_updates.append(_PendingUpdate(ctx=ctx, items=items, target=target))

        if self._reply_task is None or self._reply_task.done():
            self._reply_task = asyncio.create_task(
                self._deliver_reply(ctx.session), name="tool_executor_deliver_reply"
            )
            # let an active RunResult wait for the deferred reply to land
            run_state = ctx.session._global_run_state
            if run_state is not None:
                run_state._watch_handle(self._reply_task)

    async def _deliver_reply(self, session: AgentSession) -> None:
        from .agent_activity import ActivityClosedError

        target_agent: Agent
        try:
            if self._owning_activity is not None:
                await self._owning_activity.wait_for_idle()
                target_agent = self._owning_activity.agent
            else:
                target_activity = await session.wait_for_idle()
                target_agent = target_activity.agent
        except ActivityClosedError:
            logger.debug("dropping tool reply — owning activity closed")
            self._pending_updates.clear()
            return

        # no await after this line

        updates = self._pending_updates[:]
        self._pending_updates.clear()

        pending_items: list[ChatItem] = []
        for update in updates:
            pending_items.extend(update.items)

        if not pending_items:
            return

        # only insert again if delivery target differs (session-scoped handoff)
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN
        items_to_insert = [
            item for u in updates for item in u.items if u.target is not target_agent
        ]
        if items_to_insert:
            logger.warning(
                "agent handoff happened while tool waiting for reply delivering",
                extra={
                    "tools": [
                        u.ctx.function_call.name for u in updates if u.target is not target_agent
                    ],
                },
            )
            chat_ctx = target_agent.chat_ctx.copy()
            chat_ctx.insert(items_to_insert)

        # if the update is still the tail, the agent hasn't spoken since — summarize
        # directly; otherwise let the LLM decide whether it already covered this
        at_tail = (items := target_agent.chat_ctx.items) and items[-1].id == pending_items[-1].id
        template = (
            self._tool_options["reply_at_tail_template"]
            if at_tail
            else self._tool_options["reply_maybe_covered_template"]
        )

        call_ids = [item.call_id for item in pending_items if item.type == "function_call_output"]
        speech = session.generate_reply(
            instructions=_render(template, {"call_ids": call_ids}),
            tool_choice="none",
            chat_ctx=chat_ctx,
        )
        session.emit(
            "tool_execution_updated",
            ToolExecutionUpdatedEvent(
                update=ToolReplyUpdated(
                    update_ids=call_ids, status="scheduled", speech_id=speech.id
                )
            ),
        )
        logger.debug(
            "generate async tool reply",
            extra={
                "speech_id": speech.id,
                "items": [
                    (item.name, item.call_id)
                    for item in pending_items
                    if item.type == "function_call_output"
                ],
                "updates_at_tail": at_tail,
            },
        )

        def _on_speech_done(speech: SpeechHandle) -> None:
            reply_status: Literal["completed", "interrupted", "skipped"]
            if speech.interrupted:
                reply_status = "interrupted"
            elif not speech.chat_items:
                # the LLM judged the content already covered and produced no output
                reply_status = "skipped"
            else:
                reply_status = "completed"

            if not speech.chat_items:
                logger.debug(
                    "async tool reply was done without outputs",
                    extra={"speech_id": speech.id, "interrupted": speech.interrupted},
                )
                # TODO(long): reschedule interrupted replies?

            session.emit(
                "tool_execution_updated",
                ToolExecutionUpdatedEvent(
                    update=ToolReplyUpdated(
                        update_ids=call_ids, status=reply_status, speech_id=speech.id
                    )
                ),
            )

        speech.add_done_callback(_on_speech_done)

    async def _check_duplicate(
        self,
        fnc_name: str,
        *,
        on_duplicate: DuplicateMode,
        confirm_duplicate: bool | None,
    ) -> str | None:
        if on_duplicate == "allow":
            return None

        async with self._duplicate_check_lock:
            running_fnc_calls = [
                t.ctx.function_call
                for t in self._running_tasks.values()
                if t.ctx.function_call.name == fnc_name
            ]
            if len(running_fnc_calls) == 0:
                return None

            if on_duplicate == "replace":
                # replace must honor each in-flight task's allow_cancellation flag
                non_cancellable = [
                    fnc_call
                    for fnc_call in running_fnc_calls
                    if not self._running_tasks[fnc_call.call_id].allow_cancellation
                ]
                if non_cancellable:
                    raise ToolError(
                        f"cannot replace duplicate call of `{fnc_name}`: "
                        f"running call is not cancellable (allow_cancellation=False)"
                    )

                results = await asyncio.gather(
                    *[self.cancel(fnc_call.call_id) for fnc_call in running_fnc_calls],
                    return_exceptions=True,
                )
                exceptions = [result for result in results if isinstance(result, Exception)]
                if exceptions:
                    error_messages = "\n".join([str(e) for e in exceptions])
                    raise ToolError(f"Failed to cancel duplicate tool calls: {error_messages}")
                return None

            fnc_calls_json = [fnc_call.model_dump_json() for fnc_call in running_fnc_calls]
            args: DuplicatePromptArgs = {
                "function_name": fnc_name,
                "fnc_calls_json": fnc_calls_json,
                "fnc_calls_text": "\n".join(fnc_calls_json),
            }
            if on_duplicate == "reject":
                return _render(self._tool_options["duplicate_reject_template"], dict(args))

            if on_duplicate == "confirm" and not confirm_duplicate:
                return _render(self._tool_options["duplicate_confirm_template"], dict(args))

        return None


def _build_executor_map(
    *,
    toolsets: Sequence[Toolset],
    default: _ToolExecutor,
) -> dict[str, _ToolExecutor]:
    """Map each tool to its owning executor: AsyncToolset tools route to that
    toolset's own executor; everything else falls back to ``default``."""
    from ..llm.async_toolset import AsyncToolset

    mapping: dict[str, _ToolExecutor] = {}

    def walk(ts: Toolset, current: _ToolExecutor) -> None:
        if isinstance(ts, AsyncToolset):
            current = ts._executor
        for child in ts.tools:
            if isinstance(child, (FunctionTool, RawFunctionTool)):
                mapping[child.info.name] = current
            elif isinstance(child, Toolset):
                walk(child, current)

    for ts in toolsets:
        walk(ts, default)
    return mapping
