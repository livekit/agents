from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from typing_extensions import TypedDict

from .. import utils
from ..job import JobContext, get_job_context
from ..llm.chat_context import ChatContext, ChatItem
from ..llm.tool_context import (
    CONFIRM_DUPLICATE_PARAM,
    DuplicateMode,
    FunctionTool,
    RawFunctionTool,
    Tool,
    ToolError,
    Toolset,
    function_tool,
)
from ..llm.utils import prepare_function_arguments
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from .events import RunContext

if TYPE_CHECKING:
    from .agent import Agent
    from .agent_activity import AgentActivity
    from .agent_session import AgentSession
    from .speech_handle import SpeechHandle


UPDATE_TEMPLATE = """The tool `{function_name}` has updated, message: {message}
The task is still running, so DON'T make up or give information not included in the message above."""

DUPLICATE_REJECT = """Same tool `{function_name}` is already running:
{running_fnc_calls}
If you want to cancel the existing one, call `cancel_task` with call_id."""

DUPLICATE_CONFIRM = """Same tool `{function_name}` is already running:
{running_fnc_calls}
Re-call with confirm duplicate True to run a duplicate if needed,
or if you want to cancel the existing one, call `cancel_task` with call_id."""

# used when the pending update is the most recent item in chat_ctx — the agent
# can't have already talked about it.
REPLY_INSTRUCTIONS_AT_TAIL = """New results arrived from background tool calls (call_ids: {pending_call_ids}).
Summarize the results naturally. Do NOT repeat information you have already told the user."""

# used when newer items have been appended after the pending update — the agent
# may have already verbalized the result in its most recent turn.
REPLY_INSTRUCTIONS_MAYBE_COVERED = """New results arrived from background tool calls (call_ids: {pending_call_ids}).
Review your most recent assistant messages.
- If your previous messages already conveyed these results to the user, return an empty response (no text at all).
- Otherwise, summarize the results naturally with a transition.
Do NOT repeat information you have already told the user."""


class AsyncToolPrompts(TypedDict, total=False):
    """System-message templates injected around tool dispatch.

    Each field is a ``str.format()`` template; unmentioned keys keep their defaults.
    Set on ``AgentSession``, ``Agent``, or ``AsyncToolset`` (most specific wins).

    Placeholders:

    - ``update``: ``{function_name}`` ``{call_id}`` ``{message}``
    - ``duplicate_reject`` / ``duplicate_confirm``: ``{function_name}`` ``{running_fnc_calls}``
    - ``reply_at_tail`` / ``reply_maybe_covered``: ``{pending_call_ids}``
    """

    update: str
    """Wraps a user-provided ``ctx.update(message)`` string before it lands in chat_ctx."""
    duplicate_reject: str
    """Sent to the LLM when ``on_duplicate='reject'`` blocks a duplicate call."""
    duplicate_confirm: str
    """Sent to the LLM when ``on_duplicate='confirm'`` requires re-call with confirmation."""
    reply_at_tail: str
    """Instruction for the deferred reply when the pending update is the tail of chat_ctx."""
    reply_maybe_covered: str
    """Instruction for the deferred reply when newer items came after the pending update."""


_ASYNC_TOOL_PROMPTS_DEFAULTS: AsyncToolPrompts = {
    "update": UPDATE_TEMPLATE,
    "duplicate_reject": DUPLICATE_REJECT,
    "duplicate_confirm": DUPLICATE_CONFIRM,
    "reply_at_tail": REPLY_INSTRUCTIONS_AT_TAIL,
    "reply_maybe_covered": REPLY_INSTRUCTIONS_MAYBE_COVERED,
}


def _resolve_async_tool_prompts(
    config: AsyncToolPrompts | None = None,
) -> AsyncToolPrompts:
    """Return a fully-populated ``AsyncToolPrompts`` with defaults filled in for absent keys."""
    if config is None:
        return AsyncToolPrompts(**_ASYNC_TOOL_PROMPTS_DEFAULTS)
    return AsyncToolPrompts(**{**_ASYNC_TOOL_PROMPTS_DEFAULTS, **config})


# shared across all executors so the cancel_task / get_running_tasks companion tools
# see a single job-scoped view
_RunningTasks: dict[tuple[JobContext | None, str], _RunningTask] = {}


@function_tool
async def get_running_tasks() -> list[dict]:
    """Get the list of running tool calls that are cancellable."""
    job_ctx = get_job_context(required=False)
    return [
        task.ctx.function_call.model_dump()
        for (ctx, _), task in list(_RunningTasks.items())
        if ctx is job_ctx and task.allow_cancellation
    ]


@function_tool
async def cancel_task(call_id: str) -> str:
    """Cancel a running tool call by call_id."""
    job_ctx = get_job_context(required=False)
    task = _RunningTasks.get((job_ctx, call_id))
    if task is None or not task.allow_cancellation:
        return f"Task {call_id} not found or not cancellable."
    if await task.executor.cancel(call_id):
        return f"Task {call_id} cancelled successfully."
    return f"Task {call_id} not found or already completed."


def has_cancellable_tool(tools: Sequence[Tool | Toolset]) -> bool:
    """Return True if any tool (or nested toolset tool) sets ``allow_cancellation=True``."""
    for tool in tools:
        if isinstance(tool, (FunctionTool, RawFunctionTool)):
            if tool.info.allow_cancellation:
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
        async_tool_prompts: AsyncToolPrompts | None = None,
    ) -> None:
        self._running_tasks: dict[str, _RunningTask] = {}
        self._duplicate_check_lock = asyncio.Lock()

        self._pending_updates: list[_PendingUpdate] = []
        self._reply_task: asyncio.Task[None] | None = None

        self._owning_activity: AgentActivity | None = owning_activity
        self._tool_prompts: AsyncToolPrompts = _resolve_async_tool_prompts(async_tool_prompts)

    def set_owning_activity(self, activity: AgentActivity | None) -> None:
        self._owning_activity = activity

    def set_tool_prompts(self, prompts: AsyncToolPrompts) -> None:
        """Replace the prompt templates. Caller must pre-resolve defaults."""
        self._tool_prompts = prompts

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
    ) -> Any:
        """Run ``tool``. Returns when the first ``ctx.update()`` lands or the tool returns."""
        call_id = run_ctx.function_call.call_id
        fnc_name = run_ctx.function_call.name
        info = tool.info
        on_duplicate: DuplicateMode = info.on_duplicate
        allow_cancellation: bool = info.allow_cancellation

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
                return
            except Exception as e:
                output = e
                logger.exception(
                    "error in tool execution",
                    extra={"call_id": call_id, "function": fnc_name},
                )

            if not first_update_fut.done():
                # tool returned without ctx.update() — surface the result to dispatch
                if isinstance(output, BaseException):
                    first_update_fut.set_exception(output)
                else:
                    first_update_fut.set_result(output)
                return

            if output is None:
                return

            # the first update has already been returned to dispatch, so an Agent
            # return now has no surface to carry an agent_task back
            from .agent import Agent

            if isinstance(output, Agent):
                raise ToolError(
                    f"tool `{fnc_name}` returned an Agent after ctx.update(); "
                    "agent handoff after a progress update is not supported"
                )

            # final return goes through the coalescer as a synthetic output
            pair = run_ctx._make_update_pair(output, call_id_suffix="_final")
            run_ctx._updates.append(pair)
            await self._enqueue_reply(run_ctx, [pair[0], pair[1]])

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

        task_key = (get_job_context(required=False), call_id)
        _RunningTasks[task_key] = running_task

        def _on_done(_: asyncio.Task[Any]) -> None:
            self._running_tasks.pop(call_id, None)
            _RunningTasks.pop(task_key, None)
            # detach so a stashed RunContext can't drive the executor post-completion
            run_ctx._detach_executor()

        exe_task.add_done_callback(_on_done)

        return await first_update_fut

    async def cancel(self, call_id: str) -> bool:
        task = self._running_tasks.get(call_id)
        if task is None:
            return False
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
            self._tool_prompts["reply_at_tail"]
            if at_tail
            else self._tool_prompts["reply_maybe_covered"]
        )

        call_ids = [item.call_id for item in pending_items if item.type == "function_call_output"]
        speech = session.generate_reply(
            instructions=template.format(pending_call_ids=call_ids),
            tool_choice="none",
            chat_ctx=chat_ctx,
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
            if not speech.chat_items:
                logger.debug(
                    "async tool reply was done without outputs",
                    extra={"speech_id": speech.id, "interrupted": speech.interrupted},
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

            if on_duplicate == "reject":
                return self._tool_prompts["duplicate_reject"].format(
                    function_name=fnc_name,
                    running_fnc_calls="\n".join(
                        [fnc_call.model_dump_json() for fnc_call in running_fnc_calls]
                    ),
                )

            if on_duplicate == "confirm" and not confirm_duplicate:
                return self._tool_prompts["duplicate_confirm"].format(
                    function_name=fnc_name,
                    running_fnc_calls="\n".join(
                        [fnc_call.model_dump_json() for fnc_call in running_fnc_calls]
                    ),
                )

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
