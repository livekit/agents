from __future__ import annotations

import asyncio
import bisect
import inspect
import itertools
import json
import weakref
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol

from typing_extensions import TypedDict

from .. import utils
from ..llm.chat_context import ChatContext, ChatItem, FunctionCall, FunctionCallOutput
from ..llm.tool_context import (
    CONFIRM_DUPLICATE_PARAM,
    DELIVERED,
    DuplicateMode,
    DuplicateScope,
    FunctionTool,
    InputRequired,
    ModelRetry,
    RawFunctionTool,
    StopResponse,
    Tool,
    ToolChoice,
    ToolError,
    ToolFlag,
    ToolResult,
    Toolset,
    approval_granted,
    approval_scope,
    delivered,
    function_tool,
    is_transition,
)
from ..llm.utils import prepare_function_arguments, validated_arguments
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from .events import (
    RunContext,
    ToolCallEnded,
    ToolCallStarted,
    ToolCallUpdated,
    ToolExecutionUpdatedEvent,
    ToolReplyUpdated,
)

if TYPE_CHECKING:
    from .agent_activity import AgentActivity
    from .agent_session import LiveSession
    from .speech_handle import SpeechHandle


UPDATE_TEMPLATE = """The tool `{function_name}` has updated, message: {message}
The task is still running, so DON'T make up or give information not included in the message above."""

DUPLICATE_REJECT = """Same tool `{function_name}` is already running:
{fnc_calls_text}
Do not call it again; its result comes when it ends."""

DUPLICATE_CONFIRM = """Same tool `{function_name}` is already running:
{fnc_calls_text}
Re-call with confirm duplicate True to run a duplicate if needed,
or if you want to cancel the existing one, call `lk_agents_cancel_task` with call_id.
Only run duplicate or cancel the existing one when user explicitly requests it."""

DUPLICATE_NOT_CANCELLABLE = (
    "cannot replace duplicate call of `{function_name}`: "
    "running call is not cancellable (allow_cancellation=False)"
)
"""The error a ``replace`` duplicate gets while a running call of the tool is not cancellable."""

# used when the pending update is the most recent item in chat_ctx — the agent
# can't have already talked about it.
REPLY_INSTRUCTIONS_AT_TAIL = """New results arrived from background tool calls (call_ids: {call_ids}).
Tell each call's newest result naturally. Do NOT repeat what you already told the user."""

# used when newer items have been appended after the pending update — the agent
# may have already verbalized the result in its most recent turn.
REPLY_INSTRUCTIONS_MAYBE_COVERED = """New results arrived from background tool calls (call_ids: {call_ids}).
You may have already mentioned them in your most recent replies.
If you already told the user everything in these results, reply with an empty response (no text at all).
Otherwise, tell what each call's newest result adds, with a natural transition.
Never repeat information you have already told the user."""


def tool_replied(ctx: RunContext, reply: Sequence[ChatItem] = ()) -> None:
    """What ``reply`` said is the reply to a tool its turn let run in the background."""
    fut = ctx.speech_handle._background_tool_replies.get(ctx.function_call.call_id)
    if fut is not None and not fut.done():
        fut.set_result([i for i in reply if i.type == "message" and i.role == "assistant"])


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

        LiveSession(
            tool_handling={
                "async_options": {"update_template": "..."},
            },
        )

    Set on ``LiveSession``, ``Agent``, or ``AsyncToolset`` (most specific wins).
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
_RunningTasks: weakref.WeakKeyDictionary[LiveSession, dict[str, _RunningTask]] = (
    weakref.WeakKeyDictionary()
)


@function_tool(name="lk_agents_get_running_tasks")
async def get_running_tasks(ctx: RunContext) -> list[dict]:
    """Get the list of running tool calls that are cancellable."""
    if ctx._run is not None:
        tasks = ctx._run._tools.executor._running_tasks
    else:
        tasks = _RunningTasks.get(ctx.session, {})
    return [
        task.ctx.function_call.model_dump() for task in tasks.values() if task.allow_cancellation
    ]


@function_tool(name="lk_agents_cancel_task")
async def cancel_task(ctx: RunContext, call_id: str) -> str:
    """Cancel a running tool call by call_id."""
    if ctx._run is not None:
        executor = ctx._run._tools.executor
    else:
        task = _RunningTasks.get(ctx.session, {}).get(call_id)
        if task is None:
            raise ToolError(f"Task {call_id} not found")
        executor = task.executor

    if not await executor.cancel(call_id):
        raise ToolError(f"Task {call_id} not found or already completed")
    return f"Task {call_id} cancelled successfully."


def _reply_instructions(options: AsyncToolOptions, *, at_tail: bool, call_ids: list[str]) -> str:
    """What the reply to tool progress is told: progress at the tail of the history is
    summarized, else the model judges whether it said it already."""
    template = (
        options["reply_at_tail_template"] if at_tail else options["reply_maybe_covered_template"]
    )
    return _render(template, {"call_ids": call_ids})


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


def _canonical_args(raw_arguments: dict[str, Any]) -> str:
    """Stable JSON for argument comparison.

    ``sort_keys`` recurses into nested objects, so a provider's key-emission order
    can't make two identical calls look different. Lists stay order-sensitive —
    ``["a", "b"]`` and ``["b", "a"]`` are genuinely different arguments.
    """
    return json.dumps(raw_arguments, sort_keys=True, default=str)


def _duplicate_key(
    *,
    fnc: FunctionTool | RawFunctionTool,
    fnc_name: str,
    scope: DuplicateScope,
    raw_arguments: dict[str, Any],
) -> tuple[str, str | None]:
    """The identity a call is deduplicated on. Always name-scoped, so two tools
    can never collide; under ``"name"`` equality reduces to name equality.

    ``"name_and_args"`` compares *validated* arguments, so an optional parameter the
    LLM omitted on one call and passed explicitly on the next still reads as the same
    call (see :func:`validated_arguments`).
    """
    if scope != "name_and_args":
        return (fnc_name, None)

    # the confirm flag is harness state, not an argument
    raw = {k: v for k, v in raw_arguments.items() if k != CONFIRM_DUPLICATE_PARAM}

    try:
        # encoding is in the try too: validated values are Python objects, so a type like
        # dict[SomeEnum, int] yields keys json can't encode
        return (fnc_name, _canonical_args(validated_arguments(fnc, raw)))
    except Exception:
        # the call raises ToolError on its own later; key on what was sent, which always
        # encodes, rather than failing the guard here
        return (fnc_name, _canonical_args(raw))


@dataclass
class _RunningTask:
    ctx: RunContext
    exe_task: asyncio.Task[Any]
    executor: _ToolExecutor
    allow_cancellation: bool
    duplicate_key: tuple[str, str | None] | None  # None when the tool is on_duplicate="allow"


@dataclass
class _PendingUpdate:
    ctx: RunContext
    items: list[ChatItem]
    target: AgentActivity  # activity whose history received the eager insert
    seq: int  # call order across the executor's tools
    say: str | None = None  # exact text spoken as written, on its own
    tool_choice: ToolChoice | None = None
    """What the reply may call: an update's request, or None for a result to act on."""
    result: bool = False  # the call's end


def _reply_tool_choice(requests: Iterable[ToolChoice | None]) -> ToolChoice | None:
    """The tool choice of one reply step answering several tool outputs: a single shared request,
    else unconstrained, as one output that wants the tools is enough to need them."""
    distinct: list[ToolChoice | None] = []
    for request in requests:
        # a named choice is a dict, so this compares rather than hashes
        if request not in distinct:
            distinct.append(request)
    return distinct[0] if len(distinct) == 1 else None


class ToolSink(Protocol):
    """Where an executor's tools report: a call's activity, or a text door's queue."""

    def started(self, task: _RunningTask) -> None: ...

    def done(self, task: _RunningTask, output: Any) -> None:
        """The tool's task ended with ``output``, its result or what it raised."""

    async def stop(self, task: _RunningTask) -> None: ...

    async def arguments(
        self, tool: FunctionTool | RawFunctionTool, raw: dict[str, Any], ctx: RunContext
    ) -> dict[str, Any]: ...

    def returned(self, ctx: RunContext, output: Any) -> Any:
        """What the tool's output is to the turn that called it, when it ends before an update."""

    async def update(
        self,
        ctx: RunContext,
        pair: tuple[FunctionCall, FunctionCallOutput],
        text: str,
        message: Any,
        *,
        silent: bool,
    ) -> None: ...

    async def say(self, ctx: RunContext, text: str) -> None: ...

    async def ended(self, ctx: RunContext, output: Any) -> None:
        """The tool ended after its turn went on without it."""


class _ToolExecutor:
    """Lifecycle manager for in-flight tool calls; what they produce goes to its ``ToolSink``.

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
        sink: ToolSink | None = None,
    ) -> None:
        self._running_tasks: dict[str, _RunningTask] = {}
        self._duplicate_check_lock = asyncio.Lock()
        self._tool_options: AsyncToolOptions = _resolve_async_tool_options(async_tool_options)
        self._sink: ToolSink = sink if sink is not None else _CallSink(self, owning_activity)

    def _report(self, ctx: RunContext, message: Any, *, silent: bool) -> None:
        """The tool's first update lets its turn go on, with ``message`` as its output."""
        assert ctx._first_update_fut is not None
        ctx._suppress_reply = silent
        # a report is not a result: a model free to call tools can answer "still searching" by
        # calling the same tool again
        ctx._reply_tool_choice = "none"
        ctx._first_update_fut.set_result(message)

    def set_owning_activity(self, activity: AgentActivity | None) -> None:
        assert isinstance(self._sink, _CallSink)
        self._sink._owning_activity = activity

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
    ) -> Any:
        """Run ``tool``. Returns when the first ``ctx.update()`` lands or the tool returns."""
        call_id = run_ctx.function_call.call_id
        fnc_name = run_ctx.function_call.name
        info = tool.info
        on_duplicate: DuplicateMode = info.on_duplicate
        allow_cancellation: bool = ToolFlag.CANCELLABLE in info.flags

        confirm_duplicate: bool | None = None
        if on_duplicate == "confirm":
            confirm_duplicate = bool(raw_arguments.pop(CONFIRM_DUPLICATE_PARAM, False))

        # nothing can match an "allow" tool's key: keys are name-scoped, the mode is per tool
        dup_key: tuple[str, str | None] | None = None
        if on_duplicate != "allow":
            # derived AFTER the pop, so a confirming re-call keys like the call it confirms
            dup_key = _duplicate_key(
                fnc=tool,
                fnc_name=fnc_name,
                scope=info.duplicate_scope,
                raw_arguments=raw_arguments,
            )

            duplicate_result = await self._check_duplicate(
                dup_key, on_duplicate=on_duplicate, confirm_duplicate=confirm_duplicate
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
        sink = self._sink

        # run the tool and return its raw output (or the caught exception); _on_done
        # derives the call's single terminal entry from how the task ended
        async def _execute_tool() -> Any:
            try:
                arguments = await sink.arguments(tool, raw_arguments, run_ctx)
                fnc_args, fnc_kwargs = prepare_function_arguments(
                    fnc=tool, json_arguments=arguments, call_ctx=run_ctx
                )
                granted = isinstance(tool, FunctionTool) and tool.info.requires_approval
                if granted:
                    history = (
                        run_ctx._run._chat_ctx.items
                        if run_ctx._run is not None
                        else run_ctx.session.history.items
                    )
                    granted = approval_granted(history, run_ctx.function_call)
                if mock is not None:
                    from .run_result import _run_mock

                    output = await _run_mock(mock, *fnc_args, **fnc_kwargs)
                else:
                    with approval_scope(run_ctx.function_call, granted):
                        output = tool(*fnc_args, **fnc_kwargs)
                        if inspect.isawaitable(output):
                            output = await output
            except asyncio.CancelledError:
                logger.debug("tool cancelled", extra={"call_id": call_id, "function": fnc_name})
                if not first_update_fut.done():
                    first_update_fut.set_result(None)
                raise  # _on_done emits the cancelled terminal
            except ModelRetry as e:
                output = ToolError(e.message)  # as in text: the model reads why, and retries
            except Exception as e:
                output = e

            if not first_update_fut.done():
                # tool returned without ctx.update() — surface the result to dispatch
                output = sink.returned(run_ctx, output)
                if isinstance(output, BaseException):
                    first_update_fut.set_exception(output)
                else:
                    first_update_fut.set_result(output)
                return output

            # the first update has already been returned to dispatch, so an Agent
            # return now has no surface to carry an agent_task back
            from .agent import Agent, TaskAwaited

            if isinstance(output, (Agent, TaskAwaited)):
                logger.error(
                    f"tool `{fnc_name}` returned an Agent after ctx.update(); "
                    "agent handoff after a progress update is not supported",
                    extra={"call_id": call_id, "function": fnc_name},
                )
                raise RuntimeError("agent handoff after a progress update is not supported")

            if isinstance(output, BaseException) and not isinstance(output, StopResponse):
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

            await sink.ended(run_ctx, output)
            return output

        exe_task = asyncio.create_task(_execute_tool(), name=f"tool_exec_{fnc_name}")
        from .agent import _pass_through_activity_task_info

        _pass_through_activity_task_info(exe_task)
        run_ctx._work = exe_task

        running_task = _RunningTask(
            ctx=run_ctx,
            exe_task=exe_task,
            executor=self,
            allow_cancellation=allow_cancellation,
            duplicate_key=dup_key,
        )
        self._running_tasks[call_id] = running_task
        sink.started(running_task)

        def _on_done(task: asyncio.Task[Any]) -> None:
            self._running_tasks.pop(call_id, None)
            # detach so a stashed RunContext can't drive the executor post-completion
            run_ctx._detach_executor()

            # how the task ended: a returned value, a raised exception, or cancellation
            try:
                output = task.result()
            except BaseException as e:
                output = e

            if not first_update_fut.done():
                first_update_fut.set_result(None)  # cancelled before the first update
            sink.done(running_task, output)

        exe_task.add_done_callback(_on_done)

        return await first_update_fut

    async def cancel(self, call_id: str) -> bool:
        task = self._running_tasks.get(call_id)
        if task is None:
            return False

        if not task.allow_cancellation:
            raise ToolError(f"Tool call {call_id} is not cancellable")

        await self._sink.stop(task)
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
        tasks = [task.exe_task for task in self._running_tasks.values()]
        if isinstance(self._sink, _CallSink):
            for update in self._sink._pending_updates:
                if update.result:
                    tool_replied(update.ctx)
            self._sink._pending_updates.clear()
            if self._sink._reply_task is not None:
                tasks.append(self._sink._reply_task)
        if tasks:
            await utils.aio.cancel_and_wait(*tasks)
        self._running_tasks.clear()

    async def drain(self) -> None:
        """Cancel cancellable tools, await the rest. Reply delivery is left running;
        ``_deliver_reply`` drops itself when its target activity closes."""
        await self.cancel_all(cancellable_only=True)

    async def _check_duplicate(
        self,
        dup_key: tuple[str, str | None],
        *,
        on_duplicate: DuplicateMode,
        confirm_duplicate: bool | None,
    ) -> str | None:
        if on_duplicate == "allow":
            return None

        fnc_name = dup_key[0]

        async with self._duplicate_check_lock:
            running_fnc_calls = [
                t.ctx.function_call
                for t in self._running_tasks.values()
                if t.duplicate_key == dup_key
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
                    raise ToolError(DUPLICATE_NOT_CANCELLABLE.format(function_name=fnc_name))

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


class _CallSink:
    """A call's side of an executor: the session's running tasks and events, speech, and the
    reply to progress at the next idle."""

    def __init__(self, executor: _ToolExecutor, owning_activity: AgentActivity | None) -> None:
        self._executor = executor
        self._owning_activity = owning_activity
        self._pending_updates: list[_PendingUpdate] = []
        self._reply_task: asyncio.Task[None] | None = None
        self._seq = itertools.count()

    def started(self, task: _RunningTask) -> None:
        session = task.ctx.session
        _RunningTasks.setdefault(session, {})[task.ctx.function_call.call_id] = task
        session._tool_execution_updated(
            ToolExecutionUpdatedEvent(update=ToolCallStarted(function_call=task.ctx.function_call)),
        )

    def done(self, task: _RunningTask, output: Any) -> None:
        run_ctx = task.ctx
        call_id = run_ctx.function_call.call_id
        session = run_ctx.session
        if (session_tasks := _RunningTasks.get(session)) is not None:
            session_tasks.pop(call_id, None)

        # one terminal entry per call; deferred entries use the _final id
        from .agent import Agent

        if isinstance(output, ToolResult):
            output = output.output

        status: Literal["done", "error", "cancelled"]
        message: str | None
        if task.exe_task.cancelled() or isinstance(output, asyncio.CancelledError):
            status, message = "cancelled", None
            tool_replied(run_ctx)
        elif isinstance(output, BaseException) and not isinstance(output, StopResponse):
            status, message = "error", str(output)
        elif output is None or isinstance(output, (StopResponse, Agent)):
            status, message = "done", None
        else:
            status, message = "done", str(output)

        entry_id = call_id + "_final" if run_ctx._updates else call_id
        session._tool_execution_updated(
            ToolExecutionUpdatedEvent(
                update=ToolCallEnded(id=entry_id, call_id=call_id, message=message, status=status)
            ),
        )

    async def stop(self, task: _RunningTask) -> None:
        if not task.ctx.speech_handle.allow_interruptions:
            raise ToolError(
                f"Tool call {task.ctx.function_call.call_id} is not cancellable because "
                "interruptions are disallowed"
            )
        await utils.aio.cancel_and_wait(task.exe_task)

    async def arguments(
        self, tool: FunctionTool | RawFunctionTool, raw: dict[str, Any], ctx: RunContext
    ) -> dict[str, Any]:
        if (activity := self._owning_activity) is None:
            return raw
        # a call that answers a claimed question carries the person's words, and the history
        # shows the call as it ran
        arguments = await activity._answer_arguments(tool, raw, ctx)
        if arguments is not raw:
            ctx.function_call.arguments = json.dumps(arguments)
        return arguments

    def returned(self, ctx: RunContext, output: Any) -> Any:
        if isinstance(output, ToolResult) and output.verbatim:
            # the person hears it as written now; the model reads that it was said
            self._speaker(ctx)._say_exactly(str(output.output))
            return delivered(output)
        return output

    async def update(
        self,
        ctx: RunContext,
        pair: tuple[FunctionCall, FunctionCallOutput],
        text: str,
        message: Any,
        *,
        silent: bool,
    ) -> None:
        ctx.session._tool_execution_updated(
            ToolExecutionUpdatedEvent(
                update=ToolCallUpdated(
                    id=pair[0].call_id,
                    call_id=ctx.function_call.call_id,
                    message=text,
                    silent=silent,
                )
            ),
        )

        assert ctx._first_update_fut is not None
        if not ctx._first_update_fut.done():
            self._executor._report(ctx, message, silent=silent)
            ctx.function_call.extra["__livekit_agents_tool_non_blocking"] = True
            ctx.speech_handle._background_tool_replies[ctx.function_call.call_id] = asyncio.Future()
            return

        await self._enqueue_reply(ctx, [pair[0], pair[1]], tool_choice="none")

    async def say(self, ctx: RunContext, text: str) -> None:
        update_step = len(ctx._updates)
        pair = ctx._make_update_pair(
            ToolResult(DELIVERED.format(text=text), reply_required=False),
            call_id_suffix=f"_update_{update_step}" if update_step > 0 else "",
        )
        ctx._updates.append(pair)
        ctx.session._tool_execution_updated(
            ToolExecutionUpdatedEvent(
                update=ToolCallUpdated(
                    id=pair[0].call_id, call_id=ctx.function_call.call_id, message=text
                )
            ),
        )

        assert ctx._first_update_fut is not None
        if not ctx._first_update_fut.done():
            self._speaker(ctx)._say_exactly(text)
            ctx._first_update_fut.set_result(ToolResult(pair[1].output, reply_required=False))
            ctx.function_call.extra["__livekit_agents_tool_non_blocking"] = True
            ctx.speech_handle._background_tool_replies[ctx.function_call.call_id] = asyncio.Future()
            return

        await self._enqueue_reply(ctx, [pair[0], pair[1]], say=text)

    async def ended(self, ctx: RunContext, output: Any) -> None:
        if output is None or isinstance(output, StopResponse):
            tool_replied(ctx)
            return

        said: str | None = None
        if isinstance(output, ToolResult) and output.verbatim:
            # the person hears it as written in call order after the tool's updates
            said = str(output.output)
            output = delivered(output)

        # final return goes through the coalescer as a synthetic output
        pair = ctx._make_update_pair(output, call_id_suffix="_final")
        ctx._updates.append(pair)
        if is_transition(output) and (activity := self._owning_activity) is not None:
            # the activity acts on it as it does when a tool returns without updates:
            # an InputRequired speaks the question itself, so no reply is generated
            pair[1].reply_required = not isinstance(output, InputRequired)
            await self._enqueue_reply(ctx, [pair[0], pair[1]], result=True)
            activity._apply_transition(output, fnc_call_out=pair[1])
            return
        await self._enqueue_reply(ctx, [pair[0], pair[1]], say=said, result=True)

    def _speaker(self, ctx: RunContext) -> AgentActivity:
        """The activity that speaks for this executor's tools now."""
        if self._owning_activity is not None:
            return self._owning_activity
        return ctx.session._current_activity_or_raise()

    async def _enqueue_reply(
        self,
        ctx: RunContext,
        items: list[ChatItem],
        *,
        say: str | None = None,
        tool_choice: ToolChoice | None = None,
        result: bool = False,
    ) -> None:
        """Queue a tool's progress or ``result`` for the next idle, in call order: ``say`` is
        spoken as written, the items' outputs that ask for a reply are worded by the model,
        which may call what ``tool_choice`` allows (its default for a result)."""
        # eager insert so a reply firing before delivery sees the items
        target = self._speaker(ctx)
        chat_ctx = target._chat_ctx.copy()
        chat_ctx.insert(items)
        await target.update_chat_ctx(chat_ctx)
        ctx.session.history.insert(items)

        if say is None and not _asks_reply(items):
            if result:
                tool_replied(ctx)
            return

        self._pending_updates.append(
            _PendingUpdate(
                ctx=ctx,
                items=items,
                target=target,
                seq=next(self._seq),
                say=say,
                tool_choice=tool_choice,
                result=result,
            )
        )
        # the turn that let the call run in the background makes no reply to its older state
        ctx.speech_handle._superseded_tool_replies.add(ctx.function_call.call_id)
        self._schedule_delivery(ctx.session)

    def _schedule_delivery(self, session: LiveSession) -> None:
        if self._reply_task is None or self._reply_task.done():
            self._reply_task = asyncio.create_task(
                self._deliver_reply(session), name="tool_executor_deliver_reply"
            )
            # let an active RunResult wait for the deferred reply to land
            run_state = session._global_run_state
            if run_state is not None:
                run_state._watch_handle(self._reply_task)

    async def _deliver_reply(self, session: LiveSession) -> None:
        from .agent_activity import ActivityClosedError

        target_activity: AgentActivity
        try:
            while True:
                # an update waits for the turn that let its call run in the background, that
                # turn's tool reply included, so it is never told before or over it
                for update in self._pending_updates[:]:
                    await update.ctx.speech_handle
                if self._owning_activity is not None:
                    await self._owning_activity.wait_for_idle()
                    target_activity = self._owning_activity
                else:
                    target_activity = await session.wait_for_idle()
                if not target_activity._turn_after_question.is_set():
                    # a question in front of the person holds the floor until their next turn
                    await target_activity._turn_after_question.wait()
                    continue
                if all(u.ctx.speech_handle.done() for u in self._pending_updates):
                    break
        except ActivityClosedError:
            logger.debug("dropping tool reply — owning activity closed")
            for update in self._pending_updates:
                if update.result:
                    tool_replied(update.ctx)
            self._pending_updates.clear()
            return

        # no await after this line: the speeches of one delivery are scheduled back to back

        updates = self._pending_updates[:]
        self._pending_updates.clear()

        # consecutive updates are one reply; a say is spoken on its own, and no reply merges
        # updates from both sides of it
        group: list[_PendingUpdate] = []
        for update in updates:
            if update.say is None:
                group.append(update)
                continue
            if group:
                self._reply(session, target_activity, group)
                group = []
            speech = target_activity._say_exactly(update.say)

            def _on_say_done(speech: SpeechHandle, update: _PendingUpdate = update) -> None:
                if speech.interrupted and not speech.chat_items:
                    # cut off before any of it played: said at the next idle, in call order
                    bisect.insort(self._pending_updates, update, key=lambda u: u.seq)
                    self._schedule_delivery(session)
                elif update.result:
                    tool_replied(update.ctx, speech.chat_items)

            speech.add_done_callback(_on_say_done)
            if _asks_reply(update.items):
                group.append(update)
        if group:
            self._reply(session, target_activity, group)

    def _reply(
        self, session: LiveSession, target_activity: AgentActivity, updates: list[_PendingUpdate]
    ) -> None:
        pending_items: list[ChatItem] = []
        for update in updates:
            pending_items.extend(update.items)

        if not pending_items:
            return

        # only insert again if delivery target differs (session-scoped handoff)
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN
        items_to_insert = [
            item for u in updates for item in u.items if u.target is not target_activity
        ]
        if items_to_insert:
            logger.warning(
                "agent handoff happened while tool waiting for reply delivering",
                extra={
                    "tools": [
                        u.ctx.function_call.name for u in updates if u.target is not target_activity
                    ],
                },
            )
            chat_ctx = target_activity._chat_ctx.copy()
            chat_ctx.insert(items_to_insert)

        # only a reply made after the results landed had them in front of it and may have told
        # them; the person's words after them tell nothing
        items = target_activity._chat_ctx.items
        landed = {item.id for item in pending_items}
        first = next((n for n, item in enumerate(items) if item.id in landed), len(items))
        at_tail = not any(
            item.type == "message" and item.role == "assistant" for item in items[first + 1 :]
        )
        call_ids = [item.call_id for item in pending_items if item.type == "function_call_output"]
        tool_choice = _reply_tool_choice(u.tool_choice for u in updates)
        speech = session.generate_reply(
            instructions=_reply_instructions(
                self._executor._tool_options, at_tail=at_tail, call_ids=call_ids
            ),
            tool_choice=tool_choice if tool_choice is not None else NOT_GIVEN,
            chat_ctx=chat_ctx,
        )

        session._tool_execution_updated(
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
            # both paths record a reply cut before its first frame with no assistant text
            unheard = speech.interrupted and not any(
                item.type == "message" and item.role == "assistant" and item.text_content
                for item in speech.chat_items
            )
            reply_status: Literal["completed", "interrupted", "skipped"]
            if unheard:
                # cut off before any of it played: told after the person's turn, in call order
                for update in updates:
                    requeued = replace(update, say=None)  # its exact text was said on its own
                    bisect.insort(self._pending_updates, requeued, key=lambda u: u.seq)
                self._schedule_delivery(session)
                reply_status = "interrupted"
            elif speech.interrupted:
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

            if not unheard:
                for update in updates:
                    if update.result:
                        tool_replied(update.ctx, speech.chat_items)

            session._tool_execution_updated(
                ToolExecutionUpdatedEvent(
                    update=ToolReplyUpdated(
                        update_ids=call_ids, status=reply_status, speech_id=speech.id
                    )
                ),
            )

        speech.add_done_callback(_on_speech_done)


def _asks_reply(items: Sequence[ChatItem]) -> bool:
    return any(item.type == "function_call_output" and item.reply_required for item in items)


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
