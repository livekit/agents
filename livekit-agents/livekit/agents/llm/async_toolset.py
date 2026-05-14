from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_origin, get_type_hints

from .. import utils
from ..job import JobContext, get_job_context
from ..llm.chat_context import ChatContext, ChatItem, FunctionCall
from ..llm.tool_context import (
    FunctionTool,
    RawFunctionTool,
    Tool,
    ToolError,
    Toolset,
    function_tool,
)
from ..llm.utils import function_arguments_to_pydantic_model, prepare_function_arguments
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..voice.agent import _pass_through_activity_task_info
from ..voice.events import RunContext, Userdata_T
from ..voice.generation import ToolExecutionOutput, make_tool_output

if TYPE_CHECKING:
    from ..voice.agent import Agent
    from ..voice.agent_activity import AgentActivity
    from ..voice.agent_session import AgentSession


DuplicateMode = Literal["allow", "replace", "reject", "confirm"]

# Module-level registry of running async tasks, keyed by (JobContext | None, call_id).
# Registered when a task starts; auto-removed via done callback. Shared across all managers
# so `get_running_tasks` / `cancel_task` work uniformly whether the task came from an
# AsyncToolset's manager or from an AgentActivity's manager.
_RunningTasks: dict[tuple[JobContext | None, str], _RunningTask] = {}


@function_tool
async def get_running_tasks() -> list[dict]:
    """Get the list of running async tool calls."""
    job_ctx = get_job_context(required=False)
    return [
        task.ctx.function_call.model_dump()
        for (ctx, _), task in list(_RunningTasks.items())
        if ctx is job_ctx
    ]


@function_tool
async def cancel_task(call_id: str) -> str:
    """Cancel a running async tool call by call_id."""
    job_ctx = get_job_context(required=False)
    task = _RunningTasks.get((job_ctx, call_id))
    if task and await task.ctx._manager.cancel(call_id):
        return f"Task {call_id} cancelled successfully."
    return f"Task {call_id} not found or already completed."


UPDATE_TEMPLATE = """The tool `{function_name}` has updated, message: {message}
The task is still running, so DON'T make up or give information not included in the message above.
In your next reply to the user, naturally include this update (e.g. "by the way, ...") if you haven't already."""

DUPLICATE_REJECT = """Same tool `{function_name}` is already running:
{running_fnc_calls}
If you want to cancel the existing one, call `cancel_task` with call_id."""

DUPLICATE_CONFIRM = """Same tool `{function_name}` is already running:
{running_fnc_calls}
Re-call with confirm duplicate True to run a duplicate if needed,
or if you want to cancel the existing one, call `cancel_task` with call_id."""

REPLY_INSTRUCTIONS = """New results arrived from background tool calls (call_ids: {pending_call_ids}).
Review your most recent assistant messages.
- If your previous messages already conveyed these results to the user, return an empty response (no text at all).
- Otherwise, summarize the results naturally with a transition like "by the way, ...".
Do NOT repeat information you have already told the user."""


class AsyncRunContext(RunContext[Userdata_T]):
    """Run context for async tool functions."""

    def __init__(self, *, run_ctx: RunContext[Userdata_T], manager: _AsyncToolManager) -> None:
        super().__init__(
            session=run_ctx.session,
            speech_handle=run_ctx.speech_handle,
            function_call=run_ctx.function_call,
        )
        self._manager = manager
        self._pending_fut = asyncio.Future[Any]()
        self._step_idx: int = 0

    async def update(self, message: str | Any, *, _template: str = UPDATE_TEMPLATE) -> None:
        """Push a progress update into the conversation.

        Updates the chat context immediately and enqueues a speech delivery
        at the manager level. Multiple updates from different tools are
        coalesced — only the latest pending update triggers a reply.

        Args:
            message: The update message for the LLM (e.g. "Found 3 flights, selecting the best option...").
        """
        if isinstance(message, str):
            message = _template.format(
                function_name=self.function_call.name,
                call_id=self.function_call.call_id,
                message=message,
            )

        if not self._pending_fut.done():
            # first update will mark the tool execution in AgentActivity done
            self._pending_fut.set_result(message)
            # make the speech handle awaitable in the rest of the function
            self._function_call.extra["__livekit_agents_tool_pending"] = True
            return

        self._step_idx += 1

        tool_output = self._make_tool_output(
            message, call_id=f"{self.function_call.call_id}_update_{self._step_idx}"
        )
        if tool_output.fnc_call_out is None:
            return

        tool_items: list[ChatItem] = [tool_output.fnc_call, tool_output.fnc_call_out]
        await self._manager._enqueue_reply(self, tool_items)

    def _make_tool_output(
        self, output: Any | BaseException, call_id: str | None
    ) -> ToolExecutionOutput:
        exception: BaseException | None = None
        if isinstance(output, BaseException):
            exception = output
            output = None

        fnc_call = FunctionCall(
            call_id=call_id if call_id is not None else self.function_call.call_id,
            name=self.function_call.name,
            arguments=self.function_call.arguments,
            extra=self.function_call.extra,
        )
        return make_tool_output(fnc_call=fnc_call, output=output, exception=exception)


@dataclass
class _RunningTask:
    ctx: AsyncRunContext
    exe_task: asyncio.Task[Any]


@dataclass
class _PendingUpdate:
    ctx: AsyncRunContext
    items: list[ChatItem]
    # agent that received the eager chat_ctx insert
    target: Agent


class _AsyncToolManager:
    """Lifecycle manager for in-flight async tool calls.

    Owns the running-task registry for one scope, the duplicate-call policy, and the
    progress-reply coalescer. Used internally by :class:`AsyncToolset` (toolset-scoped
    lifetime — survives agent handoff) and by :class:`~livekit.agents.voice.AgentActivity`
    (activity-scoped lifetime — tasks are cancelled on handoff).
    """

    def __init__(
        self,
        *,
        on_duplicate_call: DuplicateMode = "confirm",
        owning_activity: AgentActivity | None = None,
    ) -> None:
        self._on_duplicate_call: DuplicateMode = on_duplicate_call
        self._running_tasks: dict[str, _RunningTask] = {}

        # speech delivery — shared across all tools in this manager
        self._pending_updates: list[_PendingUpdate] = []
        self._reply_task: asyncio.Task[None] | None = None

        # owning_activity determines which agent gets the reply and when:
        # if set, wait for this activity to be idle; if None, deliver to
        # the current activity at reply time (session-scoped toolsets).
        self._owning_activity: AgentActivity | None = owning_activity

    def set_owning_activity(self, activity: AgentActivity | None) -> None:
        self._owning_activity = activity

    @property
    def has_running_tasks(self) -> bool:
        return bool(self._running_tasks)

    async def spawn(
        self,
        *,
        function_callable: Callable[[AsyncRunContext], Awaitable[Any]],
        run_ctx: RunContext,
        confirm_duplicate: bool | None = None,
    ) -> Any:
        """Run ``function_callable`` in the background. Returns when the first
        ``ctx.update()`` lands or the callable returns (whichever comes first).

        The manager creates the :class:`AsyncRunContext` and hands it to the callable.
        Everything inside the callable — argument prep, mock substitution, real tool
        invocation — is the caller's concern; the manager only owns the async lifecycle
        (background task, duplicate detection, registry, reply coalescing)."""
        call_id = run_ctx.function_call.call_id
        fnc_name = run_ctx.function_call.name

        duplicate_result = await self._check_duplicate(fnc_name, confirm_duplicate)
        if duplicate_result is not None:
            logger.debug(
                "duplicate tool call rejected",
                extra={"call_id": call_id, "function": fnc_name},
            )
            return duplicate_result

        if call_id in self._running_tasks:
            raise ValueError(f"Task already running for call_id: {call_id}")

        async_ctx = AsyncRunContext(run_ctx=run_ctx, manager=self)

        async def _execute_tool() -> Any:
            try:
                output = await function_callable(async_ctx)
            except asyncio.CancelledError:
                logger.debug(
                    "async tool cancelled",
                    extra={"call_id": call_id, "function": fnc_name},
                )
                if not async_ctx._pending_fut.done():
                    async_ctx._pending_fut.set_result(None)
                return
            except Exception as e:
                output = e
                logger.exception(
                    "error in async tool",
                    extra={"call_id": call_id, "function": fnc_name},
                )

            if not async_ctx._pending_fut.done():
                # pending() was never called — return output directly
                if isinstance(output, BaseException):
                    async_ctx._pending_fut.set_exception(output)
                else:
                    async_ctx._pending_fut.set_result(output)
                return

            if output is None:
                return

            tool_output = async_ctx._make_tool_output(output, call_id=f"{call_id}_finished")
            if tool_output.fnc_call_out is None:
                return

            await self._enqueue_reply(async_ctx, [tool_output.fnc_call, tool_output.fnc_call_out])

        exe_task = asyncio.create_task(_execute_tool(), name=f"async_tool_{fnc_name}")
        _pass_through_activity_task_info(exe_task)

        running_task = _RunningTask(ctx=async_ctx, exe_task=exe_task)
        self._running_tasks[call_id] = running_task

        # register in the module-level registry
        task_key = (get_job_context(required=False), call_id)
        _RunningTasks[task_key] = running_task

        def _on_done(_: asyncio.Task[Any]) -> None:
            self._running_tasks.pop(call_id, None)
            _RunningTasks.pop(task_key, None)

        exe_task.add_done_callback(_on_done)

        return await async_ctx._pending_fut

    async def cancel(self, call_id: str) -> bool:
        task = self._running_tasks.get(call_id)
        if task is not None:
            if not task.ctx.speech_handle.allow_interruptions:
                raise ToolError(
                    f"Tool call {call_id} is not cancellable because interruptions are disallowed"
                )
            await utils.aio.cancel_and_wait(task.exe_task)
            return True
        return False

    async def aclose(self) -> None:
        """Cancel all running tasks and drop any buffered replies."""
        self._pending_updates.clear()
        tasks = [task.exe_task for task in self._running_tasks.values()]
        if self._reply_task is not None:
            tasks.append(self._reply_task)
        await utils.aio.cancel_and_wait(*tasks)
        self._running_tasks.clear()

    async def _enqueue_reply(self, ctx: AsyncRunContext, items: list[ChatItem]) -> None:
        # eager insert so any reply before delivery sees the items
        target = (
            self._owning_activity.agent
            if self._owning_activity is not None
            else ctx.session.current_agent
        )
        chat_ctx = target.chat_ctx.copy()
        chat_ctx.insert(items)
        await target.update_chat_ctx(chat_ctx)
        # match normal tool dispatch: items go on session history too
        ctx.session.history.insert(items)

        self._pending_updates.append(_PendingUpdate(ctx=ctx, items=items, target=target))

        if self._reply_task is None or self._reply_task.done():
            self._reply_task = asyncio.create_task(
                self._deliver_reply(ctx.session), name="async_tool_manager_deliver_reply"
            )
            # let an active run wait for the deferred reply to land
            run_state = ctx.session._global_run_state
            if run_state is not None:
                run_state._watch_handle(self._reply_task)

    async def _deliver_reply(self, session: AgentSession) -> None:
        from ..voice.agent_activity import ActivityClosedError

        target_agent: Agent
        try:
            if self._owning_activity is not None:
                await self._owning_activity.wait_for_idle()
                target_agent = self._owning_activity.agent
            else:
                target_activity = await session.wait_for_idle()
                target_agent = target_activity.agent
        except ActivityClosedError:
            logger.debug("dropping async tool reply — owning activity closed")
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
                "agent handoff happened while async tool waiting for reply delivering",
                extra={
                    "tools": [
                        u.ctx.function_call.name for u in updates if u.target is not target_agent
                    ],
                },
            )
            chat_ctx = target_agent.chat_ctx.copy()
            chat_ctx.insert(items_to_insert)

        # always fire, asks the LLM to return an empty response if these results were already conveyed
        pending_call_ids = [
            item.call_id for item in pending_items if item.type == "function_call_output"
        ]
        session.generate_reply(
            instructions=REPLY_INSTRUCTIONS.format(pending_call_ids=pending_call_ids),
            tool_choice="none",
            chat_ctx=chat_ctx,
        )

    async def _check_duplicate(self, fnc_name: str, confirm_duplicate: bool | None) -> str | None:
        """Check for duplicate running tasks. Returns a message if blocked, None otherwise."""
        if self._on_duplicate_call == "allow":
            return None

        running_fnc_calls = [
            t.ctx.function_call
            for t in self._running_tasks.values()
            if t.ctx.function_call.name == fnc_name
        ]
        if len(running_fnc_calls) == 0:
            return None

        if self._on_duplicate_call == "replace":
            results = await asyncio.gather(
                *[self.cancel(fnc_call.call_id) for fnc_call in running_fnc_calls],
                return_exceptions=True,
            )
            exceptions = [result for result in results if isinstance(result, Exception)]
            if exceptions:
                error_messages = "\n".join([str(e) for e in exceptions])
                raise ToolError(f"Failed to cancel duplicate tool calls: {error_messages}")
            return None

        if self._on_duplicate_call == "reject":
            return DUPLICATE_REJECT.format(
                function_name=fnc_name,
                running_fnc_calls="\n".join(
                    [fnc_call.model_dump_json() for fnc_call in running_fnc_calls]
                ),
            )

        elif self._on_duplicate_call == "confirm" and not confirm_duplicate:
            return DUPLICATE_CONFIRM.format(
                function_name=fnc_name,
                running_fnc_calls="\n".join(
                    [fnc_call.model_dump_json() for fnc_call in running_fnc_calls]
                ),
            )

        return None


class AsyncToolset(Toolset):
    """A toolset for running long-running functions in the background.

    Tools with an :class:`AsyncRunContext` parameter are wrapped to run in the background.
    Each ``ctx.update()`` and the final ``return`` inject a tool output into the conversation;
    the agent then generates a natural-language reply to the user based on that output.

    Example::

        @function_tool
        async def book_flight(ctx: AsyncRunContext, origin: str, destination: str) -> dict:
            await ctx.update(f"Looking up flights from {origin} to {destination}...")
            # → agent says: "Sure, let me look up flights from NYC to Tokyo for you!"

            flights = await search_flights(origin, destination)
            await ctx.update(f"Found {len(flights)} flights, picking the best one...")
            # → agent says: "I found 3 flights — picking the best option for you now."

            booking = await book_best_flight(flights)
            return {"confirmation": booking.id}
            # → agent says: "All set! Your booking confirmation number is FL-847293."

        async_tools = AsyncToolset(id="booking", tools=[book_flight])
    """

    def __init__(
        self,
        *,
        id: str,
        tools: list[Tool] | None = None,
        on_duplicate_call: DuplicateMode = "confirm",
    ) -> None:
        super().__init__(id=id, tools=tools)

        self._manager = _AsyncToolManager(on_duplicate_call=on_duplicate_call)
        self._tools = [
            self._wrap_tool(t) if isinstance(t, FunctionTool | RawFunctionTool) else t
            for t in self._tools
        ]
        self._tools.extend([get_running_tasks, cancel_task])

    async def cancel(self, call_id: str) -> bool:
        return await self._manager.cancel(call_id)

    def bind_activity(self, activity: AgentActivity | None) -> None:
        # set when the toolset lives on Agent.tools (activity-scoped reply);
        # None when it lives on AgentSession.tools (session-scoped reply).
        self._manager.set_owning_activity(activity)

    async def aclose(self) -> None:
        await super().aclose()
        await self._manager.aclose()

    def _wrap_tool(self, tool: FunctionTool | RawFunctionTool) -> FunctionTool | RawFunctionTool:
        if not _has_async_context_param(tool):
            return tool

        raw_schema = _build_raw_schema(tool)

        # inject confirm_duplicate parameter for confirm mode
        confirm_duplicate_param = "_lk_agents_confirm_duplicate"
        if self._manager._on_duplicate_call == "confirm":
            props = raw_schema["parameters"].setdefault("properties", {})
            props[confirm_duplicate_param] = {
                "type": ["boolean"],
                "description": (
                    "Set this to True to confirm you want to run a duplicate. "
                    "Only do this when user confirms the duplication is needed."
                ),
                "default": False,
            }

        manager = self._manager

        @function_tool(raw_schema=raw_schema, flags=tool.info.flags)
        async def wrapper(ctx: RunContext, raw_arguments: dict[str, Any]) -> Any:
            confirm_duplicate = raw_arguments.pop(confirm_duplicate_param, None)

            # mock lookup happens at call time, so mocks of AsyncToolset-wrapped
            # tools run through this toolset's manager — same async lifecycle as
            # the real tool.
            from ..voice.run_result import _MockToolsContextVar, _run_mock

            mock_tools = _MockToolsContextVar.get({}).get(type(ctx.session.current_agent), {})
            mock = mock_tools.get(tool.info.name)

            async def _call_tool(async_ctx: AsyncRunContext) -> Any:
                fnc_args, fnc_kwargs = prepare_function_arguments(
                    fnc=tool, json_arguments=raw_arguments, call_ctx=async_ctx
                )
                if mock is not None:
                    return await _run_mock(mock, *fnc_args, **fnc_kwargs)
                return await tool(*fnc_args, **fnc_kwargs)

            return await manager.spawn(
                function_callable=_call_tool,
                run_ctx=ctx,
                confirm_duplicate=confirm_duplicate,
            )

        # marker so _execute_tools_task knows this wrapper does its own mock dispatch
        wrapper._lk_async_wrapper = True  # type: ignore[attr-defined]
        return wrapper


def _is_async_context_type(ty: type) -> bool:
    return ty is AsyncRunContext or get_origin(ty) is AsyncRunContext


def _has_async_context_param(tool: FunctionTool | RawFunctionTool) -> bool:
    """Check if the tool has an AsyncRunContext parameter."""
    try:
        type_hints = get_type_hints(tool)
    except Exception:
        return False
    return any(_is_async_context_type(h) for h in type_hints.values())


def _is_async_toolset_wrapper(tool: object) -> bool:
    return getattr(tool, "_lk_async_wrapper", False) is True


def _build_raw_schema(tool: FunctionTool | RawFunctionTool) -> dict[str, Any]:
    """Build a raw JSON schema dict from a tool, suitable for RawFunctionTool."""
    if isinstance(tool, RawFunctionTool):
        return copy.deepcopy(tool.info.raw_schema)

    model = function_arguments_to_pydantic_model(tool)
    return {
        "name": tool.info.name,
        "description": tool.info.description or "",
        "parameters": model.model_json_schema(),
    }
