from __future__ import annotations

import asyncio
import copy
import inspect
from typing import TYPE_CHECKING, Any, Literal, get_origin, get_type_hints

from attr import dataclass
from typing_extensions import TypeVar

from .. import utils
from ..llm.chat_context import ChatItem, FunctionCall
from ..llm.tool_context import (
    FunctionTool,
    RawFunctionTool,
    Tool,
    Toolset,
    function_tool,
)
from ..llm.utils import function_arguments_to_pydantic_model
from ..log import logger
from ..voice.agent import _pass_through_activity_task_info
from ..voice.events import RunContext
from ..voice.generation import ToolExecutionOutput, make_tool_output

if TYPE_CHECKING:
    from ..voice.speech_handle import SpeechHandle


Userdata_T = TypeVar("Userdata_T")

SchedulingMode = Literal["when_idle", "interrupt", "silent"]


class AsyncRunContext(RunContext[Userdata_T]):
    """Run context for async tool functions.

    Extends :class:`RunContext` with methods to control the conversation while
    the function runs in the background.

    Example::

        async_tools = AsyncToolset(id="booking")

        @function_tool
        async def book_flight(ctx: AsyncRunContext, origin: str, destination: str) -> dict:
            ctx.pending(f"Looking up flights from {origin} to {destination}...")

            flights = await search_flights(origin, destination)
            ctx.update(f"Found {len(flights)} flights, selecting the best option...")

            booking = await book_best_flight(flights)
            return {"confirmation": booking.id}
    """

    def __init__(self, *, run_ctx: RunContext[Userdata_T]) -> None:
        super().__init__(
            session=run_ctx.session,
            speech_handle=run_ctx.speech_handle,
            function_call=run_ctx.function_call,
        )
        self._pending_fut = asyncio.Future[Any]()

        self._update_task: asyncio.Task[SpeechHandle | None] | None = None
        self._step_idx: int = 0

    def pending(
        self,
        message: str | Any,
        *,
        template: str = (
            "Tool {function_name} (call_id: {call_id}) is running in background: {message}."
        ),
    ) -> SpeechHandle:
        """Set the message returned to the LLM when the tool is first called.

        Args:
            message: The pending message for the LLM (e.g. "Searching flights...").
        """
        if self._pending_fut.done():
            raise ValueError("Pending message already set")

        if isinstance(message, str):
            message = template.format(
                function_name=self.function_call.name,
                call_id=self.function_call.call_id,
                message=message,
            )

        self._pending_fut.set_result(message)

        # make the speech handle awaitable in the rest of the function
        self._function_call.extra["__livekit_agents_tool_pending"] = True

        return self.speech_handle

    def update(
        self,
        message: str,
        *,
        template: str = (
            "This is an update for background tool call {function_name}"
            " (call_id: {call_id}): {message}."
            " The task is still running,"
            " DO NOT provide any information not from the updating message."
        ),
    ) -> asyncio.Task[SpeechHandle | None]:
        """Push an intermediate progress update into the conversation"""

        message = template.format(
            function_name=self.function_call.name,
            call_id=self.function_call.call_id,
            message=message,
        )

        return asyncio.create_task(
            self._commit_update(message, msg_type="update", old_task=self._update_task)
        )

    async def _commit_update(
        self,
        msg: Any,
        msg_type: Literal["update", "completion"],
        old_task: asyncio.Task[SpeechHandle | None] | None,
    ) -> SpeechHandle | None:
        current_step = self._step_idx = self._step_idx + 1
        if old_task is not None:
            await old_task

        # make mock tool call and output
        suffix = f"update_{current_step}" if msg_type == "update" else "completion"
        tool_output = self._make_tool_output(msg, call_id=f"{self.function_call.call_id}/{suffix}")
        if tool_output.fnc_call_out is None:
            return None
        items: list[ChatItem] = [tool_output.fnc_call, tool_output.fnc_call_out]

        # update chat context
        agent = self.session.current_agent
        new_ctx = agent._chat_ctx.copy()
        new_ctx.insert(items)
        await agent.update_chat_ctx(new_ctx)

        if isinstance(msg, BaseException):
            logger.error(
                "error in async tool",
                extra={
                    "call_id": self.function_call.call_id,
                    "function": self.function_call.name,
                    "error": str(msg),
                },
            )
            return self.session.generate_reply(tool_choice="none")

        await self.session.wait_for_inactive()

        # skip if there is a new update
        extra = {
            "call_id": self.function_call.call_id,
            "function": self.function_call.name,
            "msg_type": msg_type,
            "current_step": current_step,
            "new_step": self._step_idx,
        }
        if current_step != self._step_idx:
            logger.debug(
                "skipping reply from async update since there is a new update", extra=extra
            )
            return None

        # skip if there is already a speech after the update
        last_item = sorted(items, key=lambda x: x.created_at)[-1]
        if (agent_items := agent.chat_ctx.items) and agent_items[-1] != last_item:
            logger.debug(
                "skipping reply from async update since there was already a speech after the update",
                extra=extra,
            )
            return None

        return self.session.generate_reply(tool_choice="none")

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
    exe_task: asyncio.Task[Any]
    async_ctx: AsyncRunContext


class AsyncToolset(Toolset):
    """A toolset for running long-running functions in the background.

    Tools with an :class:`AsyncRunContext` parameter are automatically wrapped
    to run in the background. The framework injects a :class:`RunContext` which
    is converted to :class:`AsyncRunContext` before calling the user function.
    Tools without ``AsyncRunContext`` work normally.

    Example::

        async_tools = AsyncToolset(id="booking")

        @function_tool
        async def book_flight(ctx: AsyncRunContext, origin: str, destination: str) -> dict:
            ctx.pending(f"Looking up flights from {origin} to {destination}...")

            flights = await search_flights(origin, destination)
            ctx.update(f"Found {len(flights)} flights, picking the best one...")

            booking = await book_best_flight(flights)
            return {"confirmation": booking.id}

        agent = Agent(tools=[async_tools], instructions="...")

    Conversation flow:

    1. User: "Book me a flight to Paris"
    2. LLM calls ``book_flight(origin="NYC", destination="Paris")``
    3. ``ctx.pending(...)`` -> LLM says "Looking up flights from NYC to Paris..."
    4. ``ctx.update(...)`` -> LLM says "Found 3 flights, picking the best one..."
    5. Function returns -> LLM says "Your flight is booked! Confirmation: ABC123"
    """

    DuplicateMode = Literal["allow", "replace", "reject", "confirm"]

    def __init__(
        self,
        *,
        id: str,
        tools: list[Tool] | None = None,
        on_duplicate_call: DuplicateMode = "confirm",
    ) -> None:
        super().__init__(id=id, tools=tools)

        self._on_duplicate_call = on_duplicate_call
        self._tools = [
            self._wrap_tool(t) if isinstance(t, FunctionTool | RawFunctionTool) else t
            for t in self._tools
        ]

        self._running_tasks: dict[str, _RunningTask] = {}

    @function_tool
    async def get_running_tasks(self) -> list[dict]:
        """Get the list of running async tool calls."""
        return [task.async_ctx.function_call.model_dump() for task in self._running_tasks.values()]

    @function_tool
    async def cancel_task(self, call_id: str) -> str:
        """Cancel a running async tool call by call_id."""
        success = await self.cancel(call_id)
        if success:
            return f"Task {call_id} cancelled successfully."
        else:
            return f"Task {call_id} not found or already completed."

    async def cancel(self, call_id: str) -> bool:
        task = self._running_tasks.get(call_id)
        if task is not None:
            await utils.aio.cancel_and_wait(task.exe_task)
            if task.async_ctx._update_task is not None:
                await utils.aio.cancel_and_wait(task.async_ctx._update_task)
            return True
        return False

    async def aclose(self) -> None:
        """Cancel all tasks."""
        tasks = []
        for task in self._running_tasks.values():
            tasks.append(task.exe_task)
            if task.async_ctx._update_task is not None:
                tasks.append(task.async_ctx._update_task)
        await utils.aio.cancel_and_wait(*tasks)
        self._running_tasks.clear()

    def _wrap_tool(self, tool: FunctionTool | RawFunctionTool) -> FunctionTool | RawFunctionTool:
        if not _has_async_context_param(tool):
            return tool

        raw_schema = _build_raw_schema(tool)

        # inject confirm_duplicate parameter for confirm mode
        if self._on_duplicate_call == "confirm":
            props = raw_schema["parameters"].setdefault("properties", {})
            props["confirm_duplicate"] = {
                "type": ["string", "null"],
                "description": (
                    "Set this to the call_id of an existing running task "
                    "to confirm you want to run a duplicate."
                ),
                "default": None,
            }

        @function_tool(raw_schema=raw_schema, flags=tool.info.flags)
        async def wrapper(ctx: RunContext, raw_arguments: dict[str, Any]) -> Any:
            call_id = ctx.function_call.call_id
            fnc_name = ctx.function_call.name

            # duplicate detection
            duplicate_result = self._check_duplicate(fnc_name, raw_arguments)
            if duplicate_result is not None:
                return duplicate_result

            if call_id in self._running_tasks:
                raise ValueError(f"Task already running for call_id: {call_id}")

            async_ctx = AsyncRunContext(run_ctx=ctx)

            async def _execute_tool() -> Any:
                try:
                    tool_kwargs = _prepare_tool_kwargs(tool, raw_arguments, async_ctx)
                    output = await tool(**tool_kwargs)
                except asyncio.CancelledError:
                    logger.debug(
                        "async tool cancelled",
                        extra={"call_id": call_id, "function": fnc_name},
                    )
                    return
                except BaseException as e:
                    output = e

                if not async_ctx._pending_fut.done():
                    # pending() was never called — return output directly
                    if isinstance(output, BaseException):
                        async_ctx._pending_fut.set_exception(output)
                    else:
                        async_ctx._pending_fut.set_result(output)
                    return

                if output is None:
                    return
                await async_ctx._commit_update(
                    output, msg_type="completion", old_task=async_ctx._update_task
                )

            exe_task = asyncio.create_task(_execute_tool(), name=f"async_tool_{fnc_name}")
            _pass_through_activity_task_info(exe_task)

            self._running_tasks[call_id] = _RunningTask(exe_task=exe_task, async_ctx=async_ctx)
            exe_task.add_done_callback(lambda _: self._running_tasks.pop(call_id, None))

            return await async_ctx._pending_fut

        return wrapper

    def _check_duplicate(self, fnc_name: str, raw_arguments: dict[str, Any]) -> str | None:
        """Check for duplicate running tasks. Returns a message if blocked, None otherwise."""
        if self._on_duplicate_call == "allow":
            return None

        existing = next(
            (t for t in self._running_tasks.values() if t.async_ctx.function_call.name == fnc_name),
            None,
        )
        if existing is not None:
            existing_call = existing.async_ctx.function_call

            if self._on_duplicate_call == "replace":
                asyncio.ensure_future(self.cancel(existing_call.call_id))
                return None
            elif self._on_duplicate_call == "reject":
                return (
                    f"Tool `{fnc_name}` is already running "
                    f"(call_id: {existing_call.call_id}, "
                    f"args: {existing_call.arguments}). "
                    f"Use `cancel_task('{existing_call.call_id}')` to cancel it."
                )
            elif self._on_duplicate_call == "confirm":
                confirm_value = raw_arguments.pop("confirm_duplicate", None)
                if confirm_value != existing_call.call_id:
                    return (
                        f"Tool `{fnc_name}` is already running "
                        f"(call_id: {existing_call.call_id}, "
                        f"args: {existing_call.arguments}). "
                        f"Re-call with `confirm_duplicate="
                        f"'{existing_call.call_id}'` to run anyway, "
                        f"or use `cancel_task('{existing_call.call_id}')` "
                        f"to cancel the existing task."
                    )
        elif self._on_duplicate_call == "confirm":
            raw_arguments.pop("confirm_duplicate", None)

        return None


def _is_async_context_type(ty: type) -> bool:
    return ty is AsyncRunContext or get_origin(ty) is AsyncRunContext


def _has_async_context_param(tool: FunctionTool | RawFunctionTool) -> bool:
    """Check if the tool has an AsyncRunContext parameter."""
    try:
        type_hints = get_type_hints(tool)
    except Exception:
        return False
    return any(_is_async_context_type(h) for h in type_hints.values())


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


def _prepare_tool_kwargs(
    tool: FunctionTool | RawFunctionTool,
    raw_arguments: dict[str, Any],
    async_ctx: AsyncRunContext,
) -> dict[str, Any]:
    """Build kwargs to call the original tool, injecting AsyncRunContext."""
    sig = inspect.signature(tool)
    try:
        type_hints = get_type_hints(tool)
    except Exception:
        type_hints = {}

    kwargs: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        hint = type_hints.get(param_name)
        if hint is not None and _is_async_context_type(hint):
            kwargs[param_name] = async_ctx
        elif param_name == "raw_arguments" and isinstance(tool, RawFunctionTool):
            kwargs["raw_arguments"] = raw_arguments
        elif param_name in raw_arguments:
            kwargs[param_name] = raw_arguments[param_name]
        elif param.default is not inspect.Parameter.empty:
            kwargs[param_name] = param.default
    return kwargs
