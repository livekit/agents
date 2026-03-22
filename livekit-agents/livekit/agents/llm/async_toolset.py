from __future__ import annotations

import asyncio
import functools
import inspect
from typing import TYPE_CHECKING, Any, Literal, get_origin, get_type_hints

from attr import dataclass
from typing_extensions import TypeVar

from .. import utils
from ..llm.chat_context import ChatContext, ChatItem, ChatRole, FunctionCall
from ..llm.tool_context import (
    FunctionTool,
    RawFunctionTool,
    Tool,
    Toolset,
    find_function_tools,
    function_tool,
)
from ..log import logger
from ..voice.agent import _pass_through_activity_task_info
from ..voice.events import RunContext
from ..voice.generation import ToolExecutionOutput, make_tool_output

if TYPE_CHECKING:
    from ..voice.speech_handle import SpeechHandle


Userdata_T = TypeVar("Userdata_T")

SchedulingMode = Literal["when_idle", "interrupt", "silent"]


@dataclass
class AsyncResult:
    """Controls how the final result of an async tool is delivered.

    Return this from an async tool function to specify scheduling.
    Plain ``return value`` is equivalent to ``AsyncResult(output=value)``.
    """

    output: Any
    scheduling: SchedulingMode = "when_idle"


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
        self._completed_fut = asyncio.Future[None]()

        self._update_tasks: list[asyncio.Task[None]] = []

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
        scheduling: SchedulingMode = "when_idle",
        role: ChatRole | Literal["tool_output"] = "system",
        template: str = (
            "This is an update for background tool call {function_name}"
            " (call_id: {call_id}): {message}."
            " The task is still running,"
            " DO NOT provide any information not from the updating message."
        ),
        interrupt_on_completion: bool = True,
    ) -> asyncio.Task[None]:
        """Push an intermediate progress update into the conversation.

        Adds the update to the chat context and optionally triggers an LLM turn
        so the agent can narrate progress to the user.

        Args:
            message: Progress update (e.g. "Found 3 flights, selecting best...").
            scheduling: When to deliver the update. ``"when_idle"`` waits for the
                current speech to finish, ``"interrupt"`` interrupts immediately,
                ``"silent"`` adds to chat context without generating speech.
            role: The role of the message in the chat context.
                Defaults to ``"system"``.
            interrupt_on_completion: If True (default), interrupt the update speech
                when the tool completes so the final result can be delivered.

        Returns:
            asyncio.Task: A task that completes when the update speech finishes.
        """

        step = len(self._update_tasks) + 1

        async def _do_update() -> None:
            await self.speech_handle

            formatted_message = template.format(
                function_name=self.function_call.name,
                call_id=self.function_call.call_id,
                message=message,
            )

            items = _build_chat_items(self, formatted_message, role=role, suffix=f"update_{step}")
            await self._push_update(
                items,
                scheduling=scheduling,
                interrupt_on_completion=interrupt_on_completion,
            )

        task = asyncio.create_task(
            _do_update(), name=f"update_{self.function_call.name}_step_{step}"
        )
        self._update_tasks.append(task)
        return task

    async def _complete(self, output: Any, *, scheduling: SchedulingMode) -> None:
        """Deliver the final tool result into the conversation.

        Called by ``_execute_tool`` after the user function returns.
        Builds tool output items and pushes them through the same delivery
        path as ``update()``.
        """
        if isinstance(output, BaseException):
            logger.error(f"Error in async tool {self.function_call.name}", exc_info=output)

        result = self._make_tool_output(output, suffix="completed")
        if result.agent_task is not None:
            raise RuntimeError("returning Agent from async tool function is not supported")

        if result.fnc_call_out is None:
            return

        items: list[ChatItem] = [result.fnc_call, result.fnc_call_out]
        await self._push_update(items, scheduling=scheduling)

    async def _push_update(
        self,
        items: list[ChatItem],
        *,
        scheduling: SchedulingMode,
        interrupt_on_completion: bool = False,
    ) -> None:
        """Add items to the chat context and optionally generate a reply.

        This is the single delivery path used by both ``update()`` and
        ``_complete()``.  It always updates the chat context first,
        then — unless *scheduling* is ``"silent"`` — triggers a new LLM turn.
        """
        from ..voice.speech_handle import SpeechHandle

        activity = self.session._get_activity()

        # 2. If silent, we're done
        if scheduling == "silent":
            new_ctx = activity._agent._chat_ctx.copy()
            new_ctx.insert(items)
            await activity.update_chat_ctx(new_ctx)
            return

        # 3. Wait for current speech or interrupt
        if scheduling == "interrupt":
            await self.session.interrupt()
        else:
            while current_speech := self.session.current_speech:
                if current_speech._generations:
                    await current_speech._wait_for_generation()

        # 4. Generate reply and schedule
        priority = (
            SpeechHandle.SPEECH_PRIORITY_HIGH
            if scheduling == "interrupt"
            else SpeechHandle.SPEECH_PRIORITY_NORMAL
        )
        speech_handle = activity._generate_reply(
            new_messages=items, tool_choice="none", schedule_speech=False
        )
        activity._schedule_speech(speech_handle, priority=priority)

        # 5. Wait for playout, with optional early cancellation
        wait_for_playout = asyncio.create_task(speech_handle.wait_for_playout())
        try:
            if interrupt_on_completion:
                await asyncio.wait(
                    [wait_for_playout, self._completed_fut],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if self._completed_fut.done():
                    raise asyncio.CancelledError()
            else:
                await wait_for_playout
        except asyncio.CancelledError:
            logger.debug(
                "update cancelled",
                extra={
                    "call_id": self.function_call.call_id,
                    "function": self.function_call.name,
                },
            )
            await speech_handle.interrupt()
        finally:
            if not wait_for_playout.done():
                wait_for_playout.cancel()

    def _make_tool_output(self, output: Any | BaseException, *, suffix: str) -> ToolExecutionOutput:
        exception: BaseException | None = None
        if isinstance(output, BaseException):
            exception = output
            output = None

        fnc_call = FunctionCall(
            call_id=f"{self.function_call.call_id}/{suffix}",
            name=self.function_call.name,
            arguments=self.function_call.arguments,
            extra=self.function_call.extra,
        )
        return make_tool_output(fnc_call=fnc_call, output=output, exception=exception)


def _build_chat_items(
    ctx: AsyncRunContext,
    message: str,
    *,
    role: ChatRole | Literal["tool_output"],
    suffix: str,
) -> list[ChatItem]:
    """Build chat items for an update or completion message."""
    if role == "tool_output":
        result = ctx._make_tool_output(message, suffix=suffix)
        assert result.fnc_call_out is not None
        return [result.fnc_call, result.fnc_call_out]

    chat_ctx = ChatContext.empty()
    msg = chat_ctx.add_message(role=role, content=message)
    return [msg]


@dataclass
class _RunningTask:
    exe_task: asyncio.Task[Any]
    async_ctx: AsyncRunContext


def _is_async_context_type(ty: type) -> bool:
    return ty is AsyncRunContext or get_origin(ty) is AsyncRunContext


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
        on_duplicate: DuplicateMode = "confirm",
    ) -> None:
        super().__init__(id=id)
        self._on_duplicate = on_duplicate
        all_tools: list[Tool] = list(tools or [])
        all_tools.extend(find_function_tools(self))
        self._tools = [self._wrap_tool(t) for t in all_tools]
        self._tasks: dict[str, _RunningTask] = {}

    @property
    def tools(self) -> list[Tool]:
        return self._tools

    @function_tool
    async def get_running_tasks(self) -> list[dict]:
        """Get the list of running async tool calls. Use this to check if there is any running task for the same topic."""  # noqa: E501
        return [task.async_ctx.function_call.model_dump() for task in self._tasks.values()]

    @function_tool
    async def cancel_task(self, call_id: str) -> str:
        """Cancel a running async tool call"""
        success = await self.cancel(call_id)
        if success:
            return f"Task {call_id} cancelled successfully."
        else:
            return f"Task {call_id} not found or already completed."

    def _wrap_tool(self, tool: Tool) -> Tool:
        """Wrap a FunctionTool with async dispatch if it has an AsyncRunContext param."""
        if not isinstance(tool, (FunctionTool, RawFunctionTool)):
            return tool

        # find the AsyncRunContext parameter
        sig = inspect.signature(tool)
        try:
            type_hints = get_type_hints(tool)
        except Exception:
            type_hints = {}

        ctx_param_name: str | None = None
        for param_name in sig.parameters:
            hint = type_hints.get(param_name)
            if hint is not None and _is_async_context_type(hint):
                ctx_param_name = param_name
                break

        if ctx_param_name is None:
            return tool

        # build the new signature before the wrapper so bind() accepts
        # injected params like confirm_duplicate
        new_params = [
            p.replace(annotation=RunContext) if p.name == ctx_param_name else p
            for p in sig.parameters.values()
        ]
        if self._on_duplicate == "confirm":
            new_params.append(
                inspect.Parameter(
                    "confirm_duplicate",
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=str | None,
                )
            )
        wrapper_sig = sig.replace(parameters=new_params)

        # build wrapper that receives RunContext from the framework,
        # converts to AsyncRunContext, and dispatches as a background task.
        @functools.wraps(tool)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = wrapper_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            run_ctx = arguments.pop(ctx_param_name)
            assert isinstance(run_ctx, RunContext)

            call_id = run_ctx.function_call.call_id
            fnc_name = run_ctx.function_call.name

            # --- Duplicate detection ---
            if self._on_duplicate != "allow":
                existing = next(
                    (t for t in self._tasks.values() if t.async_ctx.function_call.name == fnc_name),
                    None,
                )
                if existing is not None:
                    existing_call = existing.async_ctx.function_call

                    if self._on_duplicate == "replace":
                        await self.cancel(existing_call.call_id)
                    elif self._on_duplicate == "reject":
                        return (
                            f"Tool `{fnc_name}` is already running "
                            f"(call_id: {existing_call.call_id}, "
                            f"args: {existing_call.arguments}). "
                            f"Use `cancel_task('{existing_call.call_id}')` to cancel it."
                        )
                    elif self._on_duplicate == "confirm":
                        confirm_value = arguments.pop("confirm_duplicate", None)
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
                elif self._on_duplicate == "confirm":
                    # No duplicate found — clean up confirm_duplicate from arguments
                    arguments.pop("confirm_duplicate", None)

            if call_id in self._tasks:
                raise ValueError(f"Task already running for call_id: {call_id}")

            run_ctx.session._register_async_toolset(self)  # cleanup in session.aclose

            # inject AsyncRunContext into the function arguments
            async_ctx = AsyncRunContext(run_ctx=run_ctx)
            arguments[ctx_param_name] = async_ctx

            async def _execute_tool() -> Any:
                scheduling: SchedulingMode = "when_idle"
                try:
                    output = await tool(**arguments)

                    if isinstance(output, AsyncResult):
                        scheduling = output.scheduling
                        output = output.output
                except asyncio.CancelledError:
                    logger.debug(
                        "async tool cancelled",
                        extra={
                            "call_id": call_id,
                            "function": fnc_name,
                        },
                    )
                    return
                except BaseException as e:
                    output = e
                finally:
                    if not async_ctx._completed_fut.done():
                        async_ctx._completed_fut.set_result(None)

                if not async_ctx._pending_fut.done():
                    # pending() was never called — return output directly
                    # as the tool result (handled by the framework)
                    if isinstance(output, BaseException):
                        async_ctx._pending_fut.set_exception(output)
                    else:
                        async_ctx._pending_fut.set_result(output)
                    return

                # Tool ran in background (pending was set).
                # Deliver the final result through the conversation.
                if output is None:
                    return

                await async_ctx._complete(output, scheduling=scheduling)

            exe_task = asyncio.create_task(_execute_tool(), name=f"async_tool_{fnc_name}")
            _pass_through_activity_task_info(exe_task)
            self._tasks[call_id] = _RunningTask(exe_task=exe_task, async_ctx=async_ctx)
            exe_task.add_done_callback(lambda _: self._tasks.pop(call_id, None))

            return await async_ctx._pending_fut

        wrapped_tool = tool.__class__(wrapper, tool.info)  # type: ignore[arg-type]
        wrapped_tool.__signature__ = wrapper_sig  # type: ignore[union-attr]
        wrapped_tool.__annotations__ = {
            **wrapped_tool.__annotations__,
            ctx_param_name: RunContext,
        }
        if self._on_duplicate == "confirm":
            wrapped_tool.__annotations__["confirm_duplicate"] = str | None

        return wrapped_tool

    async def cancel(self, call_id: str) -> bool:
        task = self._tasks.get(call_id)
        if task is not None:
            await utils.aio.cancel_and_wait(task.exe_task, *task.async_ctx._update_tasks)
            return True
        return False

    async def shutdown(self) -> None:
        """Cancel all tasks."""
        tasks: list[asyncio.Task[Any]] = []
        for task in self._tasks.values():
            tasks.append(task.exe_task)
            tasks.extend(task.async_ctx._update_tasks)
        await utils.aio.cancel_and_wait(*tasks)
        self._tasks.clear()
