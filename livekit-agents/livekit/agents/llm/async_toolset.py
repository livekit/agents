from __future__ import annotations

import asyncio
import functools
import inspect
from typing import TYPE_CHECKING, Any, Literal, get_origin, get_type_hints

from attr import dataclass
from typing_extensions import TypeVar

from .. import utils
from ..llm.chat_context import ChatContext, ChatRole, FunctionCall
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
        self._update_step = 0

    def pending(self, message: str | Any) -> SpeechHandle:
        """Set the message returned to the LLM when the tool is first called.

        Args:
            message: The pending message for the LLM (e.g. "Searching flights...").
        """
        if self._pending_fut.done():
            raise ValueError("Pending message already set")
        self._pending_fut.set_result(message)

        # make the speech handle awaitable in the rest of the function
        self._function_call.extra["__livekit_agents_tool_pending"] = True

        return self.speech_handle

    def update(
        self,
        message: str,
        *,
        role: ChatRole | Literal["instructions", "tool_output"] = "instructions",
        template: str = "[Update for tool call {function_name}]: {message}",
    ) -> SpeechHandle:
        """Push an intermediate progress update into the conversation.

        Triggers a new LLM turn with the update injected as instructions,
        so the agent can narrate progress to the user.

        Args:
            message: Progress update (e.g. "Found 3 flights, selecting best...").
            role: The role of the message. Defaults to "instructions".

        Returns:
            SpeechHandle: A handle to the generated reply.
        """
        self._update_step += 1

        formatted_message = template.format(function_name=self.function_call.name, message=message)
        if role == "instructions":
            return self.session.generate_reply(instructions=formatted_message, tool_choice="none")

        chat_ctx = ChatContext.empty()
        if role == "tool_output":
            result = self._make_tool_output(formatted_message, suffix=f"update_{self._update_step}")
            assert result.fnc_call_out is not None
            chat_ctx.items.extend([result.fnc_call, result.fnc_call_out])
        else:
            chat_ctx.add_message(role=role, content=formatted_message)

        return self.session.generate_reply(
            chat_ctx=chat_ctx, merge_chat_ctx=True, tool_choice="none"
        )

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

    def __init__(self, *, id: str, tools: list[Tool] | None = None) -> None:
        super().__init__(id=id)
        all_tools: list[Tool] = list(tools or [])
        all_tools.extend(find_function_tools(self))
        self._tools = [self._wrap_tool(t) for t in all_tools]
        self._tasks: dict[str, _RunningTask] = {}

    @property
    def tools(self) -> list[Tool]:
        return self._tools

    @function_tool
    async def get_running_tasks(self) -> list[dict]:
        """Get the list of running async tool calls"""
        return [task.async_ctx.function_call.model_dump() for task in self._tasks.values()]

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

        # build wrapper that receives RunContext from the framework,
        # converts to AsyncRunContext, and dispatches as a background task.
        @functools.wraps(tool)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            run_ctx = arguments.pop(ctx_param_name)
            assert isinstance(run_ctx, RunContext)

            call_id = run_ctx.function_call.call_id
            if call_id in self._tasks:
                raise ValueError(f"Task already running for call_id: {call_id}")

            run_ctx.session._register_async_toolset(self)  # cleanup in session.aclose

            # inject AsyncRunContext into the function arguments
            async_ctx = AsyncRunContext(run_ctx=run_ctx)
            arguments[ctx_param_name] = async_ctx

            exe_task = asyncio.create_task(
                tool(**arguments), name=f"async_tool_{async_ctx.function_call.name}"
            )
            _pass_through_activity_task_info(exe_task)
            self._tasks[call_id] = _RunningTask(exe_task=exe_task, async_ctx=async_ctx)

            def _on_task_done(task: asyncio.Task[Any]) -> None:
                self._tasks.pop(call_id, None)
                self._on_completed(task, async_ctx)

            exe_task.add_done_callback(_on_task_done)

            return await async_ctx._pending_fut

        wrapped_tool = tool.__class__(wrapper, tool.info)  # type: ignore[arg-type]

        # replace the AsyncRunContext annotation with RunContext so the framework injects a plain RunContext
        wrapped_tool.__signature__ = sig.replace(  # type: ignore[union-attr]
            parameters=[
                p.replace(annotation=RunContext) if p.name == ctx_param_name else p
                for p in sig.parameters.values()
            ]
        )
        wrapped_tool.__annotations__ = {
            **wrapped_tool.__annotations__,
            ctx_param_name: RunContext,
        }

        return wrapped_tool

    def _on_completed(self, task: asyncio.Task[Any], async_ctx: AsyncRunContext) -> None:
        try:
            output = task.result()
        except BaseException as e:
            if isinstance(e, asyncio.CancelledError):
                logger.debug(
                    "async tool cancelled",
                    extra={
                        "call_id": async_ctx.function_call.call_id,
                        "function": async_ctx.function_call.name,
                    },
                )
            output = e

        if not (fut := async_ctx._pending_fut).done():
            # set pending fut to stop the wrapper function from waiting
            # it will create a tool output so we return from here directly
            if isinstance(output, BaseException):
                fut.set_exception(output)
            else:
                fut.set_result(output)
            return

        if isinstance(output, BaseException):
            logger.error(f"Error in async tool {async_ctx.function_call.name}", exc_info=output)

        if output is None or isinstance(output, asyncio.CancelledError):
            return

        result = async_ctx._make_tool_output(output, suffix="completed")
        if result.agent_task is not None:
            raise RuntimeError("returning Agent from async tool function is not supported")

        if result.fnc_call_out is None:
            return

        async_ctx.session.generate_reply(
            chat_ctx=ChatContext([result.fnc_call, result.fnc_call_out]),
            merge_chat_ctx=True,
        )

    async def cancel(self, call_id: str) -> bool:
        task = self._tasks.get(call_id)
        if task is not None:
            await utils.aio.cancel_and_wait(task.exe_task)
            return True
        return False

    async def shutdown(self) -> None:
        """Cancel all tasks."""
        tasks = [task.exe_task for task in self._tasks.values()]
        await utils.aio.cancel_and_wait(*tasks)
        self._tasks.clear()
