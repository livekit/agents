# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import enum
import inspect
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

from typing_extensions import ParamSpec

from ...llm import FunctionCall, FunctionCallOutput, Tool, Toolset, function_tool as _function_tool
from ...log import logger

if TYPE_CHECKING:
    from ...voice.agent_session import AgentSession
    from ...voice.events import RunContext

T = TypeVar("T")
_P = ParamSpec("_P")
_R = TypeVar("_R", bound=Awaitable[Any])


class OperationStatus(str, enum.Enum):
    """Status of an async operation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AsyncOperation(Generic[T]):
    """Represents an async operation tracked by AsyncToolset."""

    id: str
    """Unique identifier for the operation."""

    name: str
    """Name of the async function that was called."""

    status: OperationStatus
    """Current status of the operation."""

    arguments: dict[str, Any]
    """Arguments passed to the async function."""

    result: T | None = None
    """Result of the operation if completed."""

    error: str | None = None
    """Error message if the operation failed."""

    created_at: float = field(default_factory=time.time)
    """Timestamp when the operation was created."""

    completed_at: float | None = None
    """Timestamp when the operation completed (success or failure)."""

    task: asyncio.Task[T] | None = field(default=None, repr=False)
    """The asyncio task running the operation (internal)."""


class AsyncContext:
    """Context handle passed to async tool functions for controlling the conversation.

    Provides methods to set the initial pending response and push intermediate
    progress updates back to the user while the function runs in the background.

    Example::

        @async_tools.function_tool
        async def book_flight(ctx: AsyncContext, origin: str, destination: str) -> dict:
            ctx.pending(f"Looking up flights from {origin} to {destination}...")

            flights = await search_flights(origin, destination)
            ctx.update(f"Found {len(flights)} flights, selecting the best option...")

            booking = await book_best_flight(flights)
            return {"confirmation": booking.id}
    """

    def __init__(
        self,
        *,
        session: AgentSession[Any],
        operation: AsyncOperation[Any],
    ) -> None:
        self._session = session
        self._operation = operation
        self._pending_message: str | None = None
        self._pending_event = asyncio.Event()

    @property
    def session(self) -> AgentSession[Any]:
        """The agent session this operation is running in."""
        return self._session

    @property
    def userdata(self) -> Any:
        """The session's userdata."""
        return self._session.userdata

    @property
    def operation(self) -> AsyncOperation[Any]:
        """The current async operation."""
        return self._operation

    def pending(self, message: str) -> None:
        """Set the message returned to the LLM when the tool is first called.

        This must be called synchronously before the first ``await`` in your
        function. It controls what the LLM sees as the tool output, which
        determines what the agent says to the user while the operation runs.

        Args:
            message: The pending message for the LLM (e.g. "Searching flights...").
        """
        self._pending_message = message
        self._pending_event.set()

    def update(self, message: str) -> None:
        """Push an intermediate progress update into the conversation.

        Triggers a new LLM turn with the update injected as instructions,
        so the agent can narrate progress to the user.

        Args:
            message: Progress update (e.g. "Found 3 flights, selecting best...").
        """
        self._session.generate_reply(
            instructions=f"[Progress update for {self._operation.name}]: {message}"
        )


@dataclass
class _AsyncFunctionInfo:
    name: str
    description: str
    func: Callable[..., Awaitable[Any]]


class AsyncToolset(Toolset):
    """A toolset for running long-running functions in the background.

    When the LLM calls a tool registered with ``@toolset.function_tool``, the function
    runs in the background. The function receives an :class:`AsyncContext` that lets it
    control the initial response (``ctx.pending(...)``) and push progress updates
    (``ctx.update(...)``). When the function returns, the result is automatically
    pushed back into the conversation via ``session.generate_reply``.

    The toolset can be shared across multiple agents. If agent A starts an operation and
    hands off to agent B, the result is still pushed back to the active session.

    Example::

        async_tools = AsyncToolset(id="booking")

        @async_tools.function_tool
        async def book_flight(ctx: AsyncContext, origin: str, destination: str) -> dict:
            ctx.pending(f"Looking up flights from {origin} to {destination}...")

            flights = await search_flights(origin, destination)
            ctx.update(f"Found {len(flights)} flights, picking the best one...")

            booking = await book_best_flight(flights)
            return {"confirmation": booking.id}

        agent = Agent(tools=[async_tools], instructions="...")

    Conversation flow:

    1. User: "Book me a flight to Paris"
    2. LLM calls ``book_flight(origin="NYC", destination="Paris")``
    3. ``ctx.pending(...)`` → LLM says "Looking up flights from NYC to Paris..."
    4. ``ctx.update(...)`` → LLM says "Found 3 flights, picking the best one..."
    5. Function returns → LLM says "Your flight is booked! Confirmation: ABC123"
    """

    @overload
    def function_tool(
        self,
        f: Callable[_P, _R],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[_P, _R]: ...

    @overload
    def function_tool(
        self,
        f: None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...

    def function_tool(
        self,
        f: Callable[_P, _R] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[_P, _R] | Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Register an async function as a background tool.

        The function receives an :class:`AsyncContext` as its first argument
        (auto-injected, hidden from the LLM schema). Use it to control the
        pending message and push progress updates.

        Can be used with or without arguments::

            @toolset.function_tool
            async def my_func(ctx: AsyncContext, x: int) -> int: ...

            @toolset.function_tool(name="custom_name")
            async def my_func(ctx: AsyncContext, x: int) -> int: ...
        """

        def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
            if not asyncio.iscoroutinefunction(func):
                raise ValueError(f"Function {func.__name__} must be an async function")

            tool_name = name or func.__name__
            tool_description = description or func.__doc__ or f"Run {tool_name} asynchronously"

            if tool_name in self._async_functions:
                raise ValueError(f"Async function '{tool_name}' is already registered")

            self._async_functions[tool_name] = _AsyncFunctionInfo(
                name=tool_name,
                description=tool_description,
                func=func,
            )
            self._rebuild_tools()
            logger.debug(f"Registered async function: {tool_name}")
            return func

        if f is not None:
            return decorator(f)
        return decorator

    def __init__(
        self,
        *,
        id: str = "async_toolset",
    ) -> None:
        super().__init__(id=id)
        self._async_functions: dict[str, _AsyncFunctionInfo] = {}
        self._operations: dict[str, AsyncOperation[Any]] = {}
        self._tools: list[Tool] = []

    def get_operation(self, operation_id: str) -> AsyncOperation[Any] | None:
        """Get an operation by its ID."""
        return self._operations.get(operation_id)

    def get_pending_operations(self) -> list[AsyncOperation[Any]]:
        """Get all pending or running operations."""
        return [
            op
            for op in self._operations.values()
            if op.status in (OperationStatus.PENDING, OperationStatus.RUNNING)
        ]

    def get_completed_operations(self) -> list[AsyncOperation[Any]]:
        """Get all completed operations (success or failure)."""
        return [
            op
            for op in self._operations.values()
            if op.status
            in (OperationStatus.COMPLETED, OperationStatus.FAILED, OperationStatus.CANCELLED)
        ]

    def _rebuild_tools(self) -> None:
        self._tools = [self._create_tool(info) for info in self._async_functions.values()]

    def _create_tool(self, info: _AsyncFunctionInfo) -> Tool:
        """Create the LLM-facing tool that wraps the async function."""
        from typing import get_type_hints

        from ...voice.events import RunContext

        func = info.func
        sig = inspect.signature(func)
        tool_name = info.name

        try:
            type_hints = get_type_hints(func)
        except Exception:
            type_hints = {}

        async def tool_wrapper(ctx: RunContext, **kwargs: Any) -> str:  # type: ignore[type-arg]
            return await self._dispatch(tool_name, ctx, kwargs)

        # Build signature: RunContext (auto-injected) + user params (visible to LLM).
        # Exclude 'self', 'ctx', and AsyncContext params from the LLM schema.
        user_params = [
            p
            for pname, p in sig.parameters.items()
            if pname not in ("self", "ctx")
            and p.kind != inspect.Parameter.VAR_KEYWORD
            and type_hints.get(pname) is not AsyncContext
        ]
        ctx_param = inspect.Parameter(
            "ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=RunContext
        )
        tool_wrapper.__signature__ = sig.replace(  # type: ignore[attr-defined]
            parameters=[ctx_param, *user_params]
        )
        tool_wrapper.__annotations__ = {
            "ctx": RunContext,
            **{
                n: h
                for n, h in type_hints.items()
                if n not in ("self", "ctx", "return") and h is not AsyncContext
            },
            "return": str,
        }
        tool_wrapper.__doc__ = info.description

        return _function_tool(
            tool_wrapper,
            name=tool_name,
            description=info.description,
        )

    async def _dispatch(
        self,
        func_name: str,
        ctx: RunContext,  # type: ignore[type-arg]
        arguments: dict[str, Any],
    ) -> str:
        """Start the background operation and wait for the pending message."""
        info = self._async_functions.get(func_name)
        if not info:
            return f"Error: Unknown async function '{func_name}'"

        operation_id = str(uuid.uuid4())[:8]
        operation: AsyncOperation[Any] = AsyncOperation(
            id=operation_id,
            name=func_name,
            status=OperationStatus.PENDING,
            arguments=arguments,
        )
        self._operations[operation_id] = operation

        async_ctx = AsyncContext(session=ctx.session, operation=operation)

        task = asyncio.create_task(
            self._run_and_push(operation, info.func, async_ctx, arguments)
        )
        operation.task = task

        logger.debug(f"Started async operation {operation_id} for {func_name}")

        # Wait for the function to call ctx.pending(), or use a default
        try:
            await asyncio.wait_for(async_ctx._pending_event.wait(), timeout=0.1)
        except asyncio.TimeoutError:
            pass

        return async_ctx._pending_message or (
            f"{func_name} is running in the background, let the user know"
        )

    async def _run_and_push(
        self,
        operation: AsyncOperation[T],
        func: Callable[..., Awaitable[T]],
        async_ctx: AsyncContext,
        arguments: dict[str, Any],
    ) -> None:
        """Run the operation and push the result back into the conversation."""
        operation.status = OperationStatus.RUNNING
        session = async_ctx.session

        try:
            # Inject AsyncContext if the function accepts it
            sig = inspect.signature(func)
            type_hints = {}
            try:
                type_hints = __import__("typing").get_type_hints(func)
            except Exception:
                pass

            call_kwargs: dict[str, Any] = dict(arguments)
            for pname, _param in sig.parameters.items():
                if type_hints.get(pname) is AsyncContext or pname == "ctx":
                    hint = type_hints.get(pname)
                    if hint is AsyncContext:
                        call_kwargs[pname] = async_ctx
                        break

            result = await func(**call_kwargs)
            operation.result = result
            operation.status = OperationStatus.COMPLETED
            operation.completed_at = time.time()
            logger.debug(f"Operation {operation.id} completed successfully")

            self._push_result(session, operation)

        except asyncio.CancelledError:
            operation.status = OperationStatus.CANCELLED
            operation.completed_at = time.time()
            logger.debug(f"Operation {operation.id} was cancelled")

        except Exception as e:
            operation.error = str(e)
            operation.status = OperationStatus.FAILED
            operation.completed_at = time.time()
            logger.warning(f"Operation {operation.id} failed: {e}")

            self._push_result(session, operation)

    def _push_result(
        self,
        session: AgentSession[Any],
        operation: AsyncOperation[Any],
    ) -> None:
        """Inject the result into the chat context and trigger a new LLM turn."""
        call_id = f"async_{operation.id}"

        fnc_call = FunctionCall(
            call_id=call_id,
            name=operation.name,
            arguments=str(operation.arguments),
        )

        if operation.status == OperationStatus.COMPLETED:
            fnc_output = FunctionCallOutput(
                call_id=call_id,
                name=operation.name,
                output=str(operation.result),
                is_error=False,
            )
        else:
            fnc_output = FunctionCallOutput(
                call_id=call_id,
                name=operation.name,
                output=f"Error: {operation.error}",
                is_error=True,
            )

        chat_ctx = session.chat_ctx.copy()
        chat_ctx.insert(fnc_call)
        chat_ctx.insert(fnc_output)

        session.generate_reply(chat_ctx=chat_ctx)

    async def shutdown(self) -> None:
        """Cancel all pending operations."""
        for op in list(self._operations.values()):
            if op.task and not op.task.done():
                op.task.cancel()
                try:
                    await op.task
                except asyncio.CancelledError:
                    pass

        self._operations.clear()

    @property
    def tools(self) -> list[Tool]:
        return self._tools.copy()
