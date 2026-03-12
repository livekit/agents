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


@dataclass
class _AsyncFunctionInfo:
    name: str
    description: str
    func: Callable[..., Awaitable[Any]]


class AsyncToolset(Toolset):
    """A toolset for running long-running functions in the background.

    When the LLM calls a tool registered with ``@toolset.function_tool``, the function
    runs in the background and the LLM gets an immediate response so it can keep talking.
    When the background operation completes, the result is **automatically pushed back**
    into the conversation via ``session.generate_reply`` -- no polling needed.

    The toolset can be shared across multiple agents. If agent A starts an operation and
    hands off to agent B, the result is still pushed back to the active session when ready.

    Example::

        async_tools = AsyncToolset(id="booking")

        @async_tools.function_tool
        async def book_flight(origin: str, destination: str) -> dict:
            '''Book a flight (takes ~5s).'''
            result = await booking_api.book(origin, destination)
            return {"confirmation": result.id}

        agent = Agent(tools=[async_tools], instructions="...")

    Conversation flow:

    1. User: "Book me a flight to Paris"
    2. LLM calls ``book_flight(origin="NYC", destination="Paris")``
    3. Tool returns immediately → LLM says "I'm booking that for you..."
    4. User can keep talking naturally
    5. Background task completes → result is pushed back
    6. LLM says "Your flight is booked! Confirmation: ABC123"
    """

    @overload
    def function_tool(
        self,
        f: Callable[_P, _R],
        *,
        name: str | None = None,
        description: str | None = None,
        pending_message: str | None = None,
    ) -> Callable[_P, _R]: ...

    @overload
    def function_tool(
        self,
        f: None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        pending_message: str | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...

    def function_tool(
        self,
        f: Callable[_P, _R] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        pending_message: str | None = None,
    ) -> Callable[_P, _R] | Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Register an async function as a background tool.

        Can be used with or without arguments::

            @toolset.function_tool
            async def my_func(x: int) -> int: ...

            @toolset.function_tool(name="custom_name", pending_message="Working on it...")
            async def my_func(x: int) -> int: ...

        Args:
            f: The async function (when used as bare decorator).
            name: Override tool name (defaults to function name).
            description: Override tool description (defaults to docstring).
            pending_message: Message returned to LLM while the operation runs.
                Defaults to "{name} is running in the background, let the user know".
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
            self._pending_messages[tool_name] = pending_message
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
        self._pending_messages: dict[str, str | None] = {}
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
        pending_msg = self._pending_messages.get(tool_name)

        try:
            type_hints = get_type_hints(func)
        except Exception:
            type_hints = {}

        # The wrapper accepts the original params + RunContext (auto-injected by framework)
        async def tool_wrapper(ctx: RunContext, **kwargs: Any) -> str:  # type: ignore[type-arg]
            return await self._dispatch(tool_name, ctx, kwargs, pending_msg)

        # Preserve the original signature for LLM schema generation.
        # RunContext params are auto-excluded by the framework.
        user_params = [
            p
            for pname, p in sig.parameters.items()
            if pname not in ("self", "ctx") and p.kind != inspect.Parameter.VAR_KEYWORD
        ]
        ctx_param = inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=RunContext)
        tool_wrapper.__signature__ = sig.replace(parameters=[ctx_param, *user_params])  # type: ignore[attr-defined]
        tool_wrapper.__annotations__ = {
            "ctx": RunContext,
            **{n: h for n, h in type_hints.items() if n not in ("self", "ctx", "return")},
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
        pending_message: str | None,
    ) -> str:
        """Start the background operation and return immediately."""
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

        task = asyncio.create_task(
            self._run_and_push(operation, info.func, ctx)
        )
        operation.task = task

        logger.debug(f"Started async operation {operation_id} for {func_name}")

        return pending_message or (
            f"{func_name} is running in the background, let the user know"
        )

    async def _run_and_push(
        self,
        operation: AsyncOperation[T],
        func: Callable[..., Awaitable[T]],
        ctx: RunContext,  # type: ignore[type-arg]
    ) -> None:
        """Run the operation and push the result back into the conversation."""
        operation.status = OperationStatus.RUNNING
        session = ctx.session

        try:
            result = await func(**operation.arguments)
            operation.result = result
            operation.status = OperationStatus.COMPLETED
            operation.completed_at = time.time()
            logger.debug(f"Operation {operation.id} completed successfully")

            # Push result back into the conversation
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

            # Push error back into the conversation
            self._push_result(session, operation)

    def _push_result(
        self,
        session: Any,
        operation: AsyncOperation[Any],
    ) -> None:
        """Inject the result into the chat context and trigger a new LLM turn."""
        call_id = f"async_{operation.id}"

        # Create function call + output pair in the chat context
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

        # Inject into the agent's chat context and trigger a new LLM turn
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
