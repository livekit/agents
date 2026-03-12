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
from typing import Annotated, Any, Generic, TypeVar, overload

from pydantic import Field
from typing_extensions import ParamSpec

from ...llm import Tool, Toolset, function_tool as _function_tool
from ...log import logger

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
    """Information about a registered async function."""

    name: str
    description: str
    func: Callable[..., Awaitable[Any]]


class AsyncToolset(Toolset):
    """A toolset for managing async/long-running function calls.

    AsyncToolset allows you to register functions that run in the background and
    can be checked/monitored by the LLM. This is useful for long-running operations
    like API calls, database queries, or any task that shouldn't block the conversation.

    The toolset exposes tools to the LLM for each registered function (prefixed with
    ``start_``), plus management tools to check status, get results, list, and cancel
    operations.

    Multiple agents can share the same AsyncToolset instance, allowing "execution
    ownership" to be shared -- one agent can start an operation, another can check
    its status or retrieve the result.

    Example:
        ```python
        async_tools = AsyncToolset(id="booking_tools")

        @async_tools.function_tool
        async def book_flight(origin: str, destination: str) -> dict:
            '''Book a flight from origin to destination.'''
            await asyncio.sleep(5)  # Simulate long operation
            return {"confirmation": "ABC123", "origin": origin, "destination": destination}

        # Multiple agents can share this toolset
        agent1 = Agent(tools=[async_tools], ...)
        agent2 = Agent(tools=[async_tools], ...)
        ```
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
        """Decorator to register an async function as a background tool.

        Can be used with or without arguments::

            @toolset.function_tool
            async def my_func(x: int) -> int: ...

            @toolset.function_tool(name="custom_name", description="Custom desc")
            async def my_func(x: int) -> int: ...

        Args:
            f: The async function to register (when used without parentheses).
            name: Override the tool name (defaults to the function name).
            description: Override the tool description (defaults to docstring).
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
        auto_cleanup_completed: bool = True,
        cleanup_after_seconds: float = 300.0,
    ) -> None:
        """Initialize the AsyncToolset.

        Args:
            id: Unique identifier for this toolset.
            auto_cleanup_completed: Whether to automatically clean up completed
                operations after cleanup_after_seconds.
            cleanup_after_seconds: How long to keep completed operations before
                cleaning them up (default 5 minutes).
        """
        super().__init__(id=id)
        self._async_functions: dict[str, _AsyncFunctionInfo] = {}
        self._operations: dict[str, AsyncOperation[Any]] = {}
        self._tools: list[Tool] = []
        self._auto_cleanup = auto_cleanup_completed
        self._cleanup_after = cleanup_after_seconds
        self._cleanup_task: asyncio.Task[None] | None = None

        # Build management tools
        self._check_status_tool = _function_tool(
            self._check_operation_status,
            name="check_operation_status",
            description=(
                "Check the status of one or more async operations by their IDs. "
                "Returns the current status (pending, running, completed, failed, cancelled) "
                "and any available results or error messages."
            ),
        )

        self._get_result_tool = _function_tool(
            self._get_operation_result,
            name="get_operation_result",
            description=(
                "Get the result of a completed async operation. "
                "Use this after check_operation_status shows the operation is completed. "
                "Optionally remove the operation from tracking after retrieving the result."
            ),
        )

        self._list_operations_tool = _function_tool(
            self._list_operations,
            name="list_async_operations",
            description=(
                "List all pending or recently completed async operations. "
                "Useful for getting an overview of all operations in progress."
            ),
        )

        self._cancel_operation_tool = _function_tool(
            self._cancel_operation,
            name="cancel_async_operation",
            description=(
                "Cancel a pending or running async operation. "
                "Returns whether the cancellation was successful."
            ),
        )

        self._rebuild_tools()

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
        """Rebuild the list of tools after registration changes."""
        self._tools = []

        for info in self._async_functions.values():
            self._tools.append(self._create_start_tool(info))

        self._tools.extend(
            [
                self._check_status_tool,
                self._get_result_tool,
                self._list_operations_tool,
                self._cancel_operation_tool,
            ]
        )

    def _create_start_tool(self, info: _AsyncFunctionInfo) -> Tool:
        """Create a start tool for an async function."""
        from typing import get_type_hints

        func = info.func
        sig = inspect.signature(func)

        try:
            type_hints = get_type_hints(func)
        except Exception:
            type_hints = {}

        async def start_wrapper(**kwargs: Any) -> str:
            return await self._start_operation(info.name, kwargs)

        # Preserve the signature from the original function for schema generation
        params = [
            p
            for pname, p in sig.parameters.items()
            if pname not in ("self", "ctx") and p.kind != inspect.Parameter.VAR_KEYWORD
        ]
        start_wrapper.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
        start_wrapper.__annotations__ = {
            n: hint for n, hint in type_hints.items() if n not in ("self", "ctx", "return")
        }
        start_wrapper.__annotations__["return"] = str
        start_wrapper.__doc__ = (
            f"{info.description}\n\n"
            "This starts the operation in the background and returns an operation ID. "
            "Use check_operation_status to monitor progress and get_operation_result "
            "to retrieve the final result."
        )

        return _function_tool(
            start_wrapper,
            name=f"start_{info.name}",
            description=(
                f"Start async operation: {info.description} "
                "Returns an operation ID to track the background task."
            ),
        )

    async def _start_operation(self, func_name: str, arguments: dict[str, Any]) -> str:
        """Start an async operation and return its ID."""
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

        task = asyncio.create_task(self._run_operation(operation, info.func))
        operation.task = task

        logger.debug(f"Started async operation {operation_id} for {func_name}")

        if self._auto_cleanup and self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        return (
            f"Operation started with ID: {operation_id}. "
            f"Use check_operation_status('{operation_id}') to monitor progress."
        )

    async def _run_operation(
        self, operation: AsyncOperation[T], func: Callable[..., Awaitable[T]]
    ) -> T:
        """Run the async operation."""
        operation.status = OperationStatus.RUNNING

        try:
            result = await func(**operation.arguments)
            operation.result = result
            operation.status = OperationStatus.COMPLETED
            operation.completed_at = time.time()
            logger.debug(f"Operation {operation.id} completed successfully")
            return result
        except asyncio.CancelledError:
            operation.status = OperationStatus.CANCELLED
            operation.completed_at = time.time()
            logger.debug(f"Operation {operation.id} was cancelled")
            raise
        except Exception as e:
            operation.error = str(e)
            operation.status = OperationStatus.FAILED
            operation.completed_at = time.time()
            logger.warning(f"Operation {operation.id} failed: {e}")
            raise

    async def _check_operation_status(
        self,
        operation_ids: Annotated[
            list[str],
            Field(description="List of operation IDs to check status for"),
        ],
    ) -> str:
        """Check the status of one or more operations."""
        if not operation_ids:
            return "No operation IDs provided"

        results = []
        for op_id in operation_ids:
            operation = self._operations.get(op_id)
            if not operation:
                results.append(f"Operation {op_id}: Not found")
                continue

            status_info = f"Operation {op_id} ({operation.name}): {operation.status.value}"
            if operation.status == OperationStatus.COMPLETED:
                status_info += " - Result available, use get_operation_result to retrieve"
            elif operation.status == OperationStatus.FAILED:
                status_info += f" - Error: {operation.error}"
            elif operation.status == OperationStatus.RUNNING:
                elapsed = time.time() - operation.created_at
                status_info += f" - Running for {elapsed:.1f}s"

            results.append(status_info)

        return "\n".join(results)

    async def _get_operation_result(
        self,
        operation_id: Annotated[
            str,
            Field(description="The ID of the operation to get the result for"),
        ],
        remove_after: Annotated[
            bool,
            Field(
                description="Whether to remove the operation from tracking after retrieving. "
                "Defaults to True.",
                default=True,
            ),
        ] = True,
    ) -> str:
        """Get the result of a completed operation."""
        operation = self._operations.get(operation_id)
        if not operation:
            return f"Operation {operation_id} not found"

        if operation.status == OperationStatus.PENDING:
            return f"Operation {operation_id} is still pending"
        elif operation.status == OperationStatus.RUNNING:
            elapsed = time.time() - operation.created_at
            return f"Operation {operation_id} is still running ({elapsed:.1f}s elapsed)"
        elif operation.status == OperationStatus.CANCELLED:
            if remove_after:
                del self._operations[operation_id]
            return f"Operation {operation_id} was cancelled"
        elif operation.status == OperationStatus.FAILED:
            error_msg = f"Operation {operation_id} failed: {operation.error}"
            if remove_after:
                del self._operations[operation_id]
            return error_msg
        elif operation.status == OperationStatus.COMPLETED:
            result = operation.result
            if remove_after:
                del self._operations[operation_id]
            return f"Operation {operation_id} result: {result}"

        return f"Unknown status for operation {operation_id}"

    async def _list_operations(self) -> str:
        """List all tracked operations."""
        if not self._operations:
            return "No async operations are currently tracked"

        lines = ["Current async operations:"]
        for op in self._operations.values():
            elapsed = time.time() - op.created_at
            status_detail = ""
            if op.status == OperationStatus.COMPLETED:
                status_detail = " (result available)"
            elif op.status == OperationStatus.FAILED:
                status_detail = f" (error: {op.error})"
            elif op.status == OperationStatus.RUNNING:
                status_detail = f" (running {elapsed:.1f}s)"

            lines.append(f"- {op.id}: {op.name} - {op.status.value}{status_detail}")

        return "\n".join(lines)

    async def _cancel_operation(
        self,
        operation_id: Annotated[
            str,
            Field(description="The ID of the operation to cancel"),
        ],
    ) -> str:
        """Cancel a pending or running operation."""
        operation = self._operations.get(operation_id)
        if not operation:
            return f"Operation {operation_id} not found"

        if operation.status not in (OperationStatus.PENDING, OperationStatus.RUNNING):
            return (
                f"Operation {operation_id} cannot be cancelled "
                f"(status: {operation.status.value})"
            )

        if operation.task and not operation.task.done():
            operation.task.cancel()
            try:
                await operation.task
            except asyncio.CancelledError:
                pass

        operation.status = OperationStatus.CANCELLED
        operation.completed_at = time.time()

        return f"Operation {operation_id} has been cancelled"

    async def _cleanup_loop(self) -> None:
        """Background task to clean up old completed operations."""
        try:
            while True:
                await asyncio.sleep(60)
                current_time = time.time()
                to_remove = []

                for op_id, op in self._operations.items():
                    if op.completed_at is not None:
                        age = current_time - op.completed_at
                        if age > self._cleanup_after:
                            to_remove.append(op_id)

                for op_id in to_remove:
                    del self._operations[op_id]
                    logger.debug(f"Cleaned up old operation: {op_id}")

                if not self._operations:
                    self._cleanup_task = None
                    break

        except asyncio.CancelledError:
            pass

    async def shutdown(self) -> None:
        """Shutdown the toolset and cancel all pending operations."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        for op in self._operations.values():
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
