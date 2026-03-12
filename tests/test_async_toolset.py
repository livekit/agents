import asyncio
from unittest.mock import MagicMock

import pytest

from livekit.agents.beta.tools import AsyncToolset, OperationStatus
from livekit.agents.llm import FunctionTool, ToolContext


class TestDecoratorRegistration:
    def test_bare_decorator(self):
        ts = AsyncToolset(id="bare")

        @ts.function_tool
        async def my_func(a: str) -> str:
            """Do something."""
            return a

        assert "my_func" in ts._async_functions
        tool_names = [t.id for t in ts.tools]
        assert "my_func" in tool_names

    def test_decorator_with_args(self):
        ts = AsyncToolset(id="args")

        @ts.function_tool(name="custom_name", description="Custom desc")
        async def my_func(a: str) -> str:
            """Original docstring."""
            return a

        assert "custom_name" in ts._async_functions
        tool_names = [t.id for t in ts.tools]
        assert "custom_name" in tool_names

    def test_decorator_preserves_function(self):
        ts = AsyncToolset(id="preserve")

        @ts.function_tool
        async def add(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        # The original function should still be directly callable
        result = asyncio.get_event_loop().run_until_complete(add(1, 2))
        assert result == 3

    def test_custom_pending_message(self):
        ts = AsyncToolset(id="pending")

        @ts.function_tool(pending_message="Hold tight, booking your flight...")
        async def book_flight(dest: str) -> str:
            return f"booked {dest}"

        assert ts._pending_messages["book_flight"] == "Hold tight, booking your flight..."

    def test_sync_function_fails(self):
        ts = AsyncToolset(id="sync_fail")

        with pytest.raises(ValueError, match="must be an async function"):

            @ts.function_tool
            def sync_func() -> str:
                return "sync"

    def test_duplicate_name_fails(self):
        ts = AsyncToolset(id="dup")

        @ts.function_tool
        async def my_func() -> str:
            return "first"

        with pytest.raises(ValueError, match="already registered"):

            @ts.function_tool(name="my_func")
            async def another_func() -> str:
                return "second"


class TestAsyncToolsetTools:
    def test_tools_use_original_name(self):
        """Tools should use the function name directly, no start_ prefix."""
        ts = AsyncToolset(id="names")

        @ts.function_tool
        async def book_flight(dest: str) -> str:
            return dest

        @ts.function_tool
        async def search_hotels(city: str) -> str:
            return city

        tool_names = [t.id for t in ts.tools]
        assert "book_flight" in tool_names
        assert "search_hotels" in tool_names
        assert "start_book_flight" not in tool_names

    def test_tool_context_integration(self):
        ts = AsyncToolset(id="ctx")

        @ts.function_tool
        async def my_tool(x: int) -> int:
            return x

        ctx = ToolContext([ts])
        assert "my_tool" in ctx.function_tools
        assert ts in ctx.toolsets

    def test_tool_is_function_tool(self):
        ts = AsyncToolset(id="type")

        @ts.function_tool
        async def my_tool(x: int) -> int:
            return x

        tool = ts.tools[0]
        assert isinstance(tool, FunctionTool)

    def test_no_polling_tools(self):
        """Push model means no check_status/get_result/list/cancel tools."""
        ts = AsyncToolset(id="no_poll")

        @ts.function_tool
        async def my_tool(x: int) -> int:
            return x

        tool_names = [t.id for t in ts.tools]
        assert "check_operation_status" not in tool_names
        assert "get_operation_result" not in tool_names
        assert "list_async_operations" not in tool_names
        assert "cancel_async_operation" not in tool_names


def _make_mock_ctx():
    """Create a mock RunContext with a mock session."""
    session = MagicMock()
    session.chat_ctx = MagicMock()
    session.chat_ctx.copy.return_value = MagicMock()
    session.generate_reply = MagicMock()

    ctx = MagicMock()
    ctx.session = session
    return ctx


class TestAsyncToolsetDispatch:
    async def test_dispatch_returns_pending_message(self):
        ts = AsyncToolset(id="dispatch")

        @ts.function_tool(pending_message="Working on it...")
        async def slow_task(value: str) -> str:
            await asyncio.sleep(10)
            return value

        ctx = _make_mock_ctx()
        result = await ts._dispatch("slow_task", ctx, {"value": "test"}, "Working on it...")

        assert result == "Working on it..."
        assert len(ts._operations) == 1

        await ts.shutdown()

    async def test_dispatch_default_pending_message(self):
        ts = AsyncToolset(id="default_msg")

        @ts.function_tool
        async def my_task(x: int) -> int:
            await asyncio.sleep(10)
            return x

        ctx = _make_mock_ctx()
        result = await ts._dispatch("my_task", ctx, {"x": 1}, None)

        assert "my_task" in result
        assert "background" in result

        await ts.shutdown()

    async def test_dispatch_unknown_function(self):
        ts = AsyncToolset(id="unknown")
        ctx = _make_mock_ctx()
        result = await ts._dispatch("nonexistent", ctx, {}, None)
        assert "Error" in result

    async def test_operation_completes_and_pushes_result(self):
        ts = AsyncToolset(id="push")

        @ts.function_tool
        async def fast_task(value: str) -> dict:
            return {"result": value}

        ctx = _make_mock_ctx()
        await ts._dispatch("fast_task", ctx, {"value": "hello"}, None)

        # Wait for background task to complete
        await asyncio.sleep(0.05)

        op = list(ts._operations.values())[0]
        assert op.status == OperationStatus.COMPLETED
        assert op.result == {"result": "hello"}

        # Should have called generate_reply to push the result back
        ctx.session.generate_reply.assert_called_once()

    async def test_operation_failure_pushes_error(self):
        ts = AsyncToolset(id="fail_push")

        @ts.function_tool
        async def failing_task(msg: str) -> str:
            raise ValueError(msg)

        ctx = _make_mock_ctx()
        await ts._dispatch("failing_task", ctx, {"msg": "boom"}, None)

        await asyncio.sleep(0.05)

        op = list(ts._operations.values())[0]
        assert op.status == OperationStatus.FAILED
        assert "boom" in op.error

        # Error should also be pushed back
        ctx.session.generate_reply.assert_called_once()

    async def test_cancelled_operation_does_not_push(self):
        ts = AsyncToolset(id="cancel_no_push")

        @ts.function_tool
        async def slow_task(value: str) -> str:
            await asyncio.sleep(10)
            return value

        ctx = _make_mock_ctx()
        await ts._dispatch("slow_task", ctx, {"value": "test"}, None)

        # Cancel immediately
        await ts.shutdown()

        # Cancelled operations should NOT push a reply
        ctx.session.generate_reply.assert_not_called()


class TestAsyncToolsetOperationTracking:
    async def test_get_operation(self):
        ts = AsyncToolset(id="track_get")

        @ts.function_tool
        async def my_task(x: int) -> int:
            return x

        ctx = _make_mock_ctx()
        await ts._dispatch("my_task", ctx, {"x": 42}, None)
        await asyncio.sleep(0.05)

        ops = list(ts._operations.values())
        assert len(ops) == 1

        op = ts.get_operation(ops[0].id)
        assert op is not None
        assert op.name == "my_task"
        assert op.status == OperationStatus.COMPLETED

    async def test_get_pending_operations(self):
        ts = AsyncToolset(id="track_pending")

        @ts.function_tool
        async def slow_task(value: str) -> str:
            await asyncio.sleep(10)
            return value

        ctx = _make_mock_ctx()
        await ts._dispatch("slow_task", ctx, {"value": "test"}, None)

        pending = ts.get_pending_operations()
        assert len(pending) == 1

        await ts.shutdown()

        pending = ts.get_pending_operations()
        assert len(pending) == 0

    async def test_get_completed_operations(self):
        ts = AsyncToolset(id="track_completed")

        @ts.function_tool
        async def fast_task(x: int) -> int:
            return x

        ctx = _make_mock_ctx()
        await ts._dispatch("fast_task", ctx, {"x": 1}, None)
        await asyncio.sleep(0.05)

        completed = ts.get_completed_operations()
        assert len(completed) == 1
        assert completed[0].status == OperationStatus.COMPLETED


class TestAsyncToolsetSharing:
    async def test_shared_across_contexts(self):
        """Multiple ToolContexts share the same toolset and operations."""
        ts = AsyncToolset(id="shared")

        @ts.function_tool
        async def shared_task(value: str) -> str:
            return value

        ctx1 = ToolContext([ts])
        ctx2 = ToolContext([ts])

        assert ts in ctx1.toolsets
        assert ts in ctx2.toolsets
        assert ctx1.get_function_tool("shared_task") is not None
        assert ctx2.get_function_tool("shared_task") is not None

    async def test_shared_operation_state(self):
        """Operations started via one context are visible to another."""
        ts = AsyncToolset(id="shared_ops")

        @ts.function_tool
        async def task(value: str) -> str:
            return value

        ctx = _make_mock_ctx()
        await ts._dispatch("task", ctx, {"value": "shared"}, None)
        await asyncio.sleep(0.05)

        # Any agent with this toolset sees the operation
        assert len(ts.get_completed_operations()) == 1


class TestAsyncToolsetShutdown:
    async def test_shutdown_cancels_pending(self):
        ts = AsyncToolset(id="shutdown")

        @ts.function_tool
        async def slow_task(value: str) -> str:
            await asyncio.sleep(10)
            return value

        ctx = _make_mock_ctx()
        await ts._dispatch("slow_task", ctx, {"value": "test"}, None)
        assert len(ts.get_pending_operations()) == 1

        await ts.shutdown()
        assert len(ts._operations) == 0

    async def test_shutdown_is_idempotent(self):
        ts = AsyncToolset(id="shutdown_idem")
        await ts.shutdown()
        await ts.shutdown()
