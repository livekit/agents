import asyncio

import pytest

from livekit.agents.beta.tools import AsyncToolset, OperationStatus
from livekit.agents.llm import FunctionTool, ToolContext

# -- helpers used across tests --

toolset = AsyncToolset(id="test_global", auto_cleanup_completed=False)


@toolset.function_tool
async def slow_operation(value: str, delay: float = 0.1) -> dict:
    """A slow operation that takes some time to complete."""
    await asyncio.sleep(delay)
    return {"value": value, "processed": True}


@toolset.function_tool
async def instant_operation(x: int, y: int) -> int:
    """An operation that completes instantly."""
    return x + y


class TestDecoratorRegistration:
    def test_bare_decorator(self):
        ts = AsyncToolset(id="bare")

        @ts.function_tool
        async def my_func(a: str) -> str:
            """Do something."""
            return a

        assert "my_func" in [f.name for f in ts._async_functions.values()]
        tool_names = [t.id for t in ts.tools]
        assert "start_my_func" in tool_names

    def test_decorator_with_args(self):
        ts = AsyncToolset(id="args")

        @ts.function_tool(name="custom_name", description="Custom desc")
        async def my_func(a: str) -> str:
            """Original docstring."""
            return a

        assert "custom_name" in [f.name for f in ts._async_functions.values()]
        tool_names = [t.id for t in ts.tools]
        assert "start_custom_name" in tool_names

    def test_decorator_preserves_function(self):
        ts = AsyncToolset(id="preserve")

        @ts.function_tool
        async def add(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        # The original function should still be callable directly
        result = asyncio.get_event_loop().run_until_complete(add(1, 2))
        assert result == 3

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
    def test_tools_property(self):
        tools = toolset.tools
        tool_names = [t.id for t in tools]

        assert "start_slow_operation" in tool_names
        assert "start_instant_operation" in tool_names
        assert "check_operation_status" in tool_names
        assert "get_operation_result" in tool_names
        assert "list_async_operations" in tool_names
        assert "cancel_async_operation" in tool_names

    def test_tool_context_integration(self):
        ctx = ToolContext([toolset])
        assert "start_slow_operation" in ctx.function_tools
        assert "check_operation_status" in ctx.function_tools
        assert toolset in ctx.toolsets

    def test_start_tool_is_function_tool(self):
        start_tool = None
        for tool in toolset.tools:
            if tool.id == "start_instant_operation":
                start_tool = tool
                break

        assert start_tool is not None
        assert isinstance(start_tool, FunctionTool)


class TestAsyncToolsetOperations:
    async def test_start_operation(self):
        ts = AsyncToolset(id="op_start", auto_cleanup_completed=False)

        @ts.function_tool
        async def slow(value: str, delay: float = 0.05) -> dict:
            await asyncio.sleep(delay)
            return {"value": value, "processed": True}

        result = await ts._start_operation("slow", {"value": "test", "delay": 0.05})

        assert "Operation started" in result
        assert len(ts._operations) == 1

        await asyncio.sleep(0.1)

        op = list(ts._operations.values())[0]
        assert op.status == OperationStatus.COMPLETED
        assert op.result == {"value": "test", "processed": True}

    async def test_start_unknown_function(self):
        ts = AsyncToolset(id="op_unknown")
        result = await ts._start_operation("unknown", {})
        assert "Error: Unknown async function" in result

    async def test_operation_failure(self):
        ts = AsyncToolset(id="op_fail", auto_cleanup_completed=False)

        @ts.function_tool
        async def fail_op(error_message: str) -> str:
            raise ValueError(error_message)

        await ts._start_operation("fail_op", {"error_message": "test error"})
        await asyncio.sleep(0.05)

        op = list(ts._operations.values())[0]
        assert op.status == OperationStatus.FAILED
        assert "test error" in op.error

    async def test_check_operation_status(self):
        ts = AsyncToolset(id="op_check", auto_cleanup_completed=False)

        @ts.function_tool
        async def slow(value: str, delay: float = 0.5) -> dict:
            await asyncio.sleep(delay)
            return {"value": value}

        result = await ts._start_operation("slow", {"value": "test", "delay": 0.5})
        op_id = result.split("ID: ")[1].split(".")[0]

        status = await ts._check_operation_status([op_id])
        assert "running" in status.lower() or "pending" in status.lower()

        await asyncio.sleep(0.6)
        status = await ts._check_operation_status([op_id])
        assert "completed" in status.lower()

    async def test_check_nonexistent_operation(self):
        ts = AsyncToolset(id="op_noexist")
        status = await ts._check_operation_status(["nonexistent"])
        assert "Not found" in status

    async def test_get_operation_result(self):
        ts = AsyncToolset(id="op_result", auto_cleanup_completed=False)

        @ts.function_tool
        async def add(x: int, y: int) -> int:
            return x + y

        result = await ts._start_operation("add", {"x": 5, "y": 3})
        op_id = result.split("ID: ")[1].split(".")[0]

        await asyncio.sleep(0.05)

        result = await ts._get_operation_result(op_id, remove_after=False)
        assert "8" in result
        assert op_id in ts._operations

        result = await ts._get_operation_result(op_id, remove_after=True)
        assert op_id not in ts._operations

    async def test_get_result_while_running(self):
        ts = AsyncToolset(id="op_running", auto_cleanup_completed=False)

        @ts.function_tool
        async def slow(value: str, delay: float = 0.5) -> dict:
            await asyncio.sleep(delay)
            return {"value": value}

        result = await ts._start_operation("slow", {"value": "test", "delay": 0.5})
        op_id = result.split("ID: ")[1].split(".")[0]

        result = await ts._get_operation_result(op_id)
        assert "still running" in result.lower() or "still pending" in result.lower()

        await ts.shutdown()

    async def test_list_operations(self):
        ts = AsyncToolset(id="op_list", auto_cleanup_completed=False)

        @ts.function_tool
        async def slow(value: str, delay: float = 0.5) -> dict:
            await asyncio.sleep(delay)
            return {"value": value}

        @ts.function_tool
        async def add(x: int, y: int) -> int:
            return x + y

        await ts._start_operation("slow", {"value": "test", "delay": 0.5})
        await ts._start_operation("add", {"x": 1, "y": 2})

        await asyncio.sleep(0.05)

        result = await ts._list_operations()
        assert "slow" in result
        assert "add" in result

        await ts.shutdown()

    async def test_list_empty_operations(self):
        ts = AsyncToolset(id="op_empty")
        result = await ts._list_operations()
        assert "No async operations" in result

    async def test_cancel_operation(self):
        ts = AsyncToolset(id="op_cancel", auto_cleanup_completed=False)

        @ts.function_tool
        async def slow(value: str, delay: float = 1.0) -> dict:
            await asyncio.sleep(delay)
            return {"value": value}

        result = await ts._start_operation("slow", {"value": "test", "delay": 1.0})
        op_id = result.split("ID: ")[1].split(".")[0]

        cancel_result = await ts._cancel_operation(op_id)
        assert "cancelled" in cancel_result.lower()

        op = ts._operations[op_id]
        assert op.status == OperationStatus.CANCELLED

    async def test_cancel_completed_operation_fails(self):
        ts = AsyncToolset(id="op_cancel_done", auto_cleanup_completed=False)

        @ts.function_tool
        async def add(x: int, y: int) -> int:
            return x + y

        result = await ts._start_operation("add", {"x": 1, "y": 2})
        op_id = result.split("ID: ")[1].split(".")[0]

        await asyncio.sleep(0.05)

        cancel_result = await ts._cancel_operation(op_id)
        assert "cannot be cancelled" in cancel_result.lower()

    async def test_cancel_nonexistent_operation(self):
        ts = AsyncToolset(id="op_cancel_noexist")
        result = await ts._cancel_operation("nonexistent")
        assert "not found" in result.lower()


class TestAsyncToolsetHelpers:
    async def test_get_operation(self):
        ts = AsyncToolset(id="helper_get", auto_cleanup_completed=False)

        @ts.function_tool
        async def add(x: int, y: int) -> int:
            return x + y

        result = await ts._start_operation("add", {"x": 1, "y": 2})
        op_id = result.split("ID: ")[1].split(".")[0]

        op = ts.get_operation(op_id)
        assert op is not None
        assert op.name == "add"
        await asyncio.sleep(0.05)

    async def test_get_pending_operations(self):
        ts = AsyncToolset(id="helper_pending", auto_cleanup_completed=False)

        @ts.function_tool
        async def slow(value: str, delay: float = 0.5) -> dict:
            await asyncio.sleep(delay)
            return {"value": value}

        await ts._start_operation("slow", {"value": "test", "delay": 0.5})

        pending = ts.get_pending_operations()
        assert len(pending) == 1

        await asyncio.sleep(0.6)

        pending = ts.get_pending_operations()
        assert len(pending) == 0

    async def test_get_completed_operations(self):
        ts = AsyncToolset(id="helper_completed", auto_cleanup_completed=False)

        @ts.function_tool
        async def add(x: int, y: int) -> int:
            return x + y

        await ts._start_operation("add", {"x": 1, "y": 2})
        await asyncio.sleep(0.05)

        completed = ts.get_completed_operations()
        assert len(completed) == 1
        assert completed[0].status == OperationStatus.COMPLETED


class TestAsyncToolsetSharing:
    async def test_shared_toolset_across_contexts(self):
        """Multiple ToolContexts can share the same AsyncToolset."""
        ts = AsyncToolset(id="shared", auto_cleanup_completed=False)

        @ts.function_tool
        async def slow(value: str, delay: float = 0.1) -> dict:
            await asyncio.sleep(delay)
            return {"value": value}

        ctx1 = ToolContext([ts])
        ctx2 = ToolContext([ts])

        assert ts in ctx1.toolsets
        assert ts in ctx2.toolsets

        # Both see the same start tool
        assert ctx1.get_function_tool("start_slow") is not None
        assert ctx2.get_function_tool("start_slow") is not None

        result = await ts._start_operation("slow", {"value": "shared", "delay": 0.1})
        op_id = result.split("ID: ")[1].split(".")[0]

        await asyncio.sleep(0.15)
        op = ts.get_operation(op_id)
        assert op is not None
        assert op.status == OperationStatus.COMPLETED


class TestAsyncToolsetShutdown:
    async def test_shutdown(self):
        ts = AsyncToolset(id="shutdown_test", auto_cleanup_completed=False)

        @ts.function_tool
        async def slow(value: str, delay: float = 10.0) -> dict:
            await asyncio.sleep(delay)
            return {"value": value}

        await ts._start_operation("slow", {"value": "test", "delay": 10.0})
        assert len(ts.get_pending_operations()) == 1

        await ts.shutdown()
        assert len(ts._operations) == 0
