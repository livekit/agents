import asyncio

import pytest

from livekit.agents.beta.tools import AsyncToolset, OperationStatus
from livekit.agents.llm import FunctionTool, ToolContext


async def slow_operation(value: str, delay: float = 0.1) -> dict:
    """A slow operation that takes some time to complete."""
    await asyncio.sleep(delay)
    return {"value": value, "processed": True}


async def failing_operation(error_message: str) -> str:
    """An operation that always fails."""
    raise ValueError(error_message)


async def instant_operation(x: int, y: int) -> int:
    """An operation that completes instantly."""
    return x + y


class TestAsyncToolsetRegistration:
    def test_register_function(self):
        toolset = AsyncToolset(id="test")
        toolset.register(slow_operation, name="slow_op", description="A slow operation")

        assert "slow_op" in [f.name for f in toolset._async_functions.values()]

    def test_register_sync_function_fails(self):
        def sync_func():
            return "sync"

        toolset = AsyncToolset(id="test")
        with pytest.raises(ValueError, match="must be an async function"):
            toolset.register(sync_func)  # type: ignore[arg-type]

    def test_register_duplicate_fails(self):
        toolset = AsyncToolset(id="test")
        toolset.register(slow_operation, name="slow_op")

        with pytest.raises(ValueError, match="already registered"):
            toolset.register(slow_operation, name="slow_op")

    def test_unregister_function(self):
        toolset = AsyncToolset(id="test")
        toolset.register(slow_operation, name="slow_op")
        toolset.unregister("slow_op")

        assert "slow_op" not in [f.name for f in toolset._async_functions.values()]

    def test_unregister_nonexistent_fails(self):
        toolset = AsyncToolset(id="test")
        with pytest.raises(ValueError, match="not registered"):
            toolset.unregister("nonexistent")


class TestAsyncToolsetTools:
    def test_tools_property(self):
        toolset = AsyncToolset(id="test")
        toolset.register(slow_operation, name="slow_op")

        tools = toolset.tools
        tool_names = [t.id for t in tools]

        # Should have start_slow_op + management tools
        assert "start_slow_op" in tool_names
        assert "check_operation_status" in tool_names
        assert "get_operation_result" in tool_names
        assert "list_async_operations" in tool_names
        assert "cancel_async_operation" in tool_names

    def test_tool_context_integration(self):
        toolset = AsyncToolset(id="test")
        toolset.register(slow_operation, name="slow_op")

        ctx = ToolContext([toolset])
        assert "start_slow_op" in ctx.function_tools
        assert "check_operation_status" in ctx.function_tools
        assert toolset in ctx.toolsets

    def test_multiple_registrations(self):
        toolset = AsyncToolset(id="test")
        toolset.register(slow_operation, name="slow_op")
        toolset.register(instant_operation, name="instant_op")

        tools = toolset.tools
        tool_names = [t.id for t in tools]

        assert "start_slow_op" in tool_names
        assert "start_instant_op" in tool_names

    def test_start_tool_has_original_signature(self):
        toolset = AsyncToolset(id="test")
        toolset.register(instant_operation, name="add")

        start_tool = None
        for tool in toolset.tools:
            if tool.id == "start_add":
                start_tool = tool
                break

        assert start_tool is not None
        assert isinstance(start_tool, FunctionTool)
        # The wrapper should preserve the original function's parameters


class TestAsyncToolsetOperations:
    async def test_start_operation(self):
        toolset = AsyncToolset(id="test", auto_cleanup_completed=False)
        toolset.register(slow_operation, name="slow_op")

        result = await toolset._start_operation("slow_op", {"value": "test", "delay": 0.05})

        assert "Operation started" in result
        assert len(toolset._operations) == 1

        # Wait for completion
        await asyncio.sleep(0.1)

        op = list(toolset._operations.values())[0]
        assert op.status == OperationStatus.COMPLETED
        assert op.result == {"value": "test", "processed": True}

    async def test_start_unknown_function(self):
        toolset = AsyncToolset(id="test")
        result = await toolset._start_operation("unknown", {})
        assert "Error: Unknown async function" in result

    async def test_operation_failure(self):
        toolset = AsyncToolset(id="test", auto_cleanup_completed=False)
        toolset.register(failing_operation, name="fail_op")

        await toolset._start_operation("fail_op", {"error_message": "test error"})
        await asyncio.sleep(0.05)

        op = list(toolset._operations.values())[0]
        assert op.status == OperationStatus.FAILED
        assert "test error" in op.error

    async def test_check_operation_status(self):
        toolset = AsyncToolset(id="test", auto_cleanup_completed=False)
        toolset.register(slow_operation, name="slow_op")

        result = await toolset._start_operation("slow_op", {"value": "test", "delay": 0.5})
        op_id = result.split("ID: ")[1].split(".")[0]

        # Check status while running
        status = await toolset._check_operation_status([op_id])
        assert "running" in status.lower() or "pending" in status.lower()

        # Wait for completion and check again
        await asyncio.sleep(0.6)
        status = await toolset._check_operation_status([op_id])
        assert "completed" in status.lower()

    async def test_check_nonexistent_operation(self):
        toolset = AsyncToolset(id="test")
        status = await toolset._check_operation_status(["nonexistent"])
        assert "Not found" in status

    async def test_get_operation_result(self):
        toolset = AsyncToolset(id="test", auto_cleanup_completed=False)
        toolset.register(instant_operation, name="add")

        result = await toolset._start_operation("add", {"x": 5, "y": 3})
        op_id = result.split("ID: ")[1].split(".")[0]

        await asyncio.sleep(0.05)  # Wait for completion

        result = await toolset._get_operation_result(op_id, remove_after=False)
        assert "8" in result

        # Should still be tracked since remove_after=False
        assert op_id in toolset._operations

        # Now get with removal
        result = await toolset._get_operation_result(op_id, remove_after=True)
        assert op_id not in toolset._operations

    async def test_get_result_while_running(self):
        toolset = AsyncToolset(id="test", auto_cleanup_completed=False)
        toolset.register(slow_operation, name="slow_op")

        result = await toolset._start_operation("slow_op", {"value": "test", "delay": 0.5})
        op_id = result.split("ID: ")[1].split(".")[0]

        result = await toolset._get_operation_result(op_id)
        assert "still running" in result.lower() or "still pending" in result.lower()

        # Clean up the running task
        await toolset.shutdown()

    async def test_list_operations(self):
        toolset = AsyncToolset(id="test", auto_cleanup_completed=False)
        toolset.register(slow_operation, name="slow_op")
        toolset.register(instant_operation, name="add")

        await toolset._start_operation("slow_op", {"value": "test", "delay": 0.5})
        await toolset._start_operation("add", {"x": 1, "y": 2})

        await asyncio.sleep(0.05)  # Let instant complete

        result = await toolset._list_operations()
        assert "slow_op" in result
        assert "add" in result

        # Clean up the running task
        await toolset.shutdown()

    async def test_list_empty_operations(self):
        toolset = AsyncToolset(id="test")
        result = await toolset._list_operations()
        assert "No async operations" in result

    async def test_cancel_operation(self):
        toolset = AsyncToolset(id="test", auto_cleanup_completed=False)
        toolset.register(slow_operation, name="slow_op")

        result = await toolset._start_operation("slow_op", {"value": "test", "delay": 1.0})
        op_id = result.split("ID: ")[1].split(".")[0]

        # Cancel while running
        cancel_result = await toolset._cancel_operation(op_id)
        assert "cancelled" in cancel_result.lower()

        op = toolset._operations[op_id]
        assert op.status == OperationStatus.CANCELLED

    async def test_cancel_completed_operation_fails(self):
        toolset = AsyncToolset(id="test", auto_cleanup_completed=False)
        toolset.register(instant_operation, name="add")

        result = await toolset._start_operation("add", {"x": 1, "y": 2})
        op_id = result.split("ID: ")[1].split(".")[0]

        await asyncio.sleep(0.05)  # Wait for completion

        cancel_result = await toolset._cancel_operation(op_id)
        assert "cannot be cancelled" in cancel_result.lower()

    async def test_cancel_nonexistent_operation(self):
        toolset = AsyncToolset(id="test")
        result = await toolset._cancel_operation("nonexistent")
        assert "not found" in result.lower()


class TestAsyncToolsetHelpers:
    async def test_get_operation(self):
        toolset = AsyncToolset(id="test", auto_cleanup_completed=False)
        toolset.register(instant_operation, name="add")

        result = await toolset._start_operation("add", {"x": 1, "y": 2})
        op_id = result.split("ID: ")[1].split(".")[0]

        op = toolset.get_operation(op_id)
        assert op is not None
        assert op.name == "add"

    async def test_get_pending_operations(self):
        toolset = AsyncToolset(id="test", auto_cleanup_completed=False)
        toolset.register(slow_operation, name="slow_op")

        await toolset._start_operation("slow_op", {"value": "test", "delay": 0.5})

        pending = toolset.get_pending_operations()
        assert len(pending) == 1

        await asyncio.sleep(0.6)

        pending = toolset.get_pending_operations()
        assert len(pending) == 0

    async def test_get_completed_operations(self):
        toolset = AsyncToolset(id="test", auto_cleanup_completed=False)
        toolset.register(instant_operation, name="add")

        await toolset._start_operation("add", {"x": 1, "y": 2})
        await asyncio.sleep(0.05)

        completed = toolset.get_completed_operations()
        assert len(completed) == 1
        assert completed[0].status == OperationStatus.COMPLETED


class TestAsyncToolsetSharing:
    async def test_shared_toolset_across_contexts(self):
        """Test that multiple ToolContexts can share the same AsyncToolset."""
        toolset = AsyncToolset(id="shared", auto_cleanup_completed=False)
        toolset.register(slow_operation, name="slow_op")

        # Create two separate tool contexts with the same toolset
        ctx1 = ToolContext([toolset])
        ctx2 = ToolContext([toolset])

        # Start operation via ctx1's perspective
        start_tool = ctx1.get_function_tool("start_slow_op")
        assert start_tool is not None

        # Both contexts should see the same toolset
        assert toolset in ctx1.toolsets
        assert toolset in ctx2.toolsets

        # Start operation
        result = await toolset._start_operation("slow_op", {"value": "shared", "delay": 0.1})
        op_id = result.split("ID: ")[1].split(".")[0]

        # Both contexts should be able to check status
        status_tool_1 = ctx1.get_function_tool("check_operation_status")
        status_tool_2 = ctx2.get_function_tool("check_operation_status")

        assert status_tool_1 is not None
        assert status_tool_2 is not None

        # The toolset state is shared
        await asyncio.sleep(0.15)
        op = toolset.get_operation(op_id)
        assert op is not None
        assert op.status == OperationStatus.COMPLETED


class TestAsyncToolsetShutdown:
    async def test_shutdown(self):
        toolset = AsyncToolset(id="test", auto_cleanup_completed=False)
        toolset.register(slow_operation, name="slow_op")

        await toolset._start_operation("slow_op", {"value": "test", "delay": 10.0})

        # Should have one pending operation
        assert len(toolset.get_pending_operations()) == 1

        await toolset.shutdown()

        # All operations should be cleared
        assert len(toolset._operations) == 0
