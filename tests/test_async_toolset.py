import asyncio
from unittest.mock import MagicMock

import pytest

from livekit.agents.beta.tools import AsyncContext, AsyncToolset, OperationStatus
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
    def test_tools_use_original_name(self):
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
        ts = AsyncToolset(id="no_poll")

        @ts.function_tool
        async def my_tool(x: int) -> int:
            return x

        tool_names = [t.id for t in ts.tools]
        assert "check_operation_status" not in tool_names
        assert "get_operation_result" not in tool_names

    def test_async_context_excluded_from_schema(self):
        """AsyncContext param should not appear in the LLM-facing tool schema."""
        ts = AsyncToolset(id="schema")

        @ts.function_tool
        async def my_tool(ctx: AsyncContext, x: int, y: str) -> str:
            return f"{x}{y}"

        tool = ts.tools[0]
        assert isinstance(tool, FunctionTool)
        # The schema should only contain x and y, not ctx
        from livekit.agents.llm.utils import function_arguments_to_pydantic_model

        model = function_arguments_to_pydantic_model(tool)
        schema = model.model_json_schema()
        assert "x" in schema["properties"]
        assert "y" in schema["properties"]
        # ctx (AsyncContext) should be excluded
        assert "ctx" not in schema.get("properties", {})


def _make_mock_ctx():
    """Create a mock RunContext with a mock session."""
    session = MagicMock()
    session.chat_ctx = MagicMock()
    session.chat_ctx.copy.return_value = MagicMock()
    session.generate_reply = MagicMock()

    ctx = MagicMock()
    ctx.session = session
    return ctx


class TestAsyncContext:
    async def test_pending_sets_message(self):
        session = MagicMock()
        op = MagicMock()
        op.name = "test"

        async_ctx = AsyncContext(session=session, operation=op)
        async_ctx.pending("Looking up flights...")

        assert async_ctx._pending_message == "Looking up flights..."
        assert async_ctx._pending_event.is_set()

    async def test_update_calls_generate_reply(self):
        session = MagicMock()
        session.generate_reply = MagicMock()
        op = MagicMock()
        op.name = "book_flight"

        async_ctx = AsyncContext(session=session, operation=op)
        async_ctx.update("Found 3 flights")

        session.generate_reply.assert_called_once()
        call_kwargs = session.generate_reply.call_args
        assert "Found 3 flights" in call_kwargs.kwargs["instructions"]

    async def test_session_and_userdata_accessible(self):
        session = MagicMock()
        session.userdata = {"user_id": "123"}
        op = MagicMock()

        async_ctx = AsyncContext(session=session, operation=op)
        assert async_ctx.session is session
        assert async_ctx.userdata == {"user_id": "123"}


class TestAsyncToolsetDispatch:
    async def test_dispatch_uses_pending_message(self):
        ts = AsyncToolset(id="dispatch")

        @ts.function_tool
        async def slow_task(ctx: AsyncContext, value: str) -> str:
            ctx.pending(f"Processing {value}...")
            await asyncio.sleep(10)
            return value

        run_ctx = _make_mock_ctx()
        result = await ts._dispatch("slow_task", run_ctx, {"value": "test"})

        assert result == "Processing test..."
        assert len(ts._operations) == 1

        await ts.shutdown()

    async def test_dispatch_default_when_no_pending(self):
        ts = AsyncToolset(id="default_msg")

        @ts.function_tool
        async def my_task(x: int) -> int:
            await asyncio.sleep(10)
            return x

        run_ctx = _make_mock_ctx()
        result = await ts._dispatch("my_task", run_ctx, {"x": 1})

        assert "my_task" in result
        assert "background" in result

        await ts.shutdown()

    async def test_dispatch_unknown_function(self):
        ts = AsyncToolset(id="unknown")
        run_ctx = _make_mock_ctx()
        result = await ts._dispatch("nonexistent", run_ctx, {})
        assert "Error" in result

    async def test_operation_completes_and_pushes_result(self):
        ts = AsyncToolset(id="push")

        @ts.function_tool
        async def fast_task(value: str) -> dict:
            return {"result": value}

        run_ctx = _make_mock_ctx()
        await ts._dispatch("fast_task", run_ctx, {"value": "hello"})

        await asyncio.sleep(0.05)

        op = list(ts._operations.values())[0]
        assert op.status == OperationStatus.COMPLETED
        assert op.result == {"result": "hello"}

        run_ctx.session.generate_reply.assert_called_once()

    async def test_operation_with_ctx_completes_and_pushes(self):
        ts = AsyncToolset(id="push_ctx")

        @ts.function_tool
        async def task_with_ctx(ctx: AsyncContext, value: str) -> dict:
            ctx.pending(f"Working on {value}")
            return {"result": value}

        run_ctx = _make_mock_ctx()
        result = await ts._dispatch("task_with_ctx", run_ctx, {"value": "hello"})

        assert result == "Working on hello"

        await asyncio.sleep(0.05)

        op = list(ts._operations.values())[0]
        assert op.status == OperationStatus.COMPLETED
        # generate_reply called for the final push
        run_ctx.session.generate_reply.assert_called()

    async def test_operation_with_progress_updates(self):
        ts = AsyncToolset(id="progress")

        @ts.function_tool
        async def multi_step(ctx: AsyncContext, value: str) -> str:
            ctx.pending("Starting...")
            await asyncio.sleep(0.01)
            ctx.update("Halfway there...")
            await asyncio.sleep(0.01)
            return f"done: {value}"

        run_ctx = _make_mock_ctx()
        result = await ts._dispatch("multi_step", run_ctx, {"value": "test"})

        assert result == "Starting..."

        await asyncio.sleep(0.1)

        op = list(ts._operations.values())[0]
        assert op.status == OperationStatus.COMPLETED

        # Should have called generate_reply multiple times:
        # once for update(), once for final push
        assert run_ctx.session.generate_reply.call_count >= 2

    async def test_operation_failure_pushes_error(self):
        ts = AsyncToolset(id="fail_push")

        @ts.function_tool
        async def failing_task(msg: str) -> str:
            raise ValueError(msg)

        run_ctx = _make_mock_ctx()
        await ts._dispatch("failing_task", run_ctx, {"msg": "boom"})

        await asyncio.sleep(0.05)

        op = list(ts._operations.values())[0]
        assert op.status == OperationStatus.FAILED
        assert "boom" in op.error

        run_ctx.session.generate_reply.assert_called_once()

    async def test_cancelled_operation_does_not_push(self):
        ts = AsyncToolset(id="cancel_no_push")

        @ts.function_tool
        async def slow_task(value: str) -> str:
            await asyncio.sleep(10)
            return value

        run_ctx = _make_mock_ctx()
        await ts._dispatch("slow_task", run_ctx, {"value": "test"})

        await ts.shutdown()

        run_ctx.session.generate_reply.assert_not_called()


class TestAsyncToolsetOperationTracking:
    async def test_get_operation(self):
        ts = AsyncToolset(id="track_get")

        @ts.function_tool
        async def my_task(x: int) -> int:
            return x

        run_ctx = _make_mock_ctx()
        await ts._dispatch("my_task", run_ctx, {"x": 42})
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

        run_ctx = _make_mock_ctx()
        await ts._dispatch("slow_task", run_ctx, {"value": "test"})

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

        run_ctx = _make_mock_ctx()
        await ts._dispatch("fast_task", run_ctx, {"x": 1})
        await asyncio.sleep(0.05)

        completed = ts.get_completed_operations()
        assert len(completed) == 1
        assert completed[0].status == OperationStatus.COMPLETED


class TestAsyncToolsetSharing:
    async def test_shared_across_contexts(self):
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
        ts = AsyncToolset(id="shared_ops")

        @ts.function_tool
        async def task(value: str) -> str:
            return value

        run_ctx = _make_mock_ctx()
        await ts._dispatch("task", run_ctx, {"value": "shared"})
        await asyncio.sleep(0.05)

        assert len(ts.get_completed_operations()) == 1


class TestAsyncToolsetShutdown:
    async def test_shutdown_cancels_pending(self):
        ts = AsyncToolset(id="shutdown")

        @ts.function_tool
        async def slow_task(value: str) -> str:
            await asyncio.sleep(10)
            return value

        run_ctx = _make_mock_ctx()
        await ts._dispatch("slow_task", run_ctx, {"value": "test"})
        assert len(ts.get_pending_operations()) == 1

        await ts.shutdown()
        assert len(ts._operations) == 0

    async def test_shutdown_is_idempotent(self):
        ts = AsyncToolset(id="shutdown_idem")
        await ts.shutdown()
        await ts.shutdown()
