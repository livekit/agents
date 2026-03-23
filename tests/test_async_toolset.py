import inspect

from livekit.agents.llm import RawFunctionTool, ToolContext, function_tool
from livekit.agents.llm.async_toolset import AsyncResult, AsyncRunContext, AsyncToolset
from livekit.agents.voice.events import RunContext


@function_tool
async def async_tool(ctx: AsyncRunContext, origin: str, destination: str) -> dict:
    """Book a flight.

    Args:
        origin: The origin city.
        destination: The destination city.
    """
    ctx.pending(f"Looking up flights from {origin} to {destination}...")
    return {"confirmation": "ABC123"}


@function_tool
async def regular_tool(ctx: RunContext, x: int) -> str:
    """A regular tool.

    Args:
        x: A number.
    """
    return str(x)


@function_tool
async def no_ctx_tool(x: int, y: int) -> int:
    """A tool without any context param.

    Args:
        x: First number.
        y: Second number.
    """
    return x + y


@function_tool(
    raw_schema={
        "name": "raw_async_tool",
        "description": "A raw async tool",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    }
)
async def raw_async_tool(ctx: AsyncRunContext, raw_arguments: dict[str, object]) -> str:
    """A raw async tool."""
    ctx.pending("Processing...")
    return str(raw_arguments)


class TestAsyncResult:
    def test_default_scheduling(self):
        result = AsyncResult(output={"key": "value"})
        assert result.output == {"key": "value"}
        assert result.scheduling == "when_idle"

    def test_custom_scheduling(self):
        result = AsyncResult(output="done", scheduling="interrupt")
        assert result.scheduling == "interrupt"

    def test_silent_scheduling(self):
        result = AsyncResult(output="done", scheduling="silent")
        assert result.scheduling == "silent"


class TestUpdateScheduling:
    def test_update_default_scheduling_is_when_idle(self):
        """update() should accept a scheduling parameter, defaulting to when_idle."""
        sig = inspect.signature(AsyncRunContext.update)
        param = sig.parameters["scheduling"]
        assert param.default == "when_idle"

    def test_update_accepts_scheduling_param(self):
        sig = inspect.signature(AsyncRunContext.update)
        assert "scheduling" in sig.parameters


class TestAsyncToolsetCreation:
    def test_empty_toolset(self):
        ts = AsyncToolset(id="empty")
        assert ts.id == "empty"
        assert ts.tools == []

    def test_with_tools_arg(self):
        ts = AsyncToolset(id="test", tools=[async_tool, regular_tool, no_ctx_tool])
        assert len(ts.tools) == 3

    def test_class_level_tools(self):
        class MyTools(AsyncToolset):
            @function_tool
            async def class_tool(self, ctx: AsyncRunContext, name: str) -> str:
                """Greet someone.

                Args:
                    name: The name.
                """
                ctx.pending(f"Greeting {name}...")
                return f"Hello {name}"

            @function_tool
            async def class_regular(self, ctx: RunContext, x: int) -> str:
                """Regular class tool.

                Args:
                    x: A number.
                """
                return str(x)

        ts = MyTools(id="class_tools")
        names = [t.id for t in ts.tools]
        assert "class_tool" in names
        assert "class_regular" in names

    def test_mixed_tools(self):
        ts = AsyncToolset(id="mixed", tools=[async_tool, regular_tool, no_ctx_tool])
        tool_names = [t.id for t in ts.tools]
        assert "async_tool" in tool_names
        assert "regular_tool" in tool_names
        assert "no_ctx_tool" in tool_names


class TestAsyncToolWrapping:
    def test_async_tool_is_wrapped(self):
        ts = AsyncToolset(id="test", tools=[async_tool])
        wrapped = ts.tools[0]
        # Wrapped tool should be a RawFunctionTool, not the original
        assert isinstance(wrapped, RawFunctionTool)
        assert wrapped is not async_tool

    def test_regular_tool_not_wrapped(self):
        ts = AsyncToolset(id="test", tools=[regular_tool])
        assert ts.tools[0] is regular_tool

    def test_no_ctx_tool_not_wrapped(self):
        ts = AsyncToolset(id="test", tools=[no_ctx_tool])
        assert ts.tools[0] is no_ctx_tool

    def test_wrapped_signature_has_run_context(self):
        """The wrapped tool's signature should have RunContext for context injection."""
        from typing import get_type_hints

        ts = AsyncToolset(id="test", tools=[async_tool])
        wrapped = ts.tools[0]
        hints = get_type_hints(wrapped)
        assert hints["ctx"] is RunContext

    def test_wrapped_preserves_user_params_in_schema(self):
        """User-visible params (origin, destination) should appear in the raw schema."""
        ts = AsyncToolset(id="test", tools=[async_tool])
        wrapped = ts.tools[0]
        assert isinstance(wrapped, RawFunctionTool)
        props = wrapped.info.raw_schema["parameters"]["properties"]
        assert "origin" in props
        assert "destination" in props
        assert "ctx" not in props

    def test_wrapped_preserves_tool_info(self):
        """The wrapped tool should have the same name and description."""
        ts = AsyncToolset(id="test", tools=[async_tool])
        wrapped = ts.tools[0]
        assert wrapped.info.name == "async_tool"
        assert "Book a flight." in wrapped.info.raw_schema.get("description", "")

    def test_wrapped_tool_schema_excludes_context(self):
        """The raw schema should not include the context parameter."""
        ts = AsyncToolset(id="test", tools=[async_tool])
        wrapped = ts.tools[0]
        assert isinstance(wrapped, RawFunctionTool)
        props = wrapped.info.raw_schema["parameters"]["properties"]
        assert "ctx" not in props

    def test_raw_async_tool_is_wrapped(self):
        ts = AsyncToolset(id="test", tools=[raw_async_tool])
        wrapped = ts.tools[0]
        assert isinstance(wrapped, RawFunctionTool)
        assert wrapped is not raw_async_tool

    def test_tool_context_accepts_wrapped_tools(self):
        """Wrapped tools should work with ToolContext (used by the framework)."""
        ts = AsyncToolset(id="test", tools=[async_tool, regular_tool])
        ctx = ToolContext(ts.tools)
        assert "async_tool" in ctx.function_tools
        assert "regular_tool" in ctx.function_tools

    def test_toolset_in_tool_context(self):
        """AsyncToolset as a Toolset should work with ToolContext."""
        ts = AsyncToolset(id="test", tools=[async_tool, regular_tool])
        ctx = ToolContext([ts])
        assert "async_tool" in ctx.function_tools
        assert "regular_tool" in ctx.function_tools


class TestDuplicateHandling:
    def test_default_on_duplicate_is_confirm(self):
        toolset = AsyncToolset(id="test")
        assert toolset._on_duplicate_call == "confirm"

    def test_on_duplicate_allow(self):
        toolset = AsyncToolset(id="test", on_duplicate_call="allow")
        assert toolset._on_duplicate_call == "allow"

    def test_on_duplicate_replace(self):
        toolset = AsyncToolset(id="test", on_duplicate_call="replace")
        assert toolset._on_duplicate_call == "replace"

    def test_on_duplicate_reject(self):
        toolset = AsyncToolset(id="test", on_duplicate_call="reject")
        assert toolset._on_duplicate_call == "reject"

    def test_confirm_mode_adds_confirm_param(self):
        """In confirm mode, wrapped async tools should have confirm_duplicate in schema."""
        toolset = AsyncToolset(id="test", tools=[async_tool], on_duplicate_call="confirm")
        tool = next(t for t in toolset.tools if t.id == "async_tool")
        assert isinstance(tool, RawFunctionTool)
        props = tool.info.raw_schema["parameters"]["properties"]
        assert "confirm_duplicate" in props

    def test_non_confirm_mode_no_confirm_param(self):
        """In non-confirm modes, wrapped async tools should NOT have confirm_duplicate."""
        for mode in ["allow", "replace", "reject"]:
            toolset = AsyncToolset(id="test", tools=[async_tool], on_duplicate_call=mode)
            tool = next(t for t in toolset.tools if t.id == "async_tool")
            assert isinstance(tool, RawFunctionTool)
            props = tool.info.raw_schema["parameters"]["properties"]
            assert "confirm_duplicate" not in props, (
                f"mode={mode} should not have confirm_duplicate"
            )
