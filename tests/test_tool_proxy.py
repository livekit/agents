import json
from unittest.mock import MagicMock

import pytest

from livekit.agents.beta.toolsets.tool_proxy import ToolProxyToolset
from livekit.agents.llm import ToolContext, ToolError, Toolset, function_tool


@function_tool
async def weather_tool(city: str) -> str:
    """Get weather for a city.

    Args:
        city: City name or region
    """
    return f"sunny in {city}"


@function_tool
async def forecast_tool(city: str, days: int) -> str:
    """Get weather forecast.

    Args:
        city: City name or region
        days: Number of days to forecast
    """
    return f"{days}-day forecast for {city}: mostly sunny"


@function_tool
async def stock_tool(symbol: str) -> str:
    """Get stock price for a symbol.

    Args:
        symbol: Stock ticker symbol
    """
    return f"{symbol}: $100"


@function_tool(
    raw_schema={
        "name": "raw_search",
        "description": "Search with raw schema",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
        },
    }
)
async def raw_search(raw_arguments: dict[str, object]) -> str:
    return f"results for {raw_arguments.get('query')}"


class WeatherToolset(Toolset):
    def __init__(self):
        super().__init__(id="weather", tools=[weather_tool, forecast_tool])


class FinanceToolset(Toolset):
    def __init__(self):
        super().__init__(id="finance", tools=[stock_tool])


def _mock_ctx() -> MagicMock:
    return MagicMock()


class TestToolProxyToolset:
    def test_tools_always_two(self):
        """ToolProxyToolset always exposes exactly search_tools and call_tool."""
        ts = ToolProxyToolset(
            id="proxy",
            tools=[WeatherToolset(), FinanceToolset()],
        )
        assert len(ts.tools) == 2
        tool_names = [t.id for t in ts.tools]
        assert "tool_search" in tool_names
        assert "call_tool" in tool_names

    async def test_tools_constant_after_search(self):
        """Tool list stays constant even after search — no dynamic loading."""
        ts = ToolProxyToolset(
            id="proxy",
            tools=[WeatherToolset(), FinanceToolset()],
        )
        await ts.setup()

        assert len(ts.tools) == 2
        await ts._handle_search({"query": "weather"})
        assert len(ts.tools) == 2  # still just search + call

    async def test_search_returns_full_schemas(self):
        """search_tools returns tool schemas with full JSON schema for parameters."""
        ts = ToolProxyToolset(
            id="proxy",
            tools=[WeatherToolset(), FinanceToolset()],
        )
        await ts.setup()

        result = await ts._handle_search({"query": "weather"})
        parsed = [json.loads(schema) for schema in result.split("\n")]

        assert isinstance(parsed, list)
        assert len(parsed) >= 1

        names = [t["name"] for t in parsed]
        assert "weather_tool" in names

        # Check schema has full JSON schema with types
        weather = next(t for t in parsed if t["name"] == "weather_tool")
        assert "Get weather for a city" in weather["description"]
        assert "parameters" in weather
        params = weather["parameters"]
        assert "properties" in params
        assert "city" in params["properties"]
        assert params["properties"]["city"]["type"] == "string"

    async def test_search_returns_raw_tool_schema(self):
        """search_tools returns full raw_schema parameters for RawFunctionTool."""
        ts = ToolProxyToolset(
            id="proxy",
            tools=[raw_search],
        )
        await ts.setup()

        result = await ts._handle_search({"query": "search"})
        parsed = [json.loads(schema) for schema in result.split("\n")]

        assert len(parsed) >= 1
        tool = parsed[0]
        assert tool["name"] == "raw_search"
        params = tool["parameters"]
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert params["properties"]["query"]["type"] == "string"
        assert params["properties"]["limit"]["type"] == "integer"

    async def test_search_no_results_raises_tool_error(self):
        ts = ToolProxyToolset(
            id="proxy",
            tools=[WeatherToolset()],
        )
        await ts.setup()

        with pytest.raises(ToolError, match="No tools found"):
            await ts._handle_search({"query": "xyznonexistent"})

    async def test_search_empty_query_raises_tool_error(self):
        ts = ToolProxyToolset(
            id="proxy",
            tools=[WeatherToolset()],
        )
        await ts.setup()

        with pytest.raises(ToolError, match="cannot be empty"):
            await ts._handle_search({"query": ""})

    async def test_call_tool_function_tool(self):
        """call_tool can execute a FunctionTool by name."""
        ts = ToolProxyToolset(
            id="proxy",
            tools=[WeatherToolset(), FinanceToolset()],
        )
        await ts.setup()

        result = await ts._handle_call(
            _mock_ctx(), {"name": "weather_tool", "parameters": {"city": "London"}}
        )
        assert "sunny in London" in str(result)

    async def test_call_tool_with_multiple_args(self):
        ts = ToolProxyToolset(
            id="proxy",
            tools=[WeatherToolset()],
        )
        await ts.setup()

        result = await ts._handle_call(
            _mock_ctx(), {"name": "forecast_tool", "parameters": {"city": "Tokyo", "days": 3}}
        )
        assert "3-day forecast for Tokyo" in str(result)

    async def test_call_tool_raw_function_tool(self):
        """call_tool can execute a RawFunctionTool by name."""
        ts = ToolProxyToolset(
            id="proxy",
            tools=[raw_search],
        )
        await ts.setup()

        result = await ts._handle_call(
            _mock_ctx(), {"name": "raw_search", "parameters": {"query": "test", "limit": 5}}
        )
        assert "results for test" in str(result)

    async def test_call_unknown_tool_raises_tool_error(self):
        ts = ToolProxyToolset(
            id="proxy",
            tools=[WeatherToolset()],
        )
        await ts.setup()

        with pytest.raises(ToolError, match="unknown tool"):
            await ts._handle_call(_mock_ctx(), {"name": "nonexistent_tool", "parameters": {}})

    async def test_call_missing_required_arg_raises_tool_error(self, capsys):
        """Missing required argument produces a detailed ToolError for the LLM."""
        ts = ToolProxyToolset(
            id="proxy",
            tools=[WeatherToolset()],
        )
        await ts.setup()

        with pytest.raises(ToolError, match="invalid parameters") as exc_info:
            await ts._handle_call(_mock_ctx(), {"name": "weather_tool", "parameters": {}})

        error_msg = exc_info.value.message
        print(f"Missing arg error: {error_msg}")

        # Error message should contain the missing field name and indicate it's required
        error_data = json.loads(error_msg.split(":", 1)[1].strip())
        field_names = [err["loc"][0] for err in error_data]
        assert "city" in field_names
        assert any(err["type"] == "missing" for err in error_data)

    async def test_call_wrong_type_arg_raises_tool_error(self, capsys):
        """Wrong argument type produces a detailed ToolError for the LLM."""
        ts = ToolProxyToolset(
            id="proxy",
            tools=[WeatherToolset()],
        )
        await ts.setup()

        with pytest.raises(ToolError, match="invalid parameters") as exc_info:
            await ts._handle_call(
                _mock_ctx(),
                {"name": "forecast_tool", "parameters": {"city": "Tokyo", "days": "not_a_number"}},
            )

        error_msg = exc_info.value.message
        print(f"Wrong type error: {error_msg}")

        # Error message should contain the field name and indicate a type parsing issue
        error_data = json.loads(error_msg.split(":", 1)[1].strip())
        field_names = [err["loc"][0] for err in error_data]
        assert "days" in field_names
        assert any("int" in err["type"] for err in error_data)

    async def test_call_tool_propagates_tool_error(self):
        """ToolError raised inside a tool is re-raised as-is."""

        @function_tool
        async def tool_with_tool_error(x: str) -> str:
            """A tool that raises ToolError.

            Args:
                x: Some input
            """
            raise ToolError("custom error message")

        ts = ToolProxyToolset(
            id="proxy",
            tools=[tool_with_tool_error],
        )
        await ts.setup()

        with pytest.raises(ToolError, match="custom error message"):
            await ts._handle_call(
                _mock_ctx(), {"name": "tool_with_tool_error", "parameters": {"x": "test"}}
            )

    async def test_tool_context_sees_only_proxy_tools(self):
        """ToolContext built from ToolProxyToolset only sees search + call."""
        ts = ToolProxyToolset(
            id="proxy",
            tools=[WeatherToolset(), FinanceToolset()],
        )
        await ts.setup()

        ctx = ToolContext([ts])
        assert "tool_search" in ctx.function_tools
        assert "call_tool" in ctx.function_tools
        assert len(ctx.function_tools) == 2
        assert "weather_tool" not in ctx.function_tools

    async def test_nested_toolsets(self):
        """Nested toolsets are accessible via call_tool."""

        class OuterToolset(Toolset):
            def __init__(self):
                super().__init__(id="outer", tools=[WeatherToolset(), stock_tool])

        ts = ToolProxyToolset(id="proxy", tools=[OuterToolset()])
        await ts.setup()

        # Search should find tools from nested toolsets
        result = await ts._handle_search({"query": "weather stock"})
        parsed = [json.loads(schema) for schema in result.split("\n")]
        names = [t["name"] for t in parsed]
        assert "weather_tool" in names or "stock_tool" in names

        # call_tool should work for nested tools
        result = await ts._handle_call(
            _mock_ctx(), {"name": "weather_tool", "parameters": {"city": "Paris"}}
        )
        assert "sunny in Paris" in str(result)
