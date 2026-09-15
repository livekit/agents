import pytest

from livekit.agents.beta.toolsets.tool_search import (
    BM25SearchStrategy,
    KeywordSearchStrategy,
    SearchItem,
    SearchStrategy,
    ToolSearchToolset,
    _get_tool_params,
)
from livekit.agents.llm import Tool, ToolContext, Toolset, function_tool


@function_tool
async def weather_tool(city: str) -> str:
    """Get weather for a city"""
    return f"sunny in {city}"


@function_tool
async def forecast_tool(city: str) -> str:
    """Get weather forecast for a city"""
    return f"rain in {city}"


@function_tool
async def stock_tool(symbol: str) -> str:
    """Get stock price for a symbol"""
    return f"{symbol}: $100"


@function_tool
async def search_papers(query: str) -> str:
    """Search academic papers on arxiv"""
    return f"papers about {query}"


@function_tool
async def calculator(expression: str) -> str:
    """Calculate a math expression"""
    return "42"


class WeatherToolset(Toolset):
    def __init__(self):
        super().__init__(id="weather")

    @property
    def tools(self) -> list[Tool | Toolset]:
        return [weather_tool, forecast_tool]


class FinanceToolset(Toolset):
    def __init__(self):
        super().__init__(id="finance")

    @property
    def tools(self) -> list[Tool | Toolset]:
        return [stock_tool]


class AcademicToolset(Toolset):
    def __init__(self):
        super().__init__(id="academic")

    @property
    def tools(self) -> list[Tool | Toolset]:
        return [search_papers]


def _make_items() -> list[SearchItem]:
    return [
        SearchItem(
            name="get_weather",
            description="Get current weather for a city",
            parameters={"city": "City name"},
            source=weather_tool,
        ),
        SearchItem(
            name="get_forecast",
            description="Get weather forecast for upcoming days",
            parameters={"city": "City name", "days": "Number of days"},
            source=forecast_tool,
        ),
        SearchItem(
            name="get_stock_price",
            description="Get stock price for a symbol",
            parameters={"symbol": "Stock ticker symbol"},
            source=stock_tool,
        ),
        SearchItem(
            name="search_papers",
            description="Search academic papers on arxiv",
            parameters={"query": "Search query"},
            source=search_papers,
        ),
        SearchItem(
            name="calculator",
            description="Calculate a math expression",
            parameters={"expression": "Math expression"},
            source=calculator,
        ),
    ]


STRATEGIES = [KeywordSearchStrategy(), BM25SearchStrategy()]


class TestSearchStrategies:
    """Tests that apply to all SearchStrategy implementations."""

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_match_by_name(self, strategy: SearchStrategy):
        items = _make_items()
        strategy.build_index(items)
        results = strategy.search("weather", items, 5)
        assert len(results) >= 1
        assert results[0].name in ("get_weather", "get_forecast")

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_match_by_description(self, strategy: SearchStrategy):
        items = _make_items()
        strategy.build_index(items)
        results = strategy.search("academic", items, 5)
        assert len(results) >= 1
        assert results[0].name == "search_papers"

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_match_by_parameter(self, strategy: SearchStrategy):
        items = _make_items()
        strategy.build_index(items)
        results = strategy.search("ticker symbol", items, 5)
        assert len(results) >= 1
        assert results[0].name == "get_stock_price"

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_max_results_limit(self, strategy: SearchStrategy):
        items = _make_items()
        strategy.build_index(items)
        results = strategy.search("get", items, 2)
        assert len(results) == 2

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_no_matches(self, strategy: SearchStrategy):
        items = _make_items()
        strategy.build_index(items)
        results = strategy.search("xyznonexistent", items, 5)
        assert len(results) == 0

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_empty_query(self, strategy: SearchStrategy):
        items = _make_items()
        strategy.build_index(items)
        results = strategy.search("", items, 5)
        assert len(results) == 0

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_name_weighted_higher(self, strategy: SearchStrategy):
        """Tool name matches should rank higher than description-only matches."""
        items = _make_items()
        strategy.build_index(items)
        results = strategy.search("calculator", items, 5)
        assert len(results) >= 1
        assert results[0].name == "calculator"

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_index_data_set_by_build_index(self, strategy: SearchStrategy):
        items = _make_items()
        assert items[0].index_data is None
        strategy.build_index(items)
        assert items[0].index_data is not None

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_index_data_preserved_through_search(self, strategy: SearchStrategy):
        items = _make_items()
        strategy.build_index(items)
        results = strategy.search("weather", items, 5)
        assert results[0].index_data is not None


class TestBM25Specific:
    """Tests specific to BM25 behavior."""

    def test_specific_term_ranks_higher_than_common(self):
        """Rare terms (high IDF) should rank higher than common ones."""
        strategy = BM25SearchStrategy()
        items = _make_items()
        strategy.build_index(items)

        results = strategy.search("academic papers arxiv", items, 5)
        assert len(results) >= 1
        assert results[0].name == "search_papers"

    def test_cleanup_clears_state(self):
        strategy = BM25SearchStrategy()
        items = _make_items()
        strategy.build_index(items)
        assert len(strategy._idf) > 0

        strategy.cleanup()
        assert len(strategy._idf) == 0


class TestToolSearchToolset:
    def test_initial_tools_only_search(self):
        ts = ToolSearchToolset(
            id="search",
            tools=[WeatherToolset(), FinanceToolset()],
        )
        assert len(ts.tools) == 1
        assert ts.tools[0].id == "tool_search"

    async def test_setup_indexes_tools(self):
        ts = ToolSearchToolset(
            id="search",
            tools=[WeatherToolset(), FinanceToolset()],
        )
        await ts.setup()
        assert len(ts.tools) == 1  # still only tool_search
        # Each function tool is its own SearchItem: weather_tool, forecast_tool, stock_tool
        assert len(ts._search_items) == 3

    async def test_search_loads_matching_toolset(self):
        ts = ToolSearchToolset(
            id="search",
            tools=[WeatherToolset(), FinanceToolset(), AcademicToolset()],
        )
        await ts.setup()

        await ts._handle_search({"query": "weather"})

        # Verify via ToolContext that the loaded tools are accessible
        ctx = ToolContext([ts])
        assert "tool_search" in ctx.function_tools
        assert "weather_tool" in ctx.function_tools
        assert "forecast_tool" in ctx.function_tools
        assert "stock_tool" not in ctx.function_tools

    async def test_toolset_is_atomic_unit(self):
        """If a toolset matches, ALL its tools are loaded (not just the matching one)."""
        ts = ToolSearchToolset(id="search", tools=[WeatherToolset(), FinanceToolset()])
        await ts.setup()

        # "forecast" isn't in the toolset id, but forecast_tool is in WeatherToolset
        await ts._handle_search({"query": "weather"})
        ctx = ToolContext([ts])
        assert "weather_tool" in ctx.function_tools
        assert "forecast_tool" in ctx.function_tools  # loaded atomically
        assert "stock_tool" not in ctx.function_tools

    async def test_toolset_loads_once_even_if_multiple_tools_match(self):
        """If multiple tools from the same Toolset match, the Toolset is loaded only once."""
        ts = ToolSearchToolset(
            id="search",
            tools=[WeatherToolset(), FinanceToolset()],
        )
        await ts.setup()

        # Both weather_tool and forecast_tool match "weather", but WeatherToolset loads once
        await ts._handle_search({"query": "weather"})
        assert len(ts._loaded_tools) == 1  # only one unique source (WeatherToolset)

    async def test_standalone_tools(self):
        ts = ToolSearchToolset(
            id="search",
            tools=[calculator, WeatherToolset()],
        )
        await ts.setup()

        # calculator + weather_tool + forecast_tool = 3 SearchItems
        assert len(ts._search_items) == 3

        await ts._handle_search({"query": "math calculate"})
        ctx = ToolContext([ts])
        assert "calculator" in ctx.function_tools
        assert "weather_tool" not in ctx.function_tools

    async def test_nested_toolsets(self):
        """A toolset containing another toolset should be handled correctly."""

        class OuterToolset(Toolset):
            def __init__(self):
                super().__init__(id="outer")

            @property
            def tools(self) -> list[Tool | Toolset]:
                return [WeatherToolset(), calculator]

        ts = ToolSearchToolset(id="search", tools=[OuterToolset(), FinanceToolset()])
        await ts.setup()

        await ts._handle_search({"query": "weather forecast calculator"})
        ctx = ToolContext([ts])
        # OuterToolset loaded atomically — includes its nested WeatherToolset + calculator
        assert "weather_tool" in ctx.function_tools
        assert "forecast_tool" in ctx.function_tools
        assert "calculator" in ctx.function_tools

    async def test_duplicate_search_is_idempotent(self):
        ts = ToolSearchToolset(
            id="search",
            tools=[WeatherToolset(), FinanceToolset()],
        )
        await ts.setup()

        await ts._handle_search({"query": "weather"})
        count = len(ts.tools)
        await ts._handle_search({"query": "weather"})
        assert len(ts.tools) == count

    async def test_custom_search_strategy(self):
        """Users can provide a custom SearchStrategy (e.g. embedding-based)."""

        class MockStrategy:
            def __init__(self):
                self.indexed = False

            def build_index(self, items: list[SearchItem]) -> None:
                self.indexed = True
                for item in items:
                    item.index_data = f"embedding_for_{item.name}"

            def search(
                self, query: str, items: list[SearchItem], max_results: int
            ) -> list[SearchItem]:
                # Always return items that have "weather" in their data
                return [i for i in items if i.index_data and "weather" in str(i.index_data)][
                    :max_results
                ]

        strategy = MockStrategy()
        ts = ToolSearchToolset(
            id="search",
            tools=[WeatherToolset(), FinanceToolset()],
            search_strategy=strategy,
        )
        await ts.setup()
        assert strategy.indexed

        await ts._handle_search({"query": "anything"})
        ctx = ToolContext([ts])
        assert "weather_tool" in ctx.function_tools
        assert "stock_tool" not in ctx.function_tools

    async def test_async_search_strategy(self):
        """SearchStrategy methods can be async."""

        class AsyncStrategy:
            def __init__(self):
                self.indexed = False

            async def build_index(self, items: list[SearchItem]) -> None:
                self.indexed = True

            async def search(
                self, query: str, items: list[SearchItem], max_results: int
            ) -> list[SearchItem]:
                return [i for i in items if query.lower() in i.name.lower()][:max_results]

        strategy = AsyncStrategy()
        ts = ToolSearchToolset(
            id="search",
            tools=[WeatherToolset(), FinanceToolset()],
            search_strategy=strategy,
        )
        await ts.setup()
        assert strategy.indexed

        await ts._handle_search({"query": "weather"})
        ctx = ToolContext([ts])
        assert "weather_tool" in ctx.function_tools


class TestToolSearchWithToolContext:
    async def test_tool_context_sees_search_tool(self):
        ts = ToolSearchToolset(
            id="search",
            tools=[WeatherToolset(), FinanceToolset()],
        )
        await ts.setup()

        ctx = ToolContext([ts])
        assert "tool_search" in ctx.function_tools
        assert len(ctx.function_tools) == 1

    async def test_tool_context_updates_after_search(self):
        ts = ToolSearchToolset(
            id="search",
            tools=[WeatherToolset(), FinanceToolset()],
        )
        await ts.setup()

        ctx = ToolContext([ts])
        assert len(ctx.function_tools) == 1

        await ts._handle_search({"query": "weather"})

        # Re-building ToolContext picks up new tools (this is what generation.py does)
        ctx2 = ToolContext([ts])
        assert "tool_search" in ctx2.function_tools
        assert "weather_tool" in ctx2.function_tools
        assert "forecast_tool" in ctx2.function_tools

    async def test_mixed_with_regular_tools(self):
        ts = ToolSearchToolset(
            id="search",
            tools=[WeatherToolset(), FinanceToolset()],
        )
        await ts.setup()

        ctx = ToolContext([calculator, ts])
        assert "calculator" in ctx.function_tools
        assert "tool_search" in ctx.function_tools
        assert len(ctx.function_tools) == 2


class TestGetToolParams:
    def test_function_tool_params(self):
        """FunctionTool params extracted from signature + docstring."""
        params = _get_tool_params(weather_tool)
        assert "city" in params

    def test_function_tool_with_descriptions(self):
        """FunctionTool param descriptions come from docstring."""

        @function_tool
        async def detailed_tool(name: str, age: int) -> str:
            """A tool with documented params.

            Args:
                name: The user's full name
                age: The user's age in years
            """
            return f"{name} is {age}"

        params = _get_tool_params(detailed_tool)
        assert params == {"name": "The user's full name", "age": "The user's age in years"}

    def test_function_tool_no_params(self):
        """FunctionTool with no parameters returns empty dict."""

        @function_tool
        async def no_params() -> str:
            """A tool with no params."""
            return "done"

        params = _get_tool_params(no_params)
        assert params == {}

    def test_raw_function_tool_params(self):
        """RawFunctionTool params extracted from raw_schema properties."""

        @function_tool(
            raw_schema={
                "name": "raw_with_params",
                "description": "A raw tool with params",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or region name",
                        },
                        "unit": {
                            "type": "string",
                            "description": "Temperature unit (celsius or fahrenheit)",
                        },
                    },
                },
            }
        )
        async def raw_with_params(raw_arguments: dict[str, object]) -> str:
            return str(raw_arguments)

        params = _get_tool_params(raw_with_params)
        assert params == {
            "location": "City or region name",
            "unit": "Temperature unit (celsius or fahrenheit)",
        }

    def test_raw_function_tool_no_descriptions(self):
        """RawFunctionTool properties without description field get empty string."""

        @function_tool(
            raw_schema={
                "name": "raw_no_desc",
                "description": "A raw tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                },
            }
        )
        async def raw_no_desc(raw_arguments: dict[str, object]) -> str:
            return str(raw_arguments)

        params = _get_tool_params(raw_no_desc)
        assert params == {"query": "", "limit": ""}

    def test_raw_function_tool_empty_properties(self):
        """RawFunctionTool with no properties returns empty dict."""

        @function_tool(
            raw_schema={
                "name": "raw_empty",
                "description": "A raw tool with no params",
                "parameters": {"type": "object", "properties": {}},
            }
        )
        async def raw_empty(raw_arguments: dict[str, object]) -> str:
            return "done"

        params = _get_tool_params(raw_empty)
        assert params == {}

    def test_raw_function_tool_no_properties_key(self):
        """RawFunctionTool with no 'properties' key in parameters returns empty dict."""

        @function_tool(
            raw_schema={
                "name": "raw_no_props",
                "description": "A raw tool",
                "parameters": {"type": "object"},
            }
        )
        async def raw_no_props(raw_arguments: dict[str, object]) -> str:
            return "done"

        params = _get_tool_params(raw_no_props)
        assert params == {}
