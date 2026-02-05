import enum

import pytest

from livekit.agents import Agent
from livekit.agents.llm import ProviderTool, Tool, ToolContext, Toolset, function_tool
from livekit.agents.llm.utils import (
    function_arguments_to_pydantic_model,
    prepare_function_arguments,
)


class MockOption(str, enum.Enum):
    A = "a"
    B = "b"
    C = "c"


@function_tool
async def mock_tool_1(arg1: str, opt_arg2: str | None = None) -> dict[str, str | None]:
    """Test tool 1
    Args:
        arg1: The first argument
        opt_arg2: The optional second argument
    """
    return {"arg1": arg1, "opt_arg2": opt_arg2}


@function_tool
async def mock_tool_2(arg1: MockOption) -> str:
    """Test tool 2"""
    return arg1.value


@function_tool
async def mock_tool_3() -> str:
    """Test tool 3"""
    return "tool3"


@function_tool(
    raw_schema={
        "name": "raw_tool_1",
        "description": "A raw tool",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    }
)
async def raw_tool_1() -> str:
    """Raw test tool 1"""
    return "raw1"


@function_tool(
    raw_schema={
        "name": "raw_tool_2",
        "description": "Another raw tool",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    }
)
async def raw_tool_2() -> str:
    """Raw test tool 2"""
    return "raw2"


class DummyAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a dummy agent.")

    @function_tool
    async def mock_tool_in_agent(
        self, arg1: str, opt_arg2: str | None = None
    ) -> dict[str, str | None]:
        """Mock tool in agent

        Args:
            arg1: The first argument
            opt_arg2: The optional second argument
        """
        return {"arg1": arg1, "opt_arg2": opt_arg2}

    @function_tool(
        raw_schema={
            "name": "raw_tool_in_agent",
            "description": "A raw tool in agent",
            "parameters": {
                "type": "object",
                "properties": {"arg1": {"type": "string", "description": "The first argument"}},
            },
        }
    )
    async def raw_tool_in_agent(self, raw_arguments: dict[str, object]) -> str:
        """Raw tool in agent"""
        assert "arg1" in raw_arguments
        return raw_arguments["arg1"]


class DummyProviderTool(ProviderTool):
    def __init__(self, id: str):
        super().__init__(id=id)


class MockToolset1(Toolset):
    def __init__(self):
        super().__init__(id="mock_toolset_1")

    @property
    def tools(self) -> list[Tool]:
        return [mock_tool_1, mock_tool_2]


class MockToolset2(Toolset):
    def __init__(self):
        super().__init__(id="mock_toolset_2")

    @property
    def tools(self) -> list[Tool]:
        return [mock_tool_2, DummyProviderTool("provider1")]


class TestToolContext:
    def test_equals_empty_contexts(self):
        ctx1 = ToolContext.empty()
        ctx2 = ToolContext.empty()
        assert ctx1 == ctx2

    def test_equals_same_tools(self):
        ctx1 = ToolContext([mock_tool_1, mock_tool_2])
        ctx2 = ToolContext([mock_tool_1, mock_tool_2])
        assert ctx1 == ctx2

    def test_equals_same_tools_different_order(self):
        ctx1 = ToolContext([mock_tool_1, mock_tool_2])
        ctx2 = ToolContext([mock_tool_2, mock_tool_1])
        assert ctx1 == ctx2

    def test_not_equals_different_tools(self):
        ctx1 = ToolContext([mock_tool_1, mock_tool_2])
        ctx2 = ToolContext([mock_tool_1, mock_tool_3])
        assert ctx1 != ctx2

    def test_not_equals_different_provider_tools(self):
        provider1 = DummyProviderTool("provider1")
        provider2 = DummyProviderTool("provider2")
        provider3 = DummyProviderTool("provider3")

        ctx1 = ToolContext([mock_tool_1, provider1, provider2])
        ctx2 = ToolContext([mock_tool_1, provider1, provider3])
        assert ctx1 != ctx2

    def test_equals_different_provider_tools_order(self):
        provider1 = DummyProviderTool("provider1")
        provider2 = DummyProviderTool("provider2")

        ctx1 = ToolContext([mock_tool_1, provider1, provider2])
        ctx2 = ToolContext([mock_tool_1, provider2, provider1])
        assert ctx1 == ctx2

    def test_not_equals_missing_provider_tools(self):
        provider1 = DummyProviderTool("provider1")

        ctx1 = ToolContext([mock_tool_1, provider1])
        ctx2 = ToolContext([mock_tool_1])
        assert ctx1 != ctx2

    def test_copy_equals_original(self):
        ctx1 = ToolContext([mock_tool_1, mock_tool_2])
        ctx2 = ctx1.copy()
        assert ctx1 == ctx2

    def test_update_tools_changes_equality(self):
        ctx1 = ToolContext([mock_tool_1])
        ctx2 = ToolContext([mock_tool_1])
        assert ctx1 == ctx2

        ctx1.update_tools([mock_tool_1, mock_tool_2])
        assert ctx1 != ctx2

    def test_not_equals_same_name_different_implementation(self):
        @function_tool
        async def duplicate_name() -> str:
            """First implementation"""
            return "first"

        @function_tool
        async def duplicate_name() -> str:  # noqa: F811
            """Second implementation"""
            return "second"

        ctx1 = ToolContext([mock_tool_1])
        ctx1._fnc_tools_map["duplicate_name"] = duplicate_name

        @function_tool
        async def duplicate_name() -> str:  # noqa: F811
            """Third implementation"""
            return "third"

        ctx2 = ToolContext([mock_tool_1])
        ctx2._fnc_tools_map["duplicate_name"] = duplicate_name

        assert ctx1 != ctx2

    def test_all_tools_returns_combined_list(self):
        provider1 = DummyProviderTool("provider1")
        provider2 = DummyProviderTool("provider2")

        ctx = ToolContext([mock_tool_1, mock_tool_2, provider1, provider2])
        all_tools = ctx.flatten()

        assert len(all_tools) == 4
        assert mock_tool_1 in all_tools
        assert mock_tool_2 in all_tools
        assert provider1 in all_tools
        assert provider2 in all_tools

    def test_toolset_with_regular_tools(self):
        toolset = MockToolset1()  # contains mock_tool_1, mock_tool_2
        ctx = ToolContext([toolset, mock_tool_3])

        all_tools = ctx.flatten()
        assert mock_tool_1 in all_tools
        assert mock_tool_2 in all_tools
        assert mock_tool_3 in all_tools
        assert len(ctx.function_tools) == 3
        assert toolset in ctx.toolsets

        # copy should preserve toolsets
        ctx_copy = ctx.copy()
        assert ctx == ctx_copy
        assert toolset in ctx_copy.toolsets

    def test_toolset_duplicate_name_conflict(self):
        toolset = MockToolset1()  # contains mock_tool_1, mock_tool_2

        # conflict: toolset before tool, tool before toolset, multiple toolsets
        with pytest.raises(ValueError, match="duplicate function name"):
            ToolContext([toolset, mock_tool_1])

        with pytest.raises(ValueError, match="duplicate function name"):
            ToolContext([mock_tool_1, toolset])

        with pytest.raises(ValueError, match="duplicate function name"):
            ToolContext([MockToolset1(), MockToolset2()])  # both have mock_tool_2

    def test_toolset_equality(self):
        toolset = MockToolset1()

        # same toolset instance -> equal
        ctx1 = ToolContext([toolset, mock_tool_3])
        ctx2 = ToolContext([mock_tool_3, toolset])  # different order
        assert ctx1 == ctx2

        # different toolset instances -> not equal
        ctx3 = ToolContext([MockToolset1()])
        ctx4 = ToolContext([MockToolset1()])
        assert ctx3 != ctx4

        # toolset vs expanded tools -> not equal
        ctx5 = ToolContext([toolset])
        ctx6 = ToolContext([mock_tool_1, mock_tool_2])
        assert ctx5 != ctx6


class TestToolExecution:
    def test_function_arguments_to_pydantic_model(self):
        schema1 = function_arguments_to_pydantic_model(mock_tool_1)
        assert schema1.model_json_schema() == {
            "properties": {
                "arg1": {"description": "The first argument", "title": "Arg1", "type": "string"},
                "opt_arg2": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "The optional second argument",
                    "title": "Opt Arg2",
                },
            },
            "required": ["arg1"],
            "title": "MockTool1Args",
            "type": "object",
        }

        schema2 = function_arguments_to_pydantic_model(mock_tool_2)
        assert schema2.model_json_schema() == {
            "$defs": {
                "MockOption": {"enum": ["a", "b", "c"], "title": "MockOption", "type": "string"}
            },
            "properties": {"arg1": {"$ref": "#/$defs/MockOption"}},
            "required": ["arg1"],
            "title": "MockTool2Args",
            "type": "object",
        }

        agent = DummyAgent()
        schema3 = function_arguments_to_pydantic_model(agent.mock_tool_in_agent)
        assert schema3.model_json_schema() == {
            "properties": {
                "arg1": {"description": "The first argument", "title": "Arg1", "type": "string"},
                "opt_arg2": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "The optional second argument",
                    "title": "Opt Arg2",
                },
            },
            "required": ["arg1"],
            "title": "MockToolInAgentArgs",
            "type": "object",
        }

        schema4 = function_arguments_to_pydantic_model(agent.raw_tool_in_agent)
        assert schema4.model_json_schema() == {
            "properties": {"raw_arguments": {"title": "Raw Arguments", "type": "object"}},
            "required": ["raw_arguments"],
            "title": "RawToolInAgentArgs",
            "type": "object",
        }

    async def test_tool_execution(self):
        args, kwargs = prepare_function_arguments(
            fnc=mock_tool_1, json_arguments='{"arg1": "test", "opt_arg2": "test2"}'
        )
        assert args == ("test", "test2")
        assert kwargs == {}
        output = await mock_tool_1(*args, **kwargs)
        assert output == {"arg1": "test", "opt_arg2": "test2"}

        args, kwargs = prepare_function_arguments(fnc=mock_tool_2, json_arguments='{"arg1": "a"}')
        assert args == ("a",)
        assert kwargs == {}
        output = await mock_tool_2(*args, **kwargs)
        assert output == "a"

        agent = DummyAgent()
        args, kwargs = prepare_function_arguments(
            fnc=agent.mock_tool_in_agent, json_arguments='{"arg1": "test", "opt_arg2": "test2"}'
        )
        assert args == ("test", "test2")
        assert kwargs == {}
        output = await agent.mock_tool_in_agent(*args, **kwargs)
        assert output == {"arg1": "test", "opt_arg2": "test2"}

    async def test_raw_function_tool_execution(self):
        agent = DummyAgent()
        args, kwargs = prepare_function_arguments(
            fnc=agent.raw_tool_in_agent, json_arguments='{"arg1": "test"}'
        )
        assert args == ({"arg1": "test"},)
        assert kwargs == {}
        output = await agent.raw_tool_in_agent(*args, **kwargs)
        assert output == "test"

    async def test_tool_execution_with_default_value(self):
        args, kwargs = prepare_function_arguments(
            fnc=mock_tool_1, json_arguments='{"arg1": "test"}'
        )
        assert args == ("test", None)
        assert kwargs == {}
        output = await mock_tool_1(*args, **kwargs)
        assert output == {"arg1": "test", "opt_arg2": None}

        agent = DummyAgent()
        args, kwargs = prepare_function_arguments(
            fnc=agent.mock_tool_in_agent, json_arguments='{"arg1": "test"}'
        )
        assert args == ("test", None)
        output = await agent.mock_tool_in_agent(*args, **kwargs)
        assert output == {"arg1": "test", "opt_arg2": None}

    def test_unexpected_arguments(self):
        with pytest.raises(ValueError, match="validation error"):
            prepare_function_arguments(fnc=mock_tool_1, json_arguments='{"opt_arg2": "test2"}')

        with pytest.raises(ValueError, match="Received None for required parameter"):
            prepare_function_arguments(fnc=mock_tool_2, json_arguments='{"arg1": null}')

        with pytest.raises(ValueError, match="validation error"):
            prepare_function_arguments(fnc=mock_tool_2, json_arguments='{"arg1": "d"}')

        agent = DummyAgent()
        with pytest.raises(ValueError, match="validation error"):
            prepare_function_arguments(
                fnc=agent.mock_tool_in_agent, json_arguments='{"opt_arg2": "test2"}'
            )


def assert_hashable(x: object) -> None:
    hash(x)


class TestProviderToolHashability:
    def test_openai_tools_are_hashable(self):
        from livekit.plugins.openai.tools import CodeInterpreter, FileSearch, WebSearch

        assert_hashable(WebSearch())
        assert_hashable(
            WebSearch(
                search_context_size="high",
                user_location={"type": "approximate", "city": "San Francisco", "country": "US"},
            )
        )
        assert_hashable(FileSearch(vector_store_ids=["vector_store_1"]))
        assert_hashable(
            FileSearch(
                vector_store_ids=["vector_store_1", "vector_store_2"],
                max_num_results=10,
                ranking_options={"ranker": "auto", "score_threshold": 0.5},
            )
        )
        assert_hashable(CodeInterpreter())
        assert_hashable(CodeInterpreter(container="container_id"))
        assert_hashable(CodeInterpreter(container={"type": "auto"}))

    def test_google_tools_are_hashable(self):
        from livekit.plugins.google.tools import (
            FileSearch,
            GoogleMaps,
            GoogleSearch,
            ToolCodeExecution,
            URLContext,
        )

        assert_hashable(GoogleSearch())
        assert_hashable(
            GoogleSearch(
                exclude_domains=["example.com", "test.com"],
            )
        )
        assert_hashable(GoogleMaps())
        assert_hashable(GoogleMaps(enable_widget=True))
        assert_hashable(URLContext())
        assert_hashable(FileSearch(file_search_store_names=["file_store_1"]))
        assert_hashable(
            FileSearch(
                file_search_store_names=["file_store_1", "file_store_2"],
                top_k=5,
                metadata_filter="category = 'docs'",
            )
        )
        assert_hashable(ToolCodeExecution())

    def test_xai_tools_are_hashable(self):
        from livekit.plugins.xai.tools import FileSearch, WebSearch, XSearch

        assert_hashable(WebSearch())
        assert_hashable(XSearch())
        assert_hashable(XSearch(allowed_x_handles=["@user1", "@user2"]))
        assert_hashable(FileSearch(vector_store_ids=["vector_store_1"]))
        assert_hashable(
            FileSearch(
                vector_store_ids=["vector_store_1", "vector_store_2"],
                max_num_results=10,
            )
        )
