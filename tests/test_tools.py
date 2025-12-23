from livekit.agents.llm import ProviderTool, ToolContext, function_tool


@function_tool
async def test_tool_1() -> str:
    """Test tool 1"""
    return "tool1"


@function_tool
async def test_tool_2() -> str:
    """Test tool 2"""
    return "tool2"


@function_tool
async def test_tool_3() -> str:
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


class DummyProviderTool(ProviderTool):
    def __init__(self, name: str):
        self.name = name


class TestToolContext:
    def test_equals_empty_contexts(self):
        ctx1 = ToolContext.empty()
        ctx2 = ToolContext.empty()
        assert ctx1 == ctx2

    def test_equals_same_tools(self):
        ctx1 = ToolContext([test_tool_1, test_tool_2])
        ctx2 = ToolContext([test_tool_1, test_tool_2])
        assert ctx1 == ctx2

    def test_equals_same_tools_different_order(self):
        ctx1 = ToolContext([test_tool_1, test_tool_2])
        ctx2 = ToolContext([test_tool_2, test_tool_1])
        assert ctx1 == ctx2

    def test_not_equals_different_tools(self):
        ctx1 = ToolContext([test_tool_1, test_tool_2])
        ctx2 = ToolContext([test_tool_1, test_tool_3])
        assert ctx1 != ctx2

    def test_not_equals_different_provider_tools(self):
        provider1 = DummyProviderTool("provider1")
        provider2 = DummyProviderTool("provider2")
        provider3 = DummyProviderTool("provider3")

        ctx1 = ToolContext([test_tool_1, provider1, provider2])
        ctx2 = ToolContext([test_tool_1, provider1, provider3])
        assert ctx1 != ctx2

    def test_equals_different_provider_tools_order(self):
        provider1 = DummyProviderTool("provider1")
        provider2 = DummyProviderTool("provider2")

        ctx1 = ToolContext([test_tool_1, provider1, provider2])
        ctx2 = ToolContext([test_tool_1, provider2, provider1])
        assert ctx1 == ctx2

    def test_not_equals_missing_provider_tools(self):
        provider1 = DummyProviderTool("provider1")

        ctx1 = ToolContext([test_tool_1, provider1])
        ctx2 = ToolContext([test_tool_1])
        assert ctx1 != ctx2

    def test_copy_equals_original(self):
        ctx1 = ToolContext([test_tool_1, test_tool_2])
        ctx2 = ctx1.copy()
        assert ctx1 == ctx2

    def test_update_tools_changes_equality(self):
        ctx1 = ToolContext([test_tool_1])
        ctx2 = ToolContext([test_tool_1])
        assert ctx1 == ctx2

        ctx1.update_tools([test_tool_1, test_tool_2])
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

        ctx1 = ToolContext([test_tool_1])
        ctx1._tools_map["duplicate_name"] = duplicate_name

        @function_tool
        async def duplicate_name() -> str:  # noqa: F811
            """Third implementation"""
            return "third"

        ctx2 = ToolContext([test_tool_1])
        ctx2._tools_map["duplicate_name"] = duplicate_name

        assert ctx1 != ctx2

    def test_all_tools_returns_combined_list(self):
        provider1 = DummyProviderTool("provider1")
        provider2 = DummyProviderTool("provider2")

        ctx = ToolContext([test_tool_1, test_tool_2, provider1, provider2])
        all_tools = ctx.all_tools

        assert len(all_tools) == 4
        assert test_tool_1 in all_tools
        assert test_tool_2 in all_tools
        assert provider1 in all_tools
        assert provider2 in all_tools
