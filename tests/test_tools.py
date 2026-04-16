import enum
import json
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field

from livekit.agents import Agent
from livekit.agents.llm import ProviderTool, Tool, ToolContext, Toolset, function_tool
from livekit.agents.llm._strict import to_strict_json_schema
from livekit.agents.llm.utils import (
    build_legacy_openai_schema,
    build_strict_openai_schema,
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

    def test_toolset_same_instance_dedup(self):
        # same instance appearing in multiple places is allowed (deduplication)
        toolset = MockToolset1()  # contains mock_tool_1, mock_tool_2
        ToolContext([toolset, mock_tool_1])  # same mock_tool_1 instance, no conflict
        ToolContext([mock_tool_1, toolset])
        ToolContext([MockToolset1(), MockToolset2()])  # same mock_tool_2 instance

    def test_toolset_duplicate_name_conflict(self):
        toolset = MockToolset1()  # contains mock_tool_1, mock_tool_2

        # different instances with the same name should raise
        @function_tool
        async def mock_tool_1() -> str:
            """Duplicate name, different instance"""
            return ""

        with pytest.raises(ValueError, match="duplicate function name"):
            ToolContext([toolset, mock_tool_1])

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
            "properties": {
                "raw_arguments": {
                    "additionalProperties": True,
                    "title": "Raw Arguments",
                    "type": "object",
                }
            },
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


class TestNoParametersSchema:
    """Test that functions with no parameters generate valid JSON schema."""

    def test_legacy_schema_no_parameters_has_no_required(self):
        """Legacy schema for no-param function must not include 'required'."""
        params = build_legacy_openai_schema(mock_tool_3)["function"]["parameters"]
        assert "properties" in params
        assert params["properties"] == {}
        assert "required" not in params

    def test_strict_schema_no_parameters_has_no_required(self):
        """Strict schema for no-param function must not include 'required'."""
        params = build_strict_openai_schema(mock_tool_3)["function"]["parameters"]
        assert "properties" in params
        assert params["properties"] == {}
        assert "required" not in params


class _NullableEnumModel(BaseModel):
    status: Literal["active", "inactive"] | None = Field(None)


class _NullableBoolModel(BaseModel):
    flag: bool | None = Field(None)


class _NonNullableEnumModel(BaseModel):
    status: Literal["active", "inactive"] = Field(...)


class TestStrictJsonSchema:
    def test_nullable_enum_includes_null_in_enum(self):
        schema = to_strict_json_schema(_NullableEnumModel)
        status = schema["properties"]["status"]
        assert None in status["enum"], f"enum should contain None: {status}"
        assert "null" in status["type"], f"type should contain 'null': {status}"

    def test_nullable_bool_has_null_type(self):
        schema = to_strict_json_schema(_NullableBoolModel)
        flag = schema["properties"]["flag"]
        assert "enum" not in flag, f"bool field should not have enum: {flag}"
        assert "null" in flag["type"], f"type should contain 'null': {flag}"

    def test_non_nullable_enum_excludes_null(self):
        schema = to_strict_json_schema(_NonNullableEnumModel)
        status = schema["properties"]["status"]
        assert None not in status["enum"], f"enum should not contain None: {status}"
        assert "null" not in status.get("type", []), f"type should not contain 'null': {status}"


class _CarModel(BaseModel):
    vehicle: Literal["Car"]
    brand: str
    color: str


class _BikeModel(BaseModel):
    vehicle: Literal["Bike"]
    brand: str
    color: str


class _DiscriminatedUnionModel(BaseModel):
    item: Annotated[_CarModel | _BikeModel, Field(discriminator="vehicle")]


class _NestedDiscriminatedUnionModel(BaseModel):
    items: list[Annotated[_CarModel | _BikeModel, Field(discriminator="vehicle")]]


def _has_one_of(schema: object) -> bool:
    """Recursively check if any dict in the schema tree contains 'oneOf'."""
    if isinstance(schema, dict):
        if "oneOf" in schema:
            return True
        return any(_has_one_of(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_has_one_of(v) for v in schema)
    return False


class TestDiscriminatedUnionSchema:
    """Test that discriminated unions use anyOf instead of oneOf in strict schema."""

    def test_discriminated_union_uses_anyof_not_oneof(self):
        """Pydantic emits oneOf for discriminated unions, but OpenAI strict mode
        rejects oneOf. Ensure to_strict_json_schema converts oneOf to anyOf."""
        schema = to_strict_json_schema(_DiscriminatedUnionModel)
        assert not _has_one_of(schema), (
            f"schema should not contain oneOf: {json.dumps(schema, indent=2)}"
        )
        item = schema["properties"]["item"]
        assert "anyOf" in item, f"item should have anyOf: {json.dumps(item, indent=2)}"
        assert len(item["anyOf"]) == 2, f"item should have 2 variants: {json.dumps(item, indent=2)}"

    def test_nested_discriminated_union_uses_anyof_not_oneof(self):
        """Nested discriminated unions should also convert oneOf to anyOf."""
        schema = to_strict_json_schema(_NestedDiscriminatedUnionModel)
        assert not _has_one_of(schema), (
            f"nested schema should not contain oneOf: {json.dumps(schema, indent=2)}"
        )

    def test_discriminated_union_build_strict_openai_schema(self):
        """End-to-end: build_strict_openai_schema should not produce oneOf for
        a function tool with a discriminated union parameter."""

        @function_tool
        async def lookup_vehicle(
            item: Annotated[_CarModel | _BikeModel, Field(discriminator="vehicle")],
        ) -> str:
            """Look up a vehicle."""
            return str(item)

        schema = build_strict_openai_schema(lookup_vehicle)
        schema_str = json.dumps(schema)
        assert '"oneOf"' not in schema_str, (
            f"strict openai schema should not contain oneOf: {json.dumps(schema, indent=2)}"
        )


class _OpenEnumModel(BaseModel):
    """Simulates a codegen'd "open enum" pattern (e.g. Fern Python SDK).
    Union[Literal["a", "b"], Any] produces an anyOf with a bare {} entry."""

    preference: Literal["a", "b"] | Any | None = None


class _NestedOpenEnumModel(BaseModel):
    items: list[_OpenEnumModel]


def _has_empty_schema(schema: object) -> bool:
    """Recursively check if any dict in the schema tree is an empty {}."""
    if isinstance(schema, dict):
        if schema == {}:
            return True
        return any(_has_empty_schema(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_has_empty_schema(v) for v in schema)
    return False


class TestEmptySchemaStripping:
    """Test that empty {} entries are stripped from anyOf/oneOf."""

    def test_open_enum_strips_empty_anyof(self):
        schema = to_strict_json_schema(_OpenEnumModel)
        assert not _has_empty_schema(schema), (
            f"schema should not contain empty {{}}: {json.dumps(schema, indent=2)}"
        )

    def test_nested_open_enum_strips_empty_anyof(self):
        schema = to_strict_json_schema(_NestedOpenEnumModel)
        assert not _has_empty_schema(schema), (
            f"nested schema should not contain empty {{}}: {json.dumps(schema, indent=2)}"
        )

    def test_single_variant_after_strip_is_unwrapped(self):
        """When stripping {} leaves a single variant, anyOf should be unwrapped."""
        schema = to_strict_json_schema(_OpenEnumModel)
        pref = schema["properties"]["preference"]
        # After stripping {}, the union should be simplified (no single-element anyOf)
        any_of = pref.get("anyOf")
        if any_of is not None:
            assert len(any_of) != 1, (
                f"single-element anyOf should be unwrapped: {json.dumps(pref, indent=2)}"
            )


class TestExecuteFunctionCallValidationErrors:
    """Test that argument validation errors are surfaced to the LLM."""

    @pytest.mark.asyncio
    async def test_missing_required_arg_surfaces_error(self):
        """When the LLM omits a required argument, the error details should be
        returned as a ToolError (is_error=True) instead of 'An internal error occurred'."""
        from livekit.agents.llm.llm import FunctionToolCall
        from livekit.agents.llm.utils import execute_function_call

        tool_ctx = ToolContext([mock_tool_1])
        tool_call = FunctionToolCall(
            name="mock_tool_1",
            arguments="{}",  # missing required 'arg1'
            call_id="test-call-1",
        )

        result = await execute_function_call(tool_call, tool_ctx)

        assert result.fnc_call_out is not None
        assert result.fnc_call_out.is_error is True
        # The error message should contain details about what went wrong,
        # NOT the generic "An internal error occurred"
        assert "An internal error occurred" not in result.fnc_call_out.output
        assert "arg1" in result.fnc_call_out.output

    @pytest.mark.asyncio
    async def test_wrong_type_arg_surfaces_error(self):
        """When the LLM provides an argument with the wrong type, the validation
        error details should be surfaced."""
        from livekit.agents.llm.llm import FunctionToolCall
        from livekit.agents.llm.utils import execute_function_call

        tool_ctx = ToolContext([mock_tool_2])
        tool_call = FunctionToolCall(
            name="mock_tool_2",
            arguments='{"arg1": 12345}',  # should be a MockOption string, not int
            call_id="test-call-2",
        )

        result = await execute_function_call(tool_call, tool_ctx)

        assert result.fnc_call_out is not None
        assert result.fnc_call_out.is_error is True
        assert "An internal error occurred" not in result.fnc_call_out.output

    @pytest.mark.asyncio
    async def test_valid_args_still_work(self):
        """Verify that valid arguments still execute successfully."""
        from livekit.agents.llm.llm import FunctionToolCall
        from livekit.agents.llm.utils import execute_function_call

        tool_ctx = ToolContext([mock_tool_1])
        tool_call = FunctionToolCall(
            name="mock_tool_1",
            arguments='{"arg1": "hello"}',
            call_id="test-call-3",
        )

        result = await execute_function_call(tool_call, tool_ctx)

        assert result.fnc_call_out is not None
        assert result.fnc_call_out.is_error is False
        assert "arg1" in result.fnc_call_out.output

    @pytest.mark.asyncio
    async def test_invalid_json_surfaces_error(self):
        """When the LLM provides invalid JSON, the error should be surfaced."""
        from livekit.agents.llm.llm import FunctionToolCall
        from livekit.agents.llm.utils import execute_function_call

        tool_ctx = ToolContext([mock_tool_1])
        tool_call = FunctionToolCall(
            name="mock_tool_1",
            arguments="{not valid json}",
            call_id="test-call-4",
        )

        result = await execute_function_call(tool_call, tool_ctx)

        assert result.fnc_call_out is not None
        assert result.fnc_call_out.is_error is True
        # Should contain error details, not generic message
        assert "An internal error occurred" not in result.fnc_call_out.output


class TestAsyncToolsetDedup:
    """Test that multiple AsyncToolsets can coexist without duplicate tool name conflicts."""

    def test_two_async_toolsets_no_conflict(self):
        """Two AsyncToolsets share the same get_running_tasks/cancel_task singleton tools."""
        from livekit.agents.llm.async_toolset import AsyncToolset

        ts1 = AsyncToolset(id="booking", tools=[mock_tool_1])
        ts2 = AsyncToolset(id="search", tools=[mock_tool_2])

        # should not raise — the management tools are the same module-level instances
        ctx = ToolContext([ts1, ts2])

        # only one copy of each management tool in the flattened list
        names = [t.id for t in ctx.flatten() if hasattr(t, "id")]
        assert names.count("get_running_tasks") == 1
        assert names.count("cancel_task") == 1

    def test_async_toolset_same_id_no_conflict(self):
        """Two AsyncToolsets with the same id should not conflict."""
        from livekit.agents.llm.async_toolset import AsyncToolset

        ts1 = AsyncToolset(id="same_id", tools=[mock_tool_1])
        ts2 = AsyncToolset(id="same_id", tools=[mock_tool_2])

        # should not raise
        ctx = ToolContext([ts1, ts2])

        names = [t.id for t in ctx.flatten() if hasattr(t, "id")]
        assert names.count("get_running_tasks") == 1
        assert names.count("cancel_task") == 1
