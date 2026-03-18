import enum
import json
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from livekit.agents import Agent
from livekit.agents.llm import (
    FileContent,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
    ProviderTool,
    Tool,
    ToolContext,
    ToolError,
    ToolOutput,
    Toolset,
    function_tool,
)
from livekit.agents.llm._strict import to_strict_json_schema
from livekit.agents.llm.utils import (
    build_legacy_openai_schema,
    build_strict_openai_schema,
    function_arguments_to_pydantic_model,
    make_function_call_output,
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

class TestToolOutput:
    def test_str_coercion(self):
        out = ToolOutput._coerce("hello")
        assert isinstance(out, str)
        assert isinstance(out, ToolOutput)
        assert out.text_contents == "hello"
        assert str(out) == "hello"

    def test_str_operations_backward_compat(self):
        out = ToolOutput._coerce("Error: something went wrong")
        assert out.lower().startswith("error")
        assert "something" in out
        assert out.find("went") >= 0
        assert out[0:5] == "Error"

    def test_none_coercion(self):
        out = ToolOutput._coerce(None)
        assert out.text_contents == ""
        assert str(out) == ""

    def test_dict_coercion(self):
        d = {"key": "value", "num": 1}
        out = ToolOutput._coerce(d)
        assert isinstance(out, str)
        assert out.text_contents == str(d)
        assert not out.image_contents
        assert not out.file_contents

    def test_already_tool_output_returned_as_is(self):
        original = ToolOutput._coerce("test")
        result = ToolOutput._coerce(original)
        assert result is original

    def test_image_content_coercion(self):
        img = ImageContent(image="https://example.com/img.jpg")
        out = ToolOutput._coerce(img)
        assert out.text_contents == ""
        assert str(out) == "[image]"  # no mime_type set
        assert len(out.image_contents) == 1
        assert out.image_contents[0] is img
        assert not out.file_contents

    def test_image_content_coercion_with_mime_type(self):
        img = ImageContent(image="https://example.com/img.jpg", mime_type="image/jpeg")
        out = ToolOutput._coerce(img)
        assert str(out) == "[image: image/jpeg]"

    def test_file_content_coercion(self):
        f = FileContent(name="report.pdf", data=b"%PDF", mime_type="application/pdf")
        out = ToolOutput._coerce(f)
        assert out.text_contents == ""
        assert str(out) == "[file: report.pdf]"
        assert len(out.file_contents) == 1
        assert out.file_contents[0] is f
        assert not out.image_contents

    def test_tuple_text_and_image(self):
        img = ImageContent(image="https://example.com/img.jpg")
        out = ToolOutput._coerce(("Here is the image:", img))
        assert out.text_contents == "Here is the image:"
        assert str(out) == "Here is the image: [image]"
        assert len(out.image_contents) == 1
        assert out.image_contents[0] is img

    def test_tuple_text_only(self):
        out = ToolOutput._coerce(("hello", "world"))
        assert out.text_contents == "hello\nworld"
        assert str(out) == "hello\nworld"
        assert not out.image_contents

    def test_mixed_text_image_file(self):
        img = ImageContent(image="https://example.com/chart.png")
        f = FileContent(name="data.csv", data="col1,col2\n1,2", mime_type="text/csv")
        out = ToolOutput._coerce(("Here is the chart and the raw data:", img, f))
        assert out.text_contents == "Here is the chart and the raw data:"
        assert str(out) == "Here is the chart and the raw data: [image] [file: data.csv]"
        assert len(out.image_contents) == 1
        assert out.image_contents[0] is img
        assert len(out.file_contents) == 1
        assert out.file_contents[0] is f
        assert len(out.content) == 3

    def test_list_coercion_with_image(self):
        img = ImageContent(image="https://example.com/img.jpg")
        out = ToolOutput._coerce(["result text", img])
        assert out.text_contents == "result text"
        assert len(out.image_contents) == 1

    def test_serialize_text_only_returns_str(self):
        out = ToolOutput._coerce("plain text")
        assert out._serialize() == "plain text"

    def test_serialize_with_image_returns_list(self):
        img = ImageContent(image="https://example.com/img.jpg")
        out = ToolOutput._coerce(("caption", img))
        serialized = out._serialize()
        assert isinstance(serialized, list)
        assert serialized[0] == "caption"
        assert serialized[1]["type"] == "image_content"

    def test_pydantic_coerces_plain_string(self):
        fco = FunctionCallOutput(call_id="c1", output="result", is_error=False)
        assert isinstance(fco.output, ToolOutput)
        assert fco.output.text_contents == "result"


class TestMakeFunctionCallOutput:
    def _make_call(self) -> FunctionCall:
        return FunctionCall(call_id="call_1", name="my_tool", arguments="{}")

    def test_string_output(self):
        result = make_function_call_output(
            fnc_call=self._make_call(), output="Order #123 found", exception=None
        )
        assert result.fnc_call_out is not None
        assert not result.fnc_call_out.is_error
        assert result.fnc_call_out.output.text_contents == "Order #123 found"

    def test_dict_output_becomes_repr_string(self):
        d = {"status": "ok", "count": 3}
        result = make_function_call_output(fnc_call=self._make_call(), output=d, exception=None)
        assert result.fnc_call_out is not None
        assert result.fnc_call_out.output.text_contents == str(d)

    def test_none_output(self):
        result = make_function_call_output(fnc_call=self._make_call(), output=None, exception=None)
        assert result.fnc_call_out is not None
        assert result.fnc_call_out.output.text_contents == ""

    def test_image_content_output(self):
        img = ImageContent(image="https://example.com/img.jpg")
        result = make_function_call_output(fnc_call=self._make_call(), output=img, exception=None)
        assert result.fnc_call_out is not None
        assert len(result.fnc_call_out.output.image_contents) == 1

    def test_tuple_with_image_output(self):
        img = ImageContent(image="https://example.com/img.jpg")
        result = make_function_call_output(
            fnc_call=self._make_call(), output=("Here you go:", img), exception=None
        )
        assert result.fnc_call_out is not None
        out = result.fnc_call_out.output
        assert out.text_contents == "Here you go:"
        assert len(out.image_contents) == 1

    def test_mixed_text_image_file_output(self):
        img = ImageContent(image="https://example.com/chart.png")
        f = FileContent(name="data.csv", data="col1,col2\n1,2", mime_type="text/csv")
        result = make_function_call_output(
            fnc_call=self._make_call(),
            output=("Here is the chart and the raw data:", img, f),
            exception=None,
        )
        assert result.fnc_call_out is not None
        out = result.fnc_call_out.output
        assert out.text_contents == "Here is the chart and the raw data:"
        assert len(out.image_contents) == 1
        assert len(out.file_contents) == 1

    def test_file_content_output(self):
        f = FileContent(name="data.pdf", data=b"%PDF", mime_type="application/pdf")
        result = make_function_call_output(fnc_call=self._make_call(), output=f, exception=None)
        assert result.fnc_call_out is not None
        assert len(result.fnc_call_out.output.file_contents) == 1

    def test_tool_error_exception(self):
        result = make_function_call_output(
            fnc_call=self._make_call(), output=None, exception=ToolError("not found")
        )
        assert result.fnc_call_out is not None
        assert result.fnc_call_out.is_error
        assert "not found" in result.fnc_call_out.output.text_contents

    def test_generic_exception(self):
        result = make_function_call_output(
            fnc_call=self._make_call(), output=None, exception=RuntimeError("boom")
        )
        assert result.fnc_call_out is not None
        assert result.fnc_call_out.is_error

    def test_invalid_output_returns_none_fnc_call_out(self):
        class Unsupported:
            pass

        result = make_function_call_output(
            fnc_call=self._make_call(), output=Unsupported(), exception=None
        )
        assert result.fnc_call_out is None
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
