import enum
import json
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError

from livekit.agents import Agent
from livekit.agents.llm import (
    ProviderTool,
    Tool,
    ToolContext,
    ToolError,
    ToolFlag,
    Toolset,
    function_tool,
)
from livekit.agents.llm._strict import to_strict_json_schema
from livekit.agents.llm.utils import (
    build_legacy_openai_schema,
    build_strict_openai_schema,
    function_arguments_to_pydantic_model,
    prepare_function_arguments,
)

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


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

    def test_openai_responses_raw_schema_does_not_mutate_tool_schema(self):
        ctx = ToolContext([raw_tool_1])
        raw_schema = raw_tool_1.info.raw_schema.copy()

        responses_tools = ctx.parse_function_tools("openai.responses")

        assert responses_tools[0]["type"] == "function"
        assert raw_tool_1.info.raw_schema == raw_schema
        assert "type" not in raw_tool_1.info.raw_schema

        chat_tools = ctx.parse_function_tools("openai")
        assert chat_tools[0] == {
            "type": "function",
            "function": raw_tool_1.info.raw_schema,
        }
        assert "type" not in chat_tools[0]["function"]

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

    def test_exclude_hides_toolset_member_but_keeps_toolset(self):
        toolset = MockToolset1()  # contains mock_tool_1, mock_tool_2
        ctx = ToolContext([toolset, mock_tool_3])

        ctx._exclude([mock_tool_1])

        # the excluded member is no longer callable / no longer visible to the LLM
        assert "mock_tool_1" not in ctx.function_tools
        assert mock_tool_1 not in ctx.flatten()
        # its sibling and the top-level tool remain callable
        assert "mock_tool_2" in ctx.function_tools
        assert "mock_tool_3" in ctx.function_tools
        # the toolset itself stays so executor routing and lifecycle are unaffected
        assert toolset in ctx.toolsets

    def test_exclude_empty_is_noop(self):
        ctx = ToolContext([mock_tool_1, mock_tool_2])
        ctx._exclude([])
        assert set(ctx.function_tools) == {"mock_tool_1", "mock_tool_2"}


def test_end_call_tool_ignore_on_enter_flag():
    from livekit.agents.beta import EndCallTool

    (tool,) = EndCallTool().tools  # defaults to ignore_on_enter=False
    assert not (tool.info.flags & ToolFlag.IGNORE_ON_ENTER)

    (tool,) = EndCallTool(ignore_on_enter=True).tools
    assert tool.info.flags & ToolFlag.IGNORE_ON_ENTER


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

    def test_field_constraints_preserved(self):
        # Field(...) constraints (ge/le/pattern/...) live in FieldInfo.metadata,
        # not its attributes; they must survive the signature -> pydantic model
        # conversion so the model both advertises and enforces them.
        @function_tool
        async def book(count: Annotated[int, Field(ge=1, le=10, description="how many")]) -> str:
            """Book a thing."""
            return "ok"

        model = function_arguments_to_pydantic_model(book)
        prop = model.model_json_schema()["properties"]["count"]
        assert prop["minimum"] == 1
        assert prop["maximum"] == 10
        assert prop["description"] == "how many"

        model(count=5)  # within bounds
        for bad in (0, 11):
            with pytest.raises(ValidationError):
                model(count=bad)

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

    async def test_null_uses_default_for_non_optional_param(self):
        @function_tool
        async def tool(arg1: str, count: int = 5) -> int:
            """Tool with a defaulted non-optional parameter"""
            return count

        args, kwargs = prepare_function_arguments(
            fnc=tool, json_arguments='{"arg1": "test", "count": null}'
        )
        assert args == ("test", 5)
        assert kwargs == {}

    async def test_null_uses_default_in_nested_model_arg(self):
        class Preferences(BaseModel):
            color: str = "red"
            note: str | None = None

        @function_tool
        async def set_prefs(prefs: Preferences) -> str:
            """Tool with a nested model argument"""
            return prefs.color

        args, kwargs = prepare_function_arguments(
            fnc=set_prefs, json_arguments='{"prefs": {"color": null, "note": null}}'
        )
        assert args == (Preferences(color="red", note=None),)
        assert kwargs == {}

    def test_unexpected_arguments(self):
        with pytest.raises(ToolError, match="validation error"):
            prepare_function_arguments(fnc=mock_tool_1, json_arguments='{"opt_arg2": "test2"}')

        # a null for a required parameter with no default stays null and is
        # rejected by pydantic validation
        with pytest.raises(ToolError, match="validation error"):
            prepare_function_arguments(fnc=mock_tool_2, json_arguments='{"arg1": null}')

        with pytest.raises(ToolError, match="validation error"):
            prepare_function_arguments(fnc=mock_tool_2, json_arguments='{"arg1": "d"}')

        agent = DummyAgent()
        with pytest.raises(ToolError, match="validation error"):
            prepare_function_arguments(
                fnc=agent.mock_tool_in_agent, json_arguments='{"opt_arg2": "test2"}'
            )

    def test_repairs_malformed_json_arguments(self):
        """LLMs occasionally emit syntactically invalid JSON for tool calls.
        We should recover via json_repair instead of giving up immediately."""
        # Missing closing brace.
        args, kwargs = prepare_function_arguments(
            fnc=mock_tool_1, json_arguments='{"arg1": "hi", "opt_arg2": "yo"'
        )
        assert args == ("hi", "yo")

        # Unquoted-style payload (json_repair handles loose forms).
        args, kwargs = prepare_function_arguments(fnc=mock_tool_1, json_arguments="{arg1: 'hi'}")
        assert args == ("hi", None)

    def test_unrepairable_json_arguments_raise(self):
        """If json_repair can't recover anything meaningful, the error should
        be surfaced as a ToolError so the LLM can self-correct on the next turn."""
        with pytest.raises(ToolError, match="could not parse"):
            prepare_function_arguments(fnc=mock_tool_1, json_arguments="not json at all")

    def test_repairs_gemma_template_token_leak(self):
        """Gemma chat-template tokens (<|...|>) sometimes leak into function-call
        arguments, producing unparseable JSON. The repair pass should both fix
        the JSON shape AND strip the leaked tokens so the original payload
        (an order id, here) survives the round-trip cleanly."""

        @function_tool
        async def remove_order_item(order_id: list[str]) -> str:
            return ",".join(order_id)

        # Real failing payload captured from a Gemma 4 deployment. The model
        # tried to emit `{"order_id": ["O_WAAB70"]}` but `<|"|"` template
        # tokens leaked around the value, breaking the JSON (the string is
        # never properly closed before `]}`).
        leaked = '{"order_id": ["<|\\"|\\"O_WAAB70<|\\"|\\"]}'

        args, kwargs = prepare_function_arguments(fnc=remove_order_item, json_arguments=leaked)
        assert args == (["O_WAAB70"],), f"expected order_id=['O_WAAB70'], got {args}"

    def test_strip_template_tokens_leaves_clean_strings_alone(self):
        """Token stripping should only kick in on the repair path; well-formed
        arguments that happen to contain `<|...|>` substrings should pass through
        unmodified (we only run the strip pass when json_repair was triggered)."""
        args, _ = prepare_function_arguments(fnc=mock_tool_1, json_arguments='{"arg1": "<|safe|>"}')
        assert args == ("<|safe|>", None)

    def test_parse_function_arguments_strict(self):
        """parse_function_arguments should accept valid JSON unchanged."""
        from livekit.agents.llm.utils import parse_function_arguments

        assert parse_function_arguments('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}
        # Empty/None payloads normalize to {}.
        assert parse_function_arguments("{}") == {}
        assert parse_function_arguments("null") == {}

    def test_parse_function_arguments_non_dict_raises(self):
        """parse_function_arguments must reject non-dict shapes (the caller
        always expects a kwarg-style mapping)."""
        from livekit.agents.llm.utils import parse_function_arguments

        with pytest.raises(ValueError, match="expected dict"):
            parse_function_arguments("[1, 2, 3]")
        # Bare JSON strings hit the Nova-Sonic double-encoded unwrap path,
        # which raises a more specific error.
        with pytest.raises(ValueError, match="non-JSON string"):
            parse_function_arguments('"just a string"')

    def test_prepare_function_arguments_accepts_pre_parsed_dict(self):
        """Callers that already parsed the JSON (e.g. _execute_tools_task after
        canonicalizing fnc_call.arguments) can pass the dict directly. Must
        behave the same as passing the equivalent JSON string."""
        from_string = prepare_function_arguments(
            fnc=mock_tool_1, json_arguments='{"arg1": "hi", "opt_arg2": "yo"}'
        )
        from_dict = prepare_function_arguments(
            fnc=mock_tool_1, json_arguments={"arg1": "hi", "opt_arg2": "yo"}
        )
        assert from_string == from_dict == (("hi", "yo"), {})

    @pytest.mark.asyncio
    async def test_execute_function_call_preserves_valid_argument_structure(self):
        """Canonicalization may normalize whitespace, but for already-valid JSON
        the *structure* (keys, values) must round-trip identically. The repair
        path must not silently alter legitimate arguments."""
        from livekit.agents.llm.llm import FunctionToolCall
        from livekit.agents.llm.utils import execute_function_call

        tool_ctx = ToolContext([mock_tool_1])
        original_args = '{"arg1": "hello", "opt_arg2": "world"}'
        tool_call = FunctionToolCall(
            name="mock_tool_1", arguments=original_args, call_id="call-valid-1"
        )

        result = await execute_function_call(tool_call, tool_ctx)
        assert result.fnc_call_out is not None
        assert result.fnc_call_out.is_error is False

        # Whatever ended up in fnc_call.arguments must decode to the same dict.
        assert json.loads(result.fnc_call.arguments) == {"arg1": "hello", "opt_arg2": "world"}

    @pytest.mark.asyncio
    async def test_execute_function_call_canonicalizes_repaired_arguments(self):
        """After a successful repair, the FunctionCall's `arguments` should be
        rewritten to canonical JSON so the next LLM turn doesn't see the
        broken string in conversation history (which would fail re-serialization
        and cause 5xx from providers like Vertex)."""
        from livekit.agents.llm.llm import FunctionToolCall
        from livekit.agents.llm.utils import execute_function_call

        @function_tool
        async def remove_order_item(order_id: list[str]) -> str:
            return ",".join(order_id)

        tool_ctx = ToolContext([remove_order_item])
        raw_broken = '{"order_id": ["<|\\"|\\"O_WAAB70<|\\"|\\"]}'
        tool_call = FunctionToolCall(
            name="remove_order_item",
            arguments=raw_broken,
            call_id="call-canon-1",
        )

        result = await execute_function_call(tool_call, tool_ctx)
        assert result.fnc_call_out is not None
        assert result.fnc_call_out.is_error is False
        assert result.fnc_call_out.output == "O_WAAB70"

        # The arguments stored on the FunctionCall should now be valid JSON
        # that re-serializes cleanly to the repaired payload — so subsequent
        # LLM turns don't choke on broken syntax in the history.
        assert result.fnc_call.arguments != raw_broken
        parsed = json.loads(result.fnc_call.arguments)
        assert parsed == {"order_id": ["O_WAAB70"]}

    @pytest.mark.asyncio
    async def test_execute_function_call_canonicalizes_when_validation_fails(self):
        """When JSON parses (possibly via repair) but pydantic validation then
        fails, fnc_call.arguments must STILL be canonicalized — otherwise the
        broken raw payload propagates into chat history and the next LLM turn
        gets a 5xx from providers that re-serialize."""
        from livekit.agents.llm.llm import FunctionToolCall
        from livekit.agents.llm.utils import execute_function_call

        @function_tool
        async def take_int(arg1: int) -> str:
            return str(arg1)

        tool_ctx = ToolContext([take_int])
        # malformed JSON that json_repair can fix, but the value is the wrong
        # type for pydantic validation (string where int is required)
        raw_broken = '{arg1: "not_an_int"}'
        tool_call = FunctionToolCall(
            name="take_int", arguments=raw_broken, call_id="call-validate-fail"
        )

        result = await execute_function_call(tool_call, tool_ctx)

        # Validation failed
        assert result.fnc_call_out is not None
        assert result.fnc_call_out.is_error is True

        # But fnc_call.arguments was canonicalized despite the validation error
        assert result.fnc_call.arguments != raw_broken
        parsed = json.loads(result.fnc_call.arguments)
        assert parsed == {"arg1": "not_an_int"}


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
    def test_same_id_no_conflict(self):
        """Two AsyncToolsets with the same id and different tools should not raise."""
        from livekit.agents.llm.async_toolset import AsyncToolset

        ts1 = AsyncToolset(id="same_id", tools=[mock_tool_1])
        ts2 = AsyncToolset(id="same_id", tools=[mock_tool_2])

        ctx = ToolContext([ts1, ts2])

        names = [t.id for t in ctx.flatten() if hasattr(t, "id")]
        assert "mock_tool_1" in names
        assert "mock_tool_2" in names


class TestConfirmDuplicateSchema:
    """Schema injection for @function_tool(on_duplicate='confirm')."""

    @staticmethod
    @function_tool(on_duplicate="confirm")
    async def _confirm_tool(origin: str, destination: str) -> dict[str, str]:
        """Book a flight.

        Args:
            origin: where to fly from
            destination: where to fly to
        """
        return {"origin": origin, "destination": destination}

    @staticmethod
    @function_tool(
        raw_schema={
            "name": "raw_confirm_tool",
            "description": "raw tool",
            "parameters": {
                "type": "object",
                "properties": {"gate_id": {"type": "string"}},
                "required": ["gate_id"],
            },
        },
        on_duplicate="confirm",
    )
    async def _raw_confirm_tool(raw_arguments: dict[str, Any]) -> dict[str, Any]:
        return raw_arguments

    def test_legacy_schema_has_confirm_param(self):
        params = build_legacy_openai_schema(self._confirm_tool)["function"]["parameters"]
        assert "lk_agents_confirm_duplicate" in params["properties"]
        # legacy: not in required, has a default
        assert "lk_agents_confirm_duplicate" not in params.get("required", [])
        assert params["properties"]["lk_agents_confirm_duplicate"]["default"] is False

    def test_strict_schema_has_confirm_param(self):
        params = build_strict_openai_schema(self._confirm_tool)["function"]["parameters"]
        assert "lk_agents_confirm_duplicate" in params["properties"]
        # strict: in required, nullable type for optionality
        assert "lk_agents_confirm_duplicate" in params["required"]
        prop_type = params["properties"]["lk_agents_confirm_duplicate"]["type"]
        assert "null" in prop_type and "boolean" in prop_type

    def test_raw_schema_has_confirm_param(self):
        params = self._raw_confirm_tool.info.raw_schema["parameters"]
        assert "lk_agents_confirm_duplicate" in params["properties"]
        assert "lk_agents_confirm_duplicate" in params["required"]
        prop_type = params["properties"]["lk_agents_confirm_duplicate"]["type"]
        assert "null" in prop_type and "boolean" in prop_type

    def test_plain_tool_has_no_confirm_param(self):
        params = build_strict_openai_schema(mock_tool_1)["function"]["parameters"]
        assert "lk_agents_confirm_duplicate" not in params.get("properties", {})

    @pytest.mark.asyncio
    async def test_direct_call_with_typed_args(self):
        # Wrapper preserves direct invocation with the original signature.
        result = await self._confirm_tool(origin="NYC", destination="Tokyo")
        assert result == {"origin": "NYC", "destination": "Tokyo"}

    @pytest.mark.asyncio
    async def test_wrapper_drops_confirm_kwarg(self):
        # The wrapper pops lk_agents_confirm_duplicate before calling the user fn.
        result = await self._confirm_tool(
            origin="NYC", destination="Tokyo", lk_agents_confirm_duplicate=True
        )
        assert result == {"origin": "NYC", "destination": "Tokyo"}


class TestAsyncToolOptions:
    """``AsyncToolOptions`` resolution, override layering, and routing."""

    def test_resolve_defaults_filled(self):
        from livekit.agents.voice.tool_executor import (
            _ASYNC_TOOL_OPTIONS_DEFAULTS,
            _resolve_async_tool_options,
        )

        # None → all defaults
        resolved = _resolve_async_tool_options(None)
        assert resolved == _ASYNC_TOOL_OPTIONS_DEFAULTS

    def test_resolve_partial_fills_missing_with_defaults(self):
        from livekit.agents.voice.tool_executor import (
            _ASYNC_TOOL_OPTIONS_DEFAULTS,
            _resolve_async_tool_options,
        )

        resolved = _resolve_async_tool_options({"update_template": "custom-update"})
        assert resolved["update_template"] == "custom-update"
        # other keys retain the module default
        assert (
            resolved["duplicate_reject_template"]
            == _ASYNC_TOOL_OPTIONS_DEFAULTS["duplicate_reject_template"]
        )
        assert (
            resolved["reply_at_tail_template"]
            == _ASYNC_TOOL_OPTIONS_DEFAULTS["reply_at_tail_template"]
        )

    def test_executor_uses_resolved_options(self):
        from livekit.agents.voice.tool_executor import _ToolExecutor

        executor = _ToolExecutor(async_tool_options={"duplicate_reject_template": "rejected!"})
        assert executor._tool_options["duplicate_reject_template"] == "rejected!"
        # unspecified key falls back to default, NOT to anything else
        assert "{function_name}" in executor._tool_options["update_template"]

    @staticmethod
    def _mock_scope(session_update: str = "session", agent_update: str | None = None):
        # minimal stand-ins for what AsyncToolset._attach_activity reads from
        from livekit.agents.types import NOT_GIVEN
        from livekit.agents.voice.tool_executor import _resolve_async_tool_options

        class _Session:
            _async_tool_options = _resolve_async_tool_options({"update_template": session_update})

        class _Agent:
            _async_tool_options = (
                {"update_template": agent_update} if agent_update is not None else NOT_GIVEN
            )

        class _Activity:
            _agent = _Agent()

        return _Session(), _Activity()

    def test_toolset_own_override_wins(self):
        from livekit.agents.llm.async_toolset import AsyncToolset

        ts = AsyncToolset(
            id="t",
            tools=[mock_tool_1],
            tool_handling={"async_options": {"update_template": "toolset-own"}},
        )
        session, activity = self._mock_scope(agent_update="agent")
        ts._attach_activity(activity=activity, session=session)
        assert ts._executor._tool_options["update_template"] == "toolset-own"

    def test_toolset_inherits_agent_when_no_override(self):
        from livekit.agents.llm.async_toolset import AsyncToolset

        ts = AsyncToolset(id="t", tools=[mock_tool_1])
        session, activity = self._mock_scope(agent_update="agent")
        ts._attach_activity(activity=activity, session=session)
        # whole-value override: only `update_template` was given on agent, the rest fall
        # back to module defaults (NOT to session options)
        assert ts._executor._tool_options["update_template"] == "agent"

    def test_toolset_inherits_session_when_no_agent_no_override(self):
        from livekit.agents.llm.async_toolset import AsyncToolset

        ts = AsyncToolset(id="t", tools=[mock_tool_1])
        session, activity = self._mock_scope()
        ts._attach_activity(activity=activity, session=session)
        assert ts._executor._tool_options["update_template"] == "session"

    def test_session_scoped_toolset_skips_agent(self):
        # _attach_activity(activity=None) marks the toolset as session-scoped;
        # agent options are ignored even if present.
        from livekit.agents.llm.async_toolset import AsyncToolset

        ts = AsyncToolset(id="t", tools=[mock_tool_1])
        session, _activity = self._mock_scope(agent_update="agent-should-be-ignored")
        ts._attach_activity(activity=None, session=session)
        assert ts._executor._tool_options["update_template"] == "session"
        assert ts._executor._owning_activity is None

    def test_build_executor_map_routes_toolset_tools(self):
        from livekit.agents.llm.async_toolset import AsyncToolset
        from livekit.agents.voice.tool_executor import _build_executor_map, _ToolExecutor

        ts = AsyncToolset(id="async-1", tools=[mock_tool_1])
        default = _ToolExecutor()
        mapping = _build_executor_map(toolsets=[ts], default=default)

        # tool inside AsyncToolset routes to that toolset's executor
        assert mapping["mock_tool_1"] is ts._executor
        # tools not in the map fall back to default at the call site
        assert mapping.get("not_in_map") is None

    def test_build_executor_map_nested_async_toolset_wins(self):
        from livekit.agents.llm.async_toolset import AsyncToolset
        from livekit.agents.llm.tool_context import Toolset
        from livekit.agents.voice.tool_executor import _build_executor_map, _ToolExecutor

        inner = AsyncToolset(id="inner", tools=[mock_tool_2])
        outer_async = AsyncToolset(id="outer", tools=[mock_tool_1, inner])

        # routing keeps inner's executor for inner's tools, outer's for outer's
        mapping = _build_executor_map(toolsets=[outer_async], default=_ToolExecutor())
        assert mapping["mock_tool_1"] is outer_async._executor
        assert mapping["mock_tool_2"] is inner._executor

        # plain Toolset doesn't change executor; its tools route to the surrounding default
        plain = Toolset(id="plain", tools=[mock_tool_3])
        default = _ToolExecutor()
        mapping2 = _build_executor_map(toolsets=[plain], default=default)
        assert mapping2["mock_tool_3"] is default

    def test_session_stores_resolved_options(self):
        # session-level options are resolved and stored at __init__; the actual
        # wiring onto toolset executors happens later at activity start (so
        # toolsets added after session.__init__ are picked up).
        from livekit.agents.voice.agent_session import AgentSession

        session = AgentSession(tool_handling={"async_options": {"update_template": "from-session"}})
        assert session._async_tool_options["update_template"] == "from-session"
        # other keys fall back to defaults, not to anything else
        assert "{function_name}" in session._async_tool_options["duplicate_reject_template"]

    def test_agent_stores_raw_options(self):
        from livekit.agents.utils.misc import is_given
        from livekit.agents.voice.agent import Agent

        agent = Agent(
            instructions="x",
            tool_handling={"async_options": {"update_template": "from-agent"}},
        )
        assert is_given(agent._async_tool_options)
        assert agent._async_tool_options["update_template"] == "from-agent"


# --- helpers for executor / RunContext tests ----------------------------------


def _make_run_context(
    call_id: str = "call_1",
    name: str = "test_tool",
    arguments: str = "{}",
    extra: dict[str, Any] | None = None,
    allow_interruptions: bool = True,
):
    """Build a minimal RunContext — only what the executor and update() actually read."""
    from unittest.mock import MagicMock

    from livekit.agents.llm import FunctionCall
    from livekit.agents.voice.events import RunContext

    speech_handle = MagicMock()
    speech_handle.num_steps = 1
    speech_handle.allow_interruptions = allow_interruptions

    fnc_call = FunctionCall(
        call_id=call_id,
        name=name,
        arguments=arguments,
        extra=extra if extra is not None else {},
    )
    return RunContext(
        session=MagicMock(),
        speech_handle=speech_handle,
        function_call=fnc_call,
    )


@pytest.fixture(autouse=False)
def _clear_running_tasks():
    """Wipe the module-level registry between tests to avoid cross-test bleed."""
    from livekit.agents.voice.tool_executor import _RunningTasks

    _RunningTasks.clear()
    yield
    _RunningTasks.clear()


class TestRunContextUpdate:
    """RunContext.update() — call_id suffix, extra-dict isolation, ordering."""

    @pytest.mark.asyncio
    async def test_first_update_keeps_original_call_id(self):
        """First update reuses the original call_id (so realtime/response models can
        match it server-side); later updates get an ``_update_N`` suffix."""
        ctx = _make_run_context(call_id="orig")
        await ctx.update("msg1")
        await ctx.update("msg2")

        assert ctx._updates[0][0].call_id == "orig"
        assert ctx._updates[1][0].call_id == "orig_update_1"

    @pytest.mark.asyncio
    async def test_synthesized_pair_extra_is_copy(self):
        """Each synthesized pair gets its own ``extra`` dict so later mutation doesn't leak back."""
        ctx = _make_run_context(call_id="orig", extra={"k": "v"})
        await ctx.update("msg1")

        ctx.function_call.extra["leaked"] = True

        assert "leaked" not in ctx._updates[0][0].extra
        assert ctx._updates[0][0].extra == {"k": "v"}

    @pytest.mark.asyncio
    async def test_updates_recorded_without_executor(self):
        """With no executor attached, update() just appends to ``_updates`` and returns."""
        ctx = _make_run_context()
        await ctx.update("first")
        await ctx.update("second")
        assert len(ctx._updates) == 2
        assert "first" in ctx._updates[0][1].output
        assert "second" in ctx._updates[1][1].output

    @pytest.mark.asyncio
    async def test_update_accepts_callable_template(self):
        """``template=`` may be a callable receiving the placeholder dict."""
        ctx = _make_run_context(call_id="orig", name="fn")
        await ctx.update(
            "hello", template=lambda c: f"[{c['function_name']}/{c['call_id']}] {c['message']}"
        )
        assert ctx._updates[0][1].output == "[fn/orig] hello"


class TestAsyncToolOptionsRendering:
    """``_render`` dispatches between ``str.format`` and callable templates."""

    def test_render_string(self):
        from livekit.agents.voice.tool_executor import _render

        assert _render("hi {name}", {"name": "world"}) == "hi world"

    def test_render_callable(self):
        from livekit.agents.voice.tool_executor import _render

        assert _render(lambda c: f"hi {c['name']}", {"name": "world"}) == "hi world"


class TestExecuteFunctionCallWithUpdate:
    """execute_function_call wires ctx.update() into FunctionCallResult.fnc_call_updates."""

    @pytest.mark.asyncio
    async def test_fnc_call_updates_populated(self):
        from livekit.agents.llm.llm import FunctionToolCall
        from livekit.agents.llm.utils import execute_function_call
        from livekit.agents.voice.events import RunContext

        @function_tool
        async def progress_tool(ctx: RunContext, query: str) -> str:
            """Look up something.

            Args:
                query: the query string
            """
            await ctx.update("searching...")
            return f"result for {query}"

        tool_ctx = ToolContext([progress_tool])
        run_ctx = _make_run_context(call_id="probe", name="progress_tool")
        tool_call = FunctionToolCall(
            name="progress_tool", arguments='{"query": "x"}', call_id="probe"
        )

        result = await execute_function_call(tool_call, tool_ctx, call_ctx=run_ctx)

        assert result.fnc_call_out is not None
        assert "result for x" in result.fnc_call_out.output
        assert len(result.fnc_call_updates) == 1
        assert "searching" in result.fnc_call_updates[0][1].output


class TestHasCancellableTool:
    """`has_cancellable_tool` decides whether AgentActivity auto-exposes
    cancel_task / get_running_tasks."""

    def test_plain_tools_not_cancellable(self):
        from livekit.agents.voice.tool_executor import has_cancellable_tool

        assert has_cancellable_tool([mock_tool_1, mock_tool_2]) is False

    def test_one_cancellable_tool_returns_true(self):
        from livekit.agents.voice.tool_executor import has_cancellable_tool

        @function_tool(flags=ToolFlag.CANCELLABLE)
        async def long_running() -> str:
            return "done"

        assert has_cancellable_tool([mock_tool_1, long_running]) is True

    def test_recurses_into_toolsets(self):
        from livekit.agents.llm.async_toolset import AsyncToolset
        from livekit.agents.voice.tool_executor import has_cancellable_tool

        @function_tool(flags=ToolFlag.CANCELLABLE)
        async def long_running() -> str:
            return "done"

        ts = AsyncToolset(id="t", tools=[long_running])
        assert has_cancellable_tool([mock_tool_1, ts]) is True


def _register_fake(
    executor, call_id: str, name: str, *, allow_cancellation: bool, allow_interruptions: bool = True
):
    """Stub a _RunningTask into the executor so policy methods can be tested
    without choreographing real execute() lifetimes. Caller must clean up the
    returned task via _cleanup_fakes."""
    import asyncio as _asyncio

    from livekit.agents.voice.tool_executor import _RunningTask

    async def _runner():
        try:
            await _asyncio.Event().wait()
        except _asyncio.CancelledError:
            return

    exe_task = _asyncio.create_task(_runner(), name=f"fake_{call_id}")
    run_ctx = _make_run_context(call_id=call_id, name=name, allow_interruptions=allow_interruptions)
    executor._running_tasks[call_id] = _RunningTask(
        ctx=run_ctx,
        exe_task=exe_task,
        executor=executor,
        allow_cancellation=allow_cancellation,
    )
    return exe_task


async def _cleanup_fakes(*exe_tasks):
    import asyncio as _asyncio

    for t in exe_tasks:
        if not t.done():
            t.cancel()
    await _asyncio.gather(*exe_tasks, return_exceptions=True)


class TestCheckDuplicate:
    """_check_duplicate policy decisions, exercised with a pre-populated registry."""

    @pytest.mark.asyncio
    async def test_allow_returns_none(self):
        from livekit.agents.voice.tool_executor import _ToolExecutor

        executor = _ToolExecutor()
        t = _register_fake(executor, "a", "tool_x", allow_cancellation=True)
        try:
            assert (
                await executor._check_duplicate(
                    "tool_x", on_duplicate="allow", confirm_duplicate=None
                )
                is None
            )
        finally:
            await _cleanup_fakes(t)

    @pytest.mark.asyncio
    async def test_reject_returns_message(self):
        from livekit.agents.voice.tool_executor import _ToolExecutor

        executor = _ToolExecutor()
        t = _register_fake(executor, "a", "tool_x", allow_cancellation=True)
        try:
            result = await executor._check_duplicate(
                "tool_x", on_duplicate="reject", confirm_duplicate=None
            )
            assert isinstance(result, str) and "already running" in result
        finally:
            await _cleanup_fakes(t)

    @pytest.mark.asyncio
    async def test_confirm_blocks_without_param(self):
        from livekit.agents.voice.tool_executor import _ToolExecutor

        executor = _ToolExecutor()
        t = _register_fake(executor, "a", "tool_x", allow_cancellation=True)
        try:
            result = await executor._check_duplicate(
                "tool_x", on_duplicate="confirm", confirm_duplicate=False
            )
            assert isinstance(result, str) and "confirm duplicate" in result.lower()
            # with confirm=True, the policy lets the new call through
            assert (
                await executor._check_duplicate(
                    "tool_x", on_duplicate="confirm", confirm_duplicate=True
                )
                is None
            )
        finally:
            await _cleanup_fakes(t)

    @pytest.mark.asyncio
    async def test_replace_cancels_when_cancellable(self):
        from livekit.agents.voice.tool_executor import _ToolExecutor

        executor = _ToolExecutor()
        t = _register_fake(executor, "a", "tool_x", allow_cancellation=True)
        try:
            assert (
                await executor._check_duplicate(
                    "tool_x", on_duplicate="replace", confirm_duplicate=None
                )
                is None
            )
            assert t.cancelled() or t.done()
        finally:
            await _cleanup_fakes(t)

    @pytest.mark.asyncio
    async def test_replace_raises_when_non_cancellable(self):
        """replace must honor allow_cancellation, not bypass it."""
        from livekit.agents.llm.tool_context import ToolError
        from livekit.agents.voice.tool_executor import _ToolExecutor

        executor = _ToolExecutor()
        t = _register_fake(executor, "a", "tool_x", allow_cancellation=False)
        try:
            with pytest.raises(ToolError, match="not cancellable"):
                await executor._check_duplicate(
                    "tool_x", on_duplicate="replace", confirm_duplicate=None
                )
            assert not t.cancelled()  # the running tool was left alone
        finally:
            await _cleanup_fakes(t)

    @pytest.mark.asyncio
    async def test_duplicate_check_lock_serializes(self):
        """Holding the executor's lock blocks any concurrent _check_duplicate."""
        import asyncio as _asyncio

        from livekit.agents.voice.tool_executor import _ToolExecutor

        executor = _ToolExecutor()
        t = _register_fake(executor, "a", "tool_x", allow_cancellation=True)
        try:
            await executor._duplicate_check_lock.acquire()
            pending = _asyncio.create_task(
                executor._check_duplicate("tool_x", on_duplicate="reject", confirm_duplicate=None)
            )
            await _asyncio.sleep(0)
            assert not pending.done()  # blocked on the lock
            executor._duplicate_check_lock.release()
            result = await pending
            assert isinstance(result, str) and "already running" in result
        finally:
            await _cleanup_fakes(t)


class TestCancelAll:
    """cancel_all / drain semantics, exercised with a pre-populated registry."""

    @pytest.mark.asyncio
    async def test_cancellable_only_cancels_flagged(self):
        from livekit.agents.voice.tool_executor import _ToolExecutor

        executor = _ToolExecutor()
        cancellable = _register_fake(executor, "c", "tool_c", allow_cancellation=True)
        non_cancellable = _register_fake(executor, "nc", "tool_nc", allow_cancellation=False)
        try:
            # drain cancels the cancellable but awaits the non-cancellable
            import asyncio as _asyncio

            drain_task = _asyncio.create_task(executor.drain())
            await cancellable
            assert cancellable.done()
            assert not drain_task.done()
            assert not non_cancellable.done()
        finally:
            non_cancellable.cancel()
            await _cleanup_fakes(cancellable, non_cancellable)
            await drain_task

    @pytest.mark.asyncio
    async def test_cancel_raises_when_interruptions_disallowed(self):
        """cancel() refuses when the speech handle has ``allow_interruptions=False`` —
        the same predicate gates drain and LLM-driven cancel."""
        from livekit.agents.llm.tool_context import ToolError
        from livekit.agents.voice.tool_executor import _ToolExecutor

        executor = _ToolExecutor()
        t = _register_fake(
            executor, "a", "tool_x", allow_cancellation=True, allow_interruptions=False
        )
        try:
            with pytest.raises(ToolError, match="interruptions are disallowed"):
                await executor.cancel("a")
        finally:
            await _cleanup_fakes(t)

    @pytest.mark.asyncio
    async def test_cancel_task_companion_misses_non_cancellable(self):
        """The LLM-facing ``cancel_task`` raises ToolError for tools registered with
        ``allow_cancellation=False``."""
        from livekit.agents.llm.tool_context import ToolError
        from livekit.agents.voice.tool_executor import _RunningTasks, _ToolExecutor, cancel_task

        executor = _ToolExecutor()
        t = _register_fake(executor, "a", "tool_x", allow_cancellation=False)
        # cancel_task reads from the module-level registry, keyed by session
        run_ctx = executor._running_tasks["a"].ctx
        _RunningTasks[run_ctx.session] = executor._running_tasks
        try:
            with pytest.raises(ToolError, match="not cancellable"):
                await cancel_task(run_ctx, "a")
            assert not t.cancelled()
        finally:
            _RunningTasks.pop(run_ctx.session, None)
            await _cleanup_fakes(t)


class TestToolExecutorLifecycle:
    pytestmark = pytest.mark.usefixtures("_clear_running_tasks")

    @pytest.mark.asyncio
    async def test_run_context_back_refs_cleared_on_finish(self):
        """RunContext drops its executor refs once the tool finishes — a stashed
        ctx can't drive the executor post-completion."""
        import asyncio as _asyncio

        from livekit.agents.voice.tool_executor import _ToolExecutor

        @function_tool
        async def quick_tool() -> str:
            """q"""
            return "ok"

        executor = _ToolExecutor()
        run_ctx = _make_run_context(call_id="a", name="quick_tool")
        result = await executor.execute(tool=quick_tool, run_ctx=run_ctx, raw_arguments={})
        assert result == "ok"

        await _asyncio.sleep(0)  # let _on_done fire
        assert run_ctx._executor is None
        assert run_ctx._first_update_fut is None


class TestAgentSessionWaitForIdle:
    def test_raises_runtime_error_when_no_activity(self):
        """wait_for_idle raises instead of spinning when no activity has started."""
        import asyncio as _asyncio

        from livekit.agents.voice.agent_session import AgentSession

        session = AgentSession()
        with pytest.raises(RuntimeError, match="no active AgentActivity"):
            _asyncio.get_event_loop().run_until_complete(session.wait_for_idle())


# --- tool status events ---------------------------------------------------------


def _emitted_items(session: Any) -> list[Any]:
    """Updates carried by tool_execution_updated events emitted on a mocked session."""
    from livekit.agents.voice.events import ToolExecutionUpdatedEvent

    items = []
    for call in session.emit.call_args_list:
        name, ev = call.args
        if name == "tool_execution_updated":
            assert isinstance(ev, ToolExecutionUpdatedEvent)
            items.append(ev.update)
    return items


async def _drain_executor(executor: Any) -> None:
    """Yield until in-flight tool tasks settle, so their terminal events are emitted."""
    import asyncio as _asyncio

    while executor.has_running_tasks:
        await _asyncio.sleep(0)


def _make_fake_speech():
    """A SpeechHandle stand-in that lets tests fire the done callbacks manually."""
    from unittest.mock import MagicMock

    from livekit.agents.voice import SpeechHandle

    speech = MagicMock(spec=SpeechHandle)
    speech.id = "speech_1"
    speech.interrupted = False
    speech.chat_items = ["said something"]
    callbacks: list[Any] = []
    speech.add_done_callback.side_effect = callbacks.append
    speech.fire_done = lambda: [cb(speech) for cb in list(callbacks)]
    return speech


def _make_reply_session(speech: Any) -> Any:
    """A session mock with just enough surface for _enqueue_reply/_deliver_reply."""
    from unittest.mock import AsyncMock, MagicMock

    from livekit.agents.llm import ChatContext

    session = MagicMock()
    agent = MagicMock()
    agent.chat_ctx = ChatContext.empty()
    agent.update_chat_ctx = AsyncMock()
    session.current_agent = agent
    session._global_run_state = None
    activity = MagicMock()
    activity.agent = agent
    session.wait_for_idle = AsyncMock(return_value=activity)
    session.generate_reply = MagicMock(return_value=speech)
    return session


def _make_run_context_with_session(session: Any, call_id: str, name: str):
    from unittest.mock import MagicMock

    from livekit.agents.llm import FunctionCall
    from livekit.agents.voice.events import RunContext

    speech_handle = MagicMock()
    speech_handle.num_steps = 1
    speech_handle.allow_interruptions = True
    return RunContext(
        session=session,
        speech_handle=speech_handle,
        function_call=FunctionCall(call_id=call_id, name=name, arguments="{}"),
    )


class TestToolCallEvents:
    """tool_execution_updated emission across the executor lifecycle."""

    pytestmark = pytest.mark.usefixtures("_clear_running_tasks")

    @pytest.mark.asyncio
    async def test_sync_tool_started_then_done(self):
        from livekit.agents.voice.events import ToolCallEnded, ToolCallStarted
        from livekit.agents.voice.tool_executor import _ToolExecutor

        @function_tool
        async def quick_tool() -> str:
            """q"""
            return "ok"

        executor = _ToolExecutor()
        run_ctx = _make_run_context(call_id="c1", name="quick_tool")
        result = await executor.execute(tool=quick_tool, run_ctx=run_ctx, raw_arguments={})
        assert result == "ok"
        await _drain_executor(executor)

        items = _emitted_items(run_ctx.session)
        assert isinstance(items[0], ToolCallStarted)
        assert items[0].function_call.call_id == "c1"
        assert items[1] == ToolCallEnded(id="c1", call_id="c1", message="ok", status="done")

    @pytest.mark.asyncio
    async def test_error_before_update_uses_plain_call_id(self):
        from livekit.agents.voice.events import ToolCallEnded, ToolCallStarted
        from livekit.agents.voice.tool_executor import _ToolExecutor

        @function_tool
        async def boom() -> str:
            """b"""
            raise RuntimeError("nope")

        executor = _ToolExecutor()
        run_ctx = _make_run_context(call_id="c2", name="boom")
        with pytest.raises(RuntimeError, match="nope"):
            await executor.execute(tool=boom, run_ctx=run_ctx, raw_arguments={})
        await _drain_executor(executor)

        items = _emitted_items(run_ctx.session)
        assert isinstance(items[0], ToolCallStarted)
        assert items[1] == ToolCallEnded(id="c2", call_id="c2", message="nope", status="error")

    @pytest.mark.asyncio
    async def test_cancelled_tool(self):
        import asyncio as _asyncio

        from livekit.agents.voice.events import ToolCallEnded
        from livekit.agents.voice.tool_executor import _ToolExecutor

        running = _asyncio.Event()

        @function_tool(flags=ToolFlag.CANCELLABLE)
        async def long_tool() -> str:
            """l"""
            running.set()
            await _asyncio.sleep(3600)
            return "never"

        executor = _ToolExecutor()
        run_ctx = _make_run_context(call_id="c3", name="long_tool")
        exec_task = _asyncio.create_task(
            executor.execute(tool=long_tool, run_ctx=run_ctx, raw_arguments={})
        )
        await running.wait()
        assert await executor.cancel("c3")
        assert await exec_task is None

        items = _emitted_items(run_ctx.session)
        assert items[-1] == ToolCallEnded(id="c3", call_id="c3", message=None, status="cancelled")

    @pytest.mark.asyncio
    async def test_cancel_before_first_step_releases_dispatch(self):
        """A cancel landing before the exe task ever runs must still resolve
        execute() and report the cancellation (the handler never gets to run)."""
        import asyncio as _asyncio

        from livekit.agents.voice.events import ToolCallEnded
        from livekit.agents.voice.tool_executor import _ToolExecutor

        @function_tool(flags=ToolFlag.CANCELLABLE)
        async def long_tool() -> str:
            """l"""
            await _asyncio.sleep(3600)
            return "never"

        executor = _ToolExecutor()
        run_ctx = _make_run_context(call_id="c3b", name="long_tool")
        exec_task = _asyncio.create_task(
            executor.execute(tool=long_tool, run_ctx=run_ctx, raw_arguments={})
        )
        # registration happens synchronously inside execute(); the exe task has not
        # run a single step yet when cancel() fires
        while "c3b" not in executor._running_tasks:
            await _asyncio.sleep(0)
        assert await executor.cancel("c3b")
        assert await _asyncio.wait_for(exec_task, timeout=5) is None

        items = _emitted_items(run_ctx.session)
        assert items[-1] == ToolCallEnded(id="c3b", call_id="c3b", message=None, status="cancelled")

    @pytest.mark.asyncio
    async def test_internal_tools_tracked(self):
        from livekit.agents.voice.events import ToolCallEnded, ToolCallStarted
        from livekit.agents.voice.tool_executor import _ToolExecutor

        @function_tool(name="lk_agents_test_tool")
        async def internal_tool() -> str:
            """i"""
            return "ok"

        executor = _ToolExecutor()
        run_ctx = _make_run_context(call_id="c4", name="lk_agents_test_tool")
        await executor.execute(tool=internal_tool, run_ctx=run_ctx, raw_arguments={})
        await _drain_executor(executor)

        items = _emitted_items(run_ctx.session)
        assert isinstance(items[0], ToolCallStarted)
        assert items[1] == ToolCallEnded(id="c4", call_id="c4", message="ok", status="done")

    @pytest.mark.asyncio
    async def test_updates_and_deferred_result_with_reply_lifecycle(self):
        from livekit.agents.voice.events import (
            RunContext,
            ToolCallEnded,
            ToolCallStarted,
            ToolCallUpdated,
            ToolReplyUpdated,
        )
        from livekit.agents.voice.tool_executor import _ToolExecutor

        @function_tool
        async def progress_tool(ctx: RunContext) -> str:
            """p"""
            await ctx.update("step one")
            await ctx.update("step two")
            return "all done"

        import asyncio as _asyncio

        speech = _make_fake_speech()
        session = _make_reply_session(speech)
        # hold the session busy until the tool has buffered everything, so the
        # deferred reply coalesces the second update and the final result
        idle_event = _asyncio.Event()
        activity = session.wait_for_idle.return_value

        async def _wait_for_idle():
            await idle_event.wait()
            return activity

        session.wait_for_idle = _wait_for_idle

        executor = _ToolExecutor()
        run_ctx = _make_run_context_with_session(session, call_id="c5", name="progress_tool")

        first = await executor.execute(tool=progress_tool, run_ctx=run_ctx, raw_arguments={})
        assert "step one" in first
        while executor.has_running_tasks:
            await _asyncio.sleep(0)
        idle_event.set()
        assert executor._reply_task is not None
        await executor._reply_task

        items = _emitted_items(session)
        assert isinstance(items[0], ToolCallStarted)
        # first update is inline (plain call_id), the second is buffered
        assert items[1] == ToolCallUpdated(id="c5", call_id="c5", message="step one")
        assert items[2] == ToolCallUpdated(id="c5_update_1", call_id="c5", message="step two")
        # the final return is deferred through the coalescer
        assert items[3] == ToolCallEnded(
            id="c5_final", call_id="c5", message="all done", status="done"
        )
        # the deferred reply covering the buffered ids was scheduled
        reply = items[4]
        assert isinstance(reply, ToolReplyUpdated)
        assert reply.status == "scheduled"
        assert reply.update_ids == ["c5_update_1", "c5_final"]

        speech.fire_done()
        completed = _emitted_items(session)[-1]
        assert isinstance(completed, ToolReplyUpdated)
        assert completed.status == "completed"
        assert completed.update_ids == ["c5_update_1", "c5_final"]

    @pytest.mark.asyncio
    async def test_interrupted_and_skipped_reply_outcomes(self):
        from livekit.agents.voice.events import RunContext, ToolReplyUpdated
        from livekit.agents.voice.tool_executor import _ToolExecutor

        @function_tool
        async def progress_tool(ctx: RunContext) -> str:
            """p"""
            await ctx.update("working")
            return "done"

        for interrupted, chat_items, expected in [
            (True, ["partial"], "interrupted"),
            (False, [], "skipped"),
        ]:
            speech = _make_fake_speech()
            speech.interrupted = interrupted
            speech.chat_items = chat_items
            session = _make_reply_session(speech)
            executor = _ToolExecutor()
            run_ctx = _make_run_context_with_session(session, call_id="c6", name="progress_tool")

            await executor.execute(tool=progress_tool, run_ctx=run_ctx, raw_arguments={})
            while executor._reply_task is None:
                import asyncio as _asyncio

                await _asyncio.sleep(0)
            await executor._reply_task

            speech.fire_done()
            last = _emitted_items(session)[-1]
            assert isinstance(last, ToolReplyUpdated)
            assert last.status == expected

    @pytest.mark.asyncio
    async def test_error_after_update_is_deferred_with_final_id(self):
        from livekit.agents.voice.events import RunContext, ToolCallEnded
        from livekit.agents.voice.tool_executor import _ToolExecutor

        @function_tool
        async def update_then_boom(ctx: RunContext) -> str:
            """u"""
            await ctx.update("starting")
            raise RuntimeError("late failure")

        speech = _make_fake_speech()
        session = _make_reply_session(speech)
        executor = _ToolExecutor()
        run_ctx = _make_run_context_with_session(session, call_id="c7", name="update_then_boom")

        await executor.execute(tool=update_then_boom, run_ctx=run_ctx, raw_arguments={})
        while executor._reply_task is None:
            import asyncio as _asyncio

            await _asyncio.sleep(0)
        await executor._reply_task

        items = _emitted_items(session)
        terminal = next(i for i in items if isinstance(i, ToolCallEnded))
        assert terminal == ToolCallEnded(
            id="c7_final", call_id="c7", message="late failure", status="error"
        )

    @pytest.mark.asyncio
    async def test_reply_spans_multiple_executions(self):
        import asyncio as _asyncio

        from livekit.agents.voice.events import RunContext, ToolReplyUpdated
        from livekit.agents.voice.tool_executor import _ToolExecutor

        @function_tool
        async def progress_tool(ctx: RunContext) -> str:
            """p"""
            await ctx.update("working")
            return "done"

        speech = _make_fake_speech()
        session = _make_reply_session(speech)
        # hold the session busy until both executions have buffered their results
        idle_event = _asyncio.Event()
        activity = session.wait_for_idle.return_value

        async def _wait_for_idle():
            await idle_event.wait()
            return activity

        session.wait_for_idle = _wait_for_idle

        executor = _ToolExecutor()
        ctx_a = _make_run_context_with_session(session, call_id="a", name="progress_tool")
        ctx_b = _make_run_context_with_session(session, call_id="b", name="progress_tool")

        await executor.execute(tool=progress_tool, run_ctx=ctx_a, raw_arguments={})
        await executor.execute(tool=progress_tool, run_ctx=ctx_b, raw_arguments={})
        while executor.has_running_tasks:
            await _asyncio.sleep(0)
        idle_event.set()
        assert executor._reply_task is not None
        await executor._reply_task

        scheduled = [i for i in _emitted_items(session) if isinstance(i, ToolReplyUpdated)]
        assert len(scheduled) == 1
        assert scheduled[0].status == "scheduled"
        assert set(scheduled[0].update_ids) == {"a_final", "b_final"}
