"""Regression contracts at the authored tool and native dispatch boundary."""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import AsyncIterable, Callable
from typing import Annotated, Literal, ParamSpec

import pytest
from pydantic import BaseModel, Field

from livekit.agents import (
    Agent,
    ModelSettings,
    RunContext,
    StopResponse,
    ToolError,
    VoiceSession,
    function_tool,
    llm,
)
from livekit.agents.llm import ChatContext, Tool, ToolContext
from livekit.agents.types import FlushSentinel
from tests_new.fakes import FakeLLM, tool_call

P = ParamSpec("P")


class _Selection(BaseModel):
    kind: Literal["selection"] = "selection"
    groups: dict[str, list[int | None]] = Field(default_factory=dict)
    labels: list[str] = ["default"]


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"selection": None},
        {"selection": {"kind": "selection", "groups": {"a": [1, None]}, "labels": None}},
    ],
)
async def test_nested_nullable_arguments_and_container_defaults_reach_the_tool(
    arguments: dict[str, object],
) -> None:
    """Nullable trees preserve nulls and defaults. https://github.com/livekit/agents/issues/2302"""
    seen: list[_Selection | None] = []
    call = llm.FunctionToolCall(name="choose", arguments=json.dumps(arguments), call_id="c1")
    model = FakeLLM([call, "accepted"])
    agent = Agent(instructions="Choose.", llm=model)

    @agent.tool
    def choose(
        ctx: RunContext,
        selection: Annotated[_Selection | None, Field(description="Optional selection")] = None,
    ) -> str:
        """Choose a selection."""
        seen.append(selection)
        return "chosen"

    # Building the real strict provider schema used to fail on nullable container shapes.
    schema = ToolContext([choose]).parse_function_tools("openai")[0]
    encoded = json.dumps(schema)
    assert '"const": "selection"' in encoded
    result = await agent.run("choose")
    assert result.status == "completed" and result.output == "accepted"
    assert len(seen) == 1
    if isinstance(arguments.get("selection"), dict):
        assert seen[0] == _Selection(groups={"a": [1, None]})
    else:
        assert seen == [None]


@pytest.mark.parametrize("value", [False, 0, "", [], {}])
async def test_falsy_tool_results_keep_the_value_name_and_call_id(
    value: object,
) -> None:
    """Falsy results keep their call identity. https://github.com/livekit/agents/pull/6551"""
    model = FakeLLM([tool_call("lookup", call_id="result-17"), "received"])
    agent = Agent(instructions="Look up.", llm=model)

    @agent.tool
    def lookup(ctx: RunContext) -> object:
        """Look up a value."""
        return value

    result = await agent.run("lookup")
    output = next(i for i in result.new_messages() if i.type == "function_call_output")
    assert (output.name, output.call_id, output.output, output.is_error) == (
        "lookup",
        "result-17",
        str(value),
        False,
    )
    assert result.status == "completed" and result.output == "received"
    assert any(
        i.type == "function_call_output" and i.output == str(value)
        for i in model.requests[-1].items
    )


@pytest.mark.parametrize(
    "failure", [ToolError("try a different date"), StopResponse(), RuntimeError("internal-secret")]
)
async def test_tool_exceptions_keep_feedback_control_flow_and_internal_details_separate(
    failure: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    """Tool failures preserve control flow and safe feedback. https://github.com/livekit/agents/pull/1931"""
    model = FakeLLM([tool_call("lookup"), "recovered"])
    agent = Agent(instructions="Look up.", llm=model)

    @agent.tool
    def lookup(ctx: RunContext) -> str:
        """Look up a value."""
        raise failure

    with caplog.at_level(logging.ERROR):
        result = await agent.run("lookup")
    history = result.new_messages()
    output = next(i for i in history if i.type == "function_call_output")
    if isinstance(failure, StopResponse):
        assert not output.is_error and not output.reply_required
        after = history[history.index(output) + 1 :]
        assert not [i for i in after if i.type == "message" and i.role == "assistant"]
        assert not caplog.records
    else:
        assert output.is_error
        assert result.status == "completed" and result.output == "recovered"
        if isinstance(failure, ToolError):
            assert output.output == str(failure)
        else:
            assert "internal-secret" not in output.output
            assert any(r.exc_info and r.exc_info[1] is failure for r in caplog.records)


async def test_tool_schema_preserves_parameter_docs_constraints_and_annotated_metadata() -> None:
    """The model sees argument docs and constraints. https://github.com/livekit/agents/issues/4283"""
    model = FakeLLM([tool_call("book", nights=0), tool_call("book", nights=2), "booked"])
    agent = Agent(instructions="Book.", llm=model)

    @agent.tool
    def book(
        ctx: RunContext,
        nights: Annotated[int, Field(ge=1, le=7, description="Stay length")],
        note: str = "",
    ) -> str:
        """Book a stay.

        Args:
            nights: This must not override explicit Field metadata.
            note: A note to the front desk.
        """
        assert 1 <= nights <= 7
        return "reserved"

    # the request body a provider sends
    schema = json.loads(json.dumps(ToolContext([book]).parse_function_tools("openai")[0]))
    properties = schema["function"]["parameters"]["properties"]
    assert properties["nights"]["description"] == "Stay length"
    assert (properties["nights"]["minimum"], properties["nights"]["maximum"]) == (1, 7)
    assert properties["note"]["description"] == "A note to the front desk."
    result = await agent.run("book")
    outputs = [i for i in result.new_messages() if i.type == "function_call_output"]
    assert [o.is_error for o in outputs] == [True, False]
    assert result.status == "completed" and result.output == "booked"


@pytest.mark.parametrize(
    "raw, strict",
    [
        (False, True),
        (True, True),
        (True, False),
        pytest.param(
            False,
            False,
            marks=pytest.mark.xfail(
                strict=True,
                reason="https://github.com/livekit/agents/issues/4610 reproduces: every non-strict openai typed tool schema fails (missing field name)",
                raises=ValueError,
            ),
        ),
    ],
)
async def test_parameterless_and_raw_tools_bind_without_framework_fields(
    raw: bool, strict: bool
) -> None:
    """Empty and raw tool schemas remain callable JSON. https://github.com/livekit/agents/issues/4610"""
    seen: list[dict[str, object]] = []
    if raw:

        @function_tool(
            raw_schema={"name": "lookup", "parameters": {"type": "object", "properties": {}}}
        )
        def lookup(raw_arguments: dict[str, object]) -> str:
            seen.append(raw_arguments)
            return "found"
    else:

        @function_tool
        def lookup() -> str:
            """Look up."""
            seen.append({})
            return "found"

    schema = ToolContext([lookup]).parse_function_tools("openai", strict=strict)[0]
    wire = json.loads(json.dumps(schema))
    assert wire["function"]["parameters"]["type"] == "object"
    assert wire["function"]["parameters"].get("properties", {}) == {}
    assert "flags" not in wire["function"]
    model = FakeLLM([tool_call("lookup"), "done"])
    result = await Agent(instructions="Look up.", tools=[lookup], llm=model).run("lookup")
    assert seen == [{}]
    assert result.status == "completed" and result.output == "done"


async def test_wrapped_generic_context_tool_keeps_its_name_and_application_context() -> None:
    """Wrapped tools bind generic contexts and explicit names. https://github.com/livekit/agents/pull/4674"""
    seen: list[dict[str, str]] = []

    def original(ctx: RunContext[dict[str, str]], name: str) -> str:
        seen.append(ctx.context)
        return name + ctx.context["suffix"]

    def logged(f: Callable[P, str]) -> Callable[P, str]:
        @functools.wraps(f)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> str:
            return f(*args, **kwargs)

        return wrapped

    tool = function_tool(logged(original), name="lookup")
    schema = json.loads(json.dumps(ToolContext([tool]).parse_function_tools("openai")[0]))
    assert schema["function"]["name"] == "lookup"
    assert set(schema["function"]["parameters"]["properties"]) == {"name"}
    context = {"suffix": "!"}
    model = FakeLLM([tool_call("lookup", name="Ada"), "done"])
    result = await Agent(instructions="Help.", tools=[tool], llm=model).run(
        "hello", context=context
    )
    assert seen == [context] and seen[0] is context
    assert (
        next(i.output for i in result.new_messages() if i.type == "function_call_output") == "Ada!"
    )


async def test_invalid_tool_arguments_are_never_logged_in_the_clear(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A private value in arguments that fail validation reaches the model as an error but no
    log line, except under `lk.pii.*` fields. https://github.com/livekit/agents/issues/7801"""
    secret = "private-passport-739291"
    model = FakeLLM([tool_call("book", nights=secret), "please give a number"])
    agent = Agent(instructions="Book.", llm=model)

    @agent.tool
    def book(ctx: RunContext, nights: int) -> str:
        """Book a stay."""
        pytest.fail("invalid arguments reached the body")

    with caplog.at_level(logging.DEBUG):
        result = await agent.run("book")

    result.expect.next_event().is_function_call(name="book")
    result.expect.next_event().is_function_call_output(is_error=True)
    for record in caplog.records:
        assert secret not in record.getMessage()
        clear = [key for key, value in vars(record).items() if secret in str(value)]
        assert all(key.startswith("lk.pii.") for key in clear), clear


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/4355 reproduces: node-added tools reach the model but not native dispatch",
)
async def test_tools_added_inside_llm_node_are_executed_by_native_dispatch() -> None:
    """Node-added tools reach native dispatch. https://github.com/livekit/agents/pull/4355"""
    seen: list[int] = []

    @function_tool
    def lookup(ctx: RunContext, number: int) -> str:
        """Look up a booking."""
        seen.append(number)
        return "booking found"

    model = FakeLLM([tool_call("lookup", number=17), "found it"])
    async with VoiceSession(llm=model) as session:

        @session.llm_node
        def node(
            ctx: RunContext, chat_ctx: ChatContext, tools: list[Tool], settings: ModelSettings
        ) -> AsyncIterable[llm.ChatChunk | str | FlushSentinel]:
            if lookup not in tools:
                tools.append(lookup)
            return session.default.llm_node(ctx, chat_ctx, tools, settings)

        await session.start(Agent(instructions="Help."))
        await session.generate_reply(user_input="find 17")

    output = next(i for i in session.history.items if i.type == "function_call_output")
    assert seen == [17]
    assert output.output == "booking found" and not output.is_error


@pytest.mark.parametrize("mutation", ["insert", "merge", "timestamp", "items"])
async def test_readonly_history_rejects_all_detached_mutation_paths(mutation: str) -> None:
    """A tool's read-only view of the history refuses every mutation instead of dropping it.
    https://github.com/livekit/agents/pull/7445"""
    model = FakeLLM([tool_call("inspect_history"), "done"])
    agent = Agent(instructions="Help.", llm=model)
    checked: list[bool] = []

    @agent.tool
    def inspect_history(ctx: RunContext) -> str:
        """Inspect the current history."""
        readonly = agent.chat_ctx
        assert readonly.readonly
        before = readonly.to_dict()
        message = ChatContext.empty()
        item = message.add_message(role="user", content="must not be inserted", created_at=0)
        with pytest.raises(RuntimeError, match="read-only"):
            if mutation == "insert":
                readonly.insert(item)
            elif mutation == "merge":
                readonly.merge(message)
            elif mutation == "timestamp":
                readonly.add_message(role="user", content="not inserted", created_at=0)
            else:
                readonly.items = [item]
        assert readonly.to_dict() == before
        checked.append(True)
        return "checked"

    async with VoiceSession() as session:
        await session.start(agent)
        await session.generate_reply(user_input="hello")

    assert checked == [True]
    assert all(
        "not be inserted" not in (i.text_content or "")
        for i in session.history.items
        if i.type == "message"
    )


@function_tool
async def lookup_product(product: str) -> str:
    """Look a product up.

    Args:
        product: The product to look up.
    """
    return f"{product}: US numbers only"


async def _speak_call_speak(on_call: bool) -> list[llm.ChatItem]:
    model = FakeLLM(
        [["Let me check.", tool_call("lookup_product", product="phones")], "US numbers only."]
    )
    agent = Agent(instructions="Help.", llm=model, tools=[lookup_product])
    if not on_call:
        return (await agent.run("Can I buy a phone number?")).all_messages()
    async with VoiceSession(llm=model) as session:
        await session.start(agent)
        await session.generate_reply(user_input="Can I buy a phone number?")
        return list(session.history.items)


@pytest.mark.parametrize("on_call", [False, True], ids=["run", "call"])
async def test_a_reply_that_speaks_calls_a_tool_and_speaks_again_goes_back_to_the_model(
    on_call: bool,
) -> None:
    """Each model step is its own assistant message, its calls under it, as in v1: a reply that
    speaks, calls a tool and speaks again can be sent to the model again."""
    items = await _speak_call_speak(on_call)

    messages, _ = await ChatContext(items=items).to_provider_format("openai")

    turns = [m for m in messages if m["role"] in ("assistant", "tool")]
    assert [m["role"] for m in turns] == ["assistant", "tool", "assistant"]
    assert turns[0]["content"] == "Let me check."
    assert [c["function"]["name"] for c in turns[0]["tool_calls"]] == ["lookup_product"]
    assert turns[2]["content"] == "US numbers only."
