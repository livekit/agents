"""History conversion and tool-call correlation at the public ChatContext boundary."""

from typing import Any

import pytest

from livekit.agents.llm import ChatContext, ChatMessage, FunctionCall, FunctionCallOutput

pytestmark = pytest.mark.unit


# formats that send tool call arguments as a JSON object (vs. an opaque string like openai/mistral)
_JSON_OBJECT_FORMATS = ["anthropic", "google", "aws"]


def _tool_call_input(fmt: str, messages: list[dict[str, Any]]) -> Any:
    """Pull the single tool call's arguments out of a provider-formatted context."""
    if fmt == "google":
        for turn in messages:
            for part in turn["parts"]:
                if "function_call" in part:
                    return part["function_call"]["args"]
    elif fmt == "anthropic":
        for msg in messages:
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    return block["input"]
    elif fmt == "aws":
        for msg in messages:
            for block in msg["content"]:
                if isinstance(block, dict) and "toolUse" in block:
                    return block["toolUse"]["input"]
    raise AssertionError(f"no tool call found in {fmt} messages")


@pytest.mark.parametrize("fmt", _JSON_OBJECT_FORMATS)
@pytest.mark.asyncio
async def test_to_provider_format_tolerates_unparseable_tool_arguments(fmt: str):
    """A stored tool call whose arguments never parsed must not crash a later turn.

    `FunctionCall.arguments` is kept verbatim when it can't be parsed (unrecoverable
    open-weight output, or history restored via `ChatContext.from_dict`). The
    Anthropic/Google/AWS formatters send arguments as a JSON object, so formatting
    such history previously raised `json.JSONDecodeError`. It should degrade to an
    empty object instead.
    """
    ctx = ChatContext.empty()
    ctx.insert(FunctionCall(call_id="c1", name="lookup", arguments="not-a-json-object"))
    ctx.insert(FunctionCallOutput(call_id="c1", name="lookup", output="tool error", is_error=True))

    messages, _ = await ctx.to_provider_format(format=fmt)
    assert _tool_call_input(fmt, messages) == {}


@pytest.mark.parametrize("fmt", _JSON_OBJECT_FORMATS)
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ('{"a": 1,}', {"a": 1}),
        ('{"a": <|end|>1}', {"a": "end1"}),
        ('"{\\"a\\": 1}"', {"a": 1}),
    ],
    ids=["trailing-comma", "chat-template-leak", "double-encoded"],
)
@pytest.mark.asyncio
async def test_to_provider_format_recovers_repairable_tool_arguments(
    fmt: str, arguments: str, expected: dict[str, Any]
):
    """Recoverable arguments are repaired rather than discarded.

    Routing through `llm.utils.parse_function_arguments` means json_repair, chat-template
    token stripping and double-encoded unwrapping all apply here too, so history that a
    bare `json.loads` would have dropped to `{}` keeps its real arguments.
    """
    ctx = ChatContext.empty()
    ctx.insert(FunctionCall(call_id="c1", name="lookup", arguments=arguments))
    ctx.insert(FunctionCallOutput(call_id="c1", name="lookup", output="ok", is_error=False))

    messages, _ = await ctx.to_provider_format(format=fmt)
    assert _tool_call_input(fmt, messages) == expected


@pytest.mark.parametrize("fmt", _JSON_OBJECT_FORMATS)
@pytest.mark.parametrize("arguments", ["", "[1, 2, 3]", "42", "null"])
@pytest.mark.asyncio
async def test_to_provider_format_non_object_tool_arguments(fmt: str, arguments: str):
    """Empty or non-object arguments degrade to `{}`; a call's arguments are always a
    named-parameter object, so an array/scalar/null is treated as no arguments."""
    ctx = ChatContext.empty()
    ctx.insert(FunctionCall(call_id="c1", name="lookup", arguments=arguments))
    ctx.insert(FunctionCallOutput(call_id="c1", name="lookup", output="ok", is_error=False))

    messages, _ = await ctx.to_provider_format(format=fmt)
    assert _tool_call_input(fmt, messages) == {}


def test_removing_framework_instructions_preserves_authored_system_messages() -> None:
    from livekit.agents.llm.utils import remove_instructions

    authored = ChatMessage(id="system-prompt", role="system", content=["Keep this instruction."])
    framework = ChatMessage(
        id="lk.agent_task.instructions", role="system", content=["Framework prompt."]
    )
    reply = ChatMessage(id="reply", role="assistant", content=["Hello."])
    context = ChatContext([framework, authored, framework.model_copy(), reply])
    remove_instructions(context)
    assert context.items == [authored, reply]
