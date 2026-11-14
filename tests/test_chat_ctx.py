import base64
import random
import tracemalloc
from typing import Any

import pytest

from livekit.agents.llm import (
    AgentHandoff,
    ChatContext,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
    utils,
)
from livekit.agents.llm.chat_context import _ReadOnlyChatContext
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]

_IMAGE_BYTES = b"fake image bytes"


@pytest.mark.parametrize(
    ("mime_type", "expected_format"),
    [
        ("image/jpeg", "jpeg"),
        ("image/png", "png"),
        ("image/gif", "gif"),
        ("image/webp", "webp"),
    ],
)
def test_aws_image_content_uses_serialized_image_format(mime_type: str, expected_format: str):

    encoded = base64.b64encode(_IMAGE_BYTES).decode("utf-8")
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(
        role="user",
        content=[ImageContent(image=f"data:{mime_type};base64,{encoded}")],
    )

    messages, _ = chat_ctx.to_provider_format(format="aws")

    image = messages[0]["content"][0]["image"]
    assert image["format"] == expected_format
    assert image["source"]["bytes"] == _IMAGE_BYTES


def test_aws_image_content_rejects_external_urls():

    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(
        role="user",
        content=[ImageContent(image="https://example.com/image.png", mime_type="image/png")],
    )

    with pytest.raises(ValueError, match="external_url is not supported by AWS Bedrock"):
        chat_ctx.to_provider_format(format="aws")


def _ctx_with_per_turn_instructions() -> tuple[ChatContext, str]:
    # the shape generate_reply(instructions=...) produces: a trailing system message
    instructions = "Ask the caller for the year they were born."
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="system", content=["You are a helpful assistant."])
    chat_ctx.add_message(role="assistant", content=["Hello! How can I help you?"])
    chat_ctx.add_message(role="user", content=["I'd like to refill my prescription."])
    chat_ctx.add_message(role="system", content=[instructions])
    return chat_ctx, instructions


_AWS_CACHE_POINT = {"cachePoint": {"type": "default"}}


def _aws_cache_points(messages: list[dict]) -> list[tuple[int, int]]:
    return [
        (mi, bi)
        for mi, m in enumerate(messages)
        for bi, block in enumerate(m["content"])
        if block == _AWS_CACHE_POINT
    ]


def test_aws_cache_messages_is_off_by_default():
    chat_ctx, _ = _ctx_with_per_turn_instructions()

    default, _ = chat_ctx.to_provider_format(format="aws")
    disabled, _ = chat_ctx.to_provider_format(format="aws", cache_messages=False)

    assert default == disabled
    assert _aws_cache_points(default) == []


def test_aws_cache_messages_ends_before_per_turn_instructions():
    chat_ctx, instructions = _ctx_with_per_turn_instructions()

    messages, extra = chat_ctx.to_provider_format(format="aws", cache_messages=True)

    assert extra.system_messages == ["You are a helpful assistant."]
    # messages[0] is the "(empty)" user turn Bedrock needs before the greeting
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert _aws_cache_points(messages) == [(2, 1)]
    user_turn = messages[2]["content"]
    assert user_turn[0] == {"text": "I'd like to refill my prescription."}
    assert instructions in user_turn[2]["text"]


def test_aws_cache_messages_ends_after_the_tool_result():
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content=["What's the weather in Paris?"])
    chat_ctx.insert(FunctionCall(call_id="c1", name="get_weather", arguments='{"city": "Paris"}'))
    chat_ctx.insert(
        FunctionCallOutput(call_id="c1", name="get_weather", output="sunny", is_error=False)
    )

    messages, _ = chat_ctx.to_provider_format(format="aws", cache_messages=True)

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[-1]["content"][0]["toolResult"]["toolUseId"] == "c1"
    assert _aws_cache_points(messages) == [(2, 1)]


def test_aws_cache_messages_next_turn_starts_with_the_cached_prefix():
    chat_ctx, _ = _ctx_with_per_turn_instructions()
    first, _ = chat_ctx.to_provider_format(format="aws", cache_messages=True)
    [(mi, bi)] = _aws_cache_points(first)
    cached = [*first[:mi], {**first[mi], "content": first[mi]["content"][:bi]}]

    # next turn: the per-turn instructions are gone, the reply and a new message follow
    chat_ctx.items.pop()
    chat_ctx.add_message(role="assistant", content=["Sure, which pharmacy?"])
    chat_ctx.add_message(role="user", content=["The one on Main Street."])
    second, _ = chat_ctx.to_provider_format(format="aws", cache_messages=True)

    assert second[:mi] == cached[:mi]
    assert second[mi]["content"][:bi] == cached[mi]["content"]
    assert _aws_cache_points(second) == [(len(second) - 1, 1)]


def test_aws_cache_messages_after_an_assistant_turn_ends_that_turn():
    # a reply generated right after the agent spoke: the instructions open a new user turn
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content=["Hi"])
    chat_ctx.add_message(role="assistant", content=["Hello! How can I help you?"])
    chat_ctx.add_message(role="system", content=["Ask if they are still there."])

    messages, _ = chat_ctx.to_provider_format(format="aws", cache_messages=True)

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert _aws_cache_points(messages) == [(1, 1)]


def test_aws_cache_messages_needs_conversation_content():
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="system", content=["You are a helpful assistant."])
    messages, _ = chat_ctx.to_provider_format(format="aws", cache_messages=True)
    assert _aws_cache_points(messages) == []

    # an item with no Bedrock content isn't conversation content either
    chat_ctx.add_message(role="user", content=[""])
    messages, _ = chat_ctx.to_provider_format(format="aws", cache_messages=True)
    assert _aws_cache_points(messages) == []


def test_aws_cache_messages_follows_serialized_order_for_late_tool_results():
    # the older call's result arrives last, but it's serialized next to its call
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content=["Look up both."])
    chat_ctx.insert(FunctionCall(id="item_a", call_id="slow", name="slow_lookup", arguments="{}"))
    chat_ctx.insert(FunctionCall(id="item_b", call_id="fast", name="fast_lookup", arguments="{}"))
    for call_id, name in (("fast", "fast_lookup"), ("slow", "slow_lookup")):
        chat_ctx.insert(
            FunctionCallOutput(call_id=call_id, name=name, output="done", is_error=False)
        )

    messages, _ = chat_ctx.to_provider_format(format="aws", cache_messages=True)

    assert messages[-1]["content"][0]["toolResult"]["toolUseId"] == "fast"
    assert _aws_cache_points(messages) == [(len(messages) - 1, 1)]


def test_aws_cache_messages_empty_item_keeps_the_last_conversation_block():
    # the empty item adds nothing, so the boundary stays after the last real block, and
    # the rewritten instructions in between never become the boundary
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content=["Hello"])
    chat_ctx.add_message(role="system", content=["Ask for the year they were born."])
    chat_ctx.add_message(role="user", content=[""])

    messages, _ = chat_ctx.to_provider_format(format="aws", cache_messages=True)

    assert len(messages) == 1
    assert messages[0]["content"][:2] == [{"text": "Hello"}, _AWS_CACHE_POINT]
    assert "Ask for the year" in messages[0]["content"][2]["text"]
    assert _aws_cache_points(messages) == [(0, 1)]


def test_openai_format_preserves_mid_conversation_system_messages():
    # intentional pass-through: the openai serializer keeps system messages where they are;
    # providers that need repositioning handle it in their own serializer
    chat_ctx, instructions = _ctx_with_per_turn_instructions()

    messages, _ = chat_ctx.to_provider_format(format="openai")

    assert [m["role"] for m in messages] == ["system", "assistant", "user", "system"]
    assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert messages[-1]["content"] == instructions


def test_mistralai_format_converts_mid_conversation_instructions():
    chat_ctx, instructions = _ctx_with_per_turn_instructions()

    entries, extra_data = chat_ctx.to_provider_format(format="mistralai")

    assert extra_data.instructions == "You are a helpful assistant."
    assert entries[-1] == {
        "type": "message.input",
        "role": "user",
        "content": f"<instructions>\n{instructions}\n</instructions>",
    }


def test_per_turn_instructions_convert_without_a_preamble():
    # no base system message: the trailing per-turn message is still mid-conversation
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content=["I'd like to refill my prescription."])
    chat_ctx.add_message(role="system", content=["Ask the caller for the year they were born."])

    entries, extra_data = chat_ctx.to_provider_format(format="mistralai")

    assert extra_data.instructions is None
    assert [e["role"] for e in entries] == ["user", "user"]
    assert entries[-1]["content"].startswith("<instructions>")


def test_empty_mid_conversation_system_messages_are_dropped():
    # a text-less later system message must not shadow the base preamble
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="system", content=["You are a helpful assistant."])
    chat_ctx.add_message(role="user", content=["Hi!"])
    chat_ctx.add_message(role="system", content=[""])

    entries, extra_data = chat_ctx.to_provider_format(format="mistralai")

    assert extra_data.instructions == "You are a helpful assistant."
    assert [e["role"] for e in entries] == ["user"]


def test_chat_ctx_can_be_serialized_and_deserialized_with_defaults():

    items = [
        AgentHandoff(new_agent_id="default_agent", old_agent_id=None),
        ChatMessage(role="user", content=["Hello, world!"]),
        ChatMessage(role="assistant", content=["Hi there!"]),
    ]
    chat_ctx = ChatContext(items)
    assert chat_ctx.is_equivalent(ChatContext.from_dict(chat_ctx.to_dict()))


def test_mistralai_format_injects_trailing_user_message_after_assistant():

    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="Hello")
    chat_ctx.add_message(role="assistant", content="Hi there")

    messages, _ = chat_ctx.to_provider_format(format="mistralai")

    assert messages[-1] == {
        "type": "message.input",
        "role": "user",
        "content": "(empty)",
    }


def test_mistralai_format_can_skip_trailing_user_message_injection():

    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="assistant", content="Hi there")

    messages, _ = chat_ctx.to_provider_format(format="mistralai", inject_dummy_user_message=False)

    assert messages == [{"type": "message.output", "role": "assistant", "content": "Hi there"}]


class _FixedSummaryLLM(FakeLLM):
    """FakeLLM that returns a fixed summary string for any input."""

    def __init__(self, summary: str) -> None:
        super().__init__()
        self._summary = summary

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: Any = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: Any = NOT_GIVEN,
        extra_kwargs: Any = NOT_GIVEN,
    ):
        last_msg = chat_ctx.items[-1]
        input_text = last_msg.text_content
        self._fake_response_map[input_text] = FakeLLMResponse(
            input=input_text,
            content=self._summary,
            ttft=0.0,
            duration=0.0,
        )
        return super().chat(
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )


CANNED_SUMMARY = "User asked about earbuds. Agent resolved the issue."


def _build_conversation_ctx() -> ChatContext:
    """Build a ChatContext with system, user/assistant pairs, and interleaved tool calls."""

    ctx = ChatContext()
    ctx.add_message(role="system", content="You are a helpful assistant.")
    ctx.add_message(role="user", content="Hi, my earbuds are broken.")
    ctx.add_message(role="assistant", content="Can you share your order number?")
    ctx.add_message(role="user", content="Order #123.")
    ctx.items.append(FunctionCall(name="lookup_order", call_id="c1", arguments='{"order": "123"}'))
    ctx.items.append(
        FunctionCallOutput(
            name="lookup_order", call_id="c1", output='{"status":"delivered"}', is_error=False
        )
    )
    ctx.add_message(role="assistant", content="Found your order. Let me check warranty.")
    ctx.add_message(role="user", content="Thanks.")
    ctx.add_message(role="assistant", content="You are under warranty.")
    return ctx


@pytest.mark.asyncio
async def test_summarize_head_tail_split_basic():

    ctx = ChatContext()
    ctx.add_message(role="system", content="System prompt.")
    ctx.add_message(role="user", content="msg1")
    ctx.add_message(role="assistant", content="reply1")
    ctx.add_message(role="user", content="msg2")
    ctx.add_message(role="assistant", content="reply2")
    ctx.add_message(role="user", content="msg3")
    ctx.add_message(role="assistant", content="reply3")

    llm = _FixedSummaryLLM(CANNED_SUMMARY)
    result = await ctx._summarize(llm, keep_last_turns=1)

    tail_msgs = [
        it
        for it in result.items
        if it.type == "message"
        and it.role in ("user", "assistant")
        and not it.extra.get("is_summary")
    ]
    assert len(tail_msgs) == 2
    assert tail_msgs[0].text_content == "msg3"
    assert tail_msgs[1].text_content == "reply3"

    summaries = [it for it in result.items if it.type == "message" and it.extra.get("is_summary")]
    assert len(summaries) == 1
    assert CANNED_SUMMARY in summaries[0].text_content

    system_msgs = [it for it in result.items if it.type == "message" and it.role == "system"]
    assert len(system_msgs) == 1


@pytest.mark.asyncio
async def test_summarize_keeps_prior_summary():
    ctx = ChatContext()
    ctx.add_message(role="system", content="System prompt.")
    ctx.add_message(
        role="assistant",
        content="<chat_history_summary>earlier summary</chat_history_summary>",
        extra={"is_summary": True},
    )
    ctx.add_message(role="user", content="msg1")
    ctx.add_message(role="assistant", content="reply1")
    ctx.add_message(role="user", content="msg2")
    ctx.add_message(role="assistant", content="reply2")

    llm = _FixedSummaryLLM(CANNED_SUMMARY)
    result = await ctx._summarize(llm, keep_last_turns=1)

    summaries = [it for it in result.items if it.type == "message" and it.extra.get("is_summary")]
    texts = [it.text_content for it in summaries]
    assert len(summaries) == 2
    assert any("earlier summary" in t for t in texts)
    assert any(CANNED_SUMMARY in t for t in texts)


@pytest.mark.asyncio
async def test_summarize_head_tail_split_with_renderables():
    ctx = _build_conversation_ctx()

    llm = _FixedSummaryLLM(CANNED_SUMMARY)
    result = await ctx._summarize(llm, keep_last_turns=2)

    # With keep_last_turns=2, the backward walk counts 4 ChatMessages:
    #   "You are under warranty." (1), "Thanks." (2),
    #   "Found your order..." (3), "Order #123." (4) ← split here
    # The FunctionCall + FunctionCallOutput between "Order #123." and
    # "Found your order..." fall inside the tail and must be preserved.
    tail_msgs = [
        it
        for it in result.items
        if it.type == "message"
        and it.role in ("user", "assistant")
        and not it.extra.get("is_summary")
    ]
    assert len(tail_msgs) == 4
    assert tail_msgs[0].text_content == "Order #123."
    assert tail_msgs[1].text_content == "Found your order. Let me check warranty."
    assert tail_msgs[2].text_content == "Thanks."
    assert tail_msgs[3].text_content == "You are under warranty."

    fn_items = [it for it in result.items if it.type in ("function_call", "function_call_output")]
    assert len(fn_items) == 2

    summaries = [it for it in result.items if it.type == "message" and it.extra.get("is_summary")]
    assert len(summaries) == 1


@pytest.mark.asyncio
async def test_summarize_keep_last_turns_zero():
    ctx = _build_conversation_ctx()

    llm = _FixedSummaryLLM(CANNED_SUMMARY)
    result = await ctx._summarize(llm, keep_last_turns=0)

    raw_msgs = [
        it
        for it in result.items
        if it.type == "message"
        and it.role in ("user", "assistant")
        and not it.extra.get("is_summary")
    ]
    assert len(raw_msgs) == 0

    fn_items = [it for it in result.items if it.type in ("function_call", "function_call_output")]
    assert len(fn_items) == 0

    summaries = [it for it in result.items if it.type == "message" and it.extra.get("is_summary")]
    assert len(summaries) == 1

    system_msgs = [it for it in result.items if it.type == "message" and it.role == "system"]
    assert len(system_msgs) == 1


@pytest.mark.asyncio
async def test_summarize_preserves_structural_items():

    ctx = ChatContext()
    ctx.add_message(role="system", content="System prompt.")
    ctx.add_message(role="user", content="Hello.")
    ctx.add_message(role="assistant", content="Hi there.")
    ctx.items.append(AgentHandoff(old_agent_id="AgentA", new_agent_id="AgentB"))
    ctx.add_message(role="user", content="Transfer me.")
    ctx.add_message(role="assistant", content="Done.")
    ctx.add_message(role="user", content="Thanks.")
    ctx.add_message(role="assistant", content="Welcome.")

    llm = _FixedSummaryLLM(CANNED_SUMMARY)
    result = await ctx._summarize(llm, keep_last_turns=1)

    # system message preserved
    system_msgs = [it for it in result.items if it.type == "message" and it.role == "system"]
    assert len(system_msgs) == 1

    # agent handoff preserved
    handoffs = [it for it in result.items if it.type == "agent_handoff"]
    assert len(handoffs) == 1
    assert handoffs[0].old_agent_id == "AgentA"
    assert handoffs[0].new_agent_id == "AgentB"


@pytest.mark.asyncio
async def test_summarize_skips_when_not_enough_messages():

    ctx = ChatContext()
    ctx.add_message(role="system", content="System prompt.")
    ctx.add_message(role="user", content="Hello.")
    ctx.add_message(role="assistant", content="Hi there.")

    original_items = list(ctx.items)

    llm = _FixedSummaryLLM(CANNED_SUMMARY)
    result = await ctx._summarize(llm, keep_last_turns=1)

    # budget covers all messages, so nothing to summarize — early return
    assert len(result.items) == len(original_items)
    for a, b in zip(result.items, original_items, strict=True):
        assert a.id == b.id


def _make_ctx(*roles: str) -> ChatContext:
    ctx = ChatContext()
    for role in roles:
        ctx.add_message(role=role, content=f"msg-{role}")  # type: ignore[arg-type]
    return ctx


@pytest.mark.parametrize(
    ("roles", "max_items", "kept"),
    [
        (["system", "user", "assistant"], 5, ["system", "user", "assistant"]),
        (["user", "assistant", "user", "assistant"], 2, ["user", "assistant"]),
        (["system", "user", "assistant", "user", "assistant"], 2, ["system", "user", "assistant"]),
        (
            ["developer", "user", "assistant", "user", "assistant"],
            2,
            ["developer", "user", "assistant"],
        ),
        # the instruction is already in the tail, so it is not inserted again
        (["system", "user", "assistant"], 3, ["system", "user", "assistant"]),
        (["system", "user", "assistant", "user"], 0, ["system"]),
        (["user", "assistant", "user"], 0, []),
    ],
    ids=[
        "under-limit",
        "basic",
        "keeps-system",
        "keeps-developer",
        "no-duplication",
        "zero-keeps-instruction",
        "zero-without-instruction",
    ],
)
def test_truncate(roles: list[str], max_items: int, kept: list[str]) -> None:
    ctx = _make_ctx(*roles)

    ctx.truncate(max_items=max_items)

    assert [item.role for item in ctx.items] == kept  # type: ignore[union-attr]


def test_truncate_keeps_only_the_first_instruction() -> None:
    ctx = _make_ctx("system", "developer", "user", "user", "user")

    ctx.truncate(max_items=2)

    assert [item.role for item in ctx.items] == ["system", "user", "user"]  # type: ignore[union-attr]


def test_truncate_negative_max_items_raises_and_leaves_the_context_alone() -> None:
    ctx = _make_ctx("developer", "user", "assistant", "user")

    with pytest.raises(ValueError, match="max_items must be non-negative"):
        ctx.truncate(max_items=-2)

    assert [item.role for item in ctx.items] == ["developer", "user", "assistant", "user"]  # type: ignore[union-attr]


def test_truncate_noop_when_under_limit():
    ctx = _make_ctx("system", "user", "assistant")
    original_ids = [item.id for item in ctx.items]
    ctx.truncate(max_items=5)
    assert [item.id for item in ctx.items] == original_ids


def test_remove_by_id_or_item() -> None:
    ctx = _make_ctx("system", "user", "assistant")
    first, second = ctx.items[0], ctx.items[1]

    ctx.remove(first.id)
    ctx.remove(second)

    assert [item.role for item in ctx.items] == ["assistant"]  # type: ignore[union-attr]


def test_remove_nonexistent_raises():
    ctx = _make_ctx("user", "assistant")
    with pytest.raises(ValueError):
        ctx.remove("nonexistent_id")


def test_upsert_places_a_backdated_item_where_it_happened():
    """A realtime user turn is stamped when it began, which can predate what is already there."""
    ctx = ChatContext.empty()
    first = ctx.add_message(role="user", content="first", id="a", created_at=100.0)
    reply = ctx.add_message(role="assistant", content="reply", id="b", created_at=200.0)

    backdated = ChatMessage(role="user", content=["spoken over the reply"], id="c")
    backdated.created_at = 150.0
    ctx._upsert_item(backdated)

    assert [item.id for item in ctx.items] == [first.id, backdated.id, reply.id]


def test_upsert_appends_an_item_stamped_now():
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content="first", id="a", created_at=100.0)

    fresh = ChatMessage(role="assistant", content=["later"], id="b")
    ctx._upsert_item(fresh)

    assert [item.id for item in ctx.items] == ["a", "b"]


def test_upsert_still_replaces_an_item_in_place():
    """A corrected transcript re-delivers the same id, and must not become a second item."""
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content="first", id="a", created_at=100.0)
    ctx.add_message(role="assistant", content="reply", id="b", created_at=200.0)

    corrected = ChatMessage(role="user", content=["first, corrected"], id="a")
    corrected.created_at = 100.0
    ctx._upsert_item(corrected)

    assert [item.id for item in ctx.items] == ["a", "b"]
    assert ctx.items[0].text_content == "first, corrected"


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
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ('{"order": "123"}', {"order": "123"}),
        ('{"a": 1,}', {"a": 1}),
        ('{"a": <|end|>1}', {"a": "end1"}),
        ('"{\\"a\\": 1}"', {"a": 1}),
        # a stored call whose arguments never parsed must not crash a later turn
        ("not-a-json-object", {}),
        # a call's arguments are a named-parameter object; anything else means no arguments
        ("", {}),
        ("[1, 2, 3]", {}),
        ("42", {}),
        ("null", {}),
    ],
    ids=[
        "valid",
        "trailing-comma",
        "chat-template-leak",
        "double-encoded",
        "unparseable",
        "empty",
        "array",
        "scalar",
        "null",
    ],
)
def test_to_provider_format_tool_arguments(fmt: str, arguments: str, expected: dict[str, Any]):
    ctx = ChatContext.empty()
    ctx.insert(FunctionCall(call_id="c1", name="lookup", arguments=arguments))
    ctx.insert(FunctionCallOutput(call_id="c1", name="lookup", output="ok", is_error=False))

    messages, _ = ctx.to_provider_format(format=fmt)
    assert _tool_call_input(fmt, messages) == expected


def test_copy_keeps_a_tool_output_with_no_name():
    """`FunctionCallOutput.name` is optional, so a name-less output is paired by `call_id`."""
    ctx = ChatContext.empty()
    ctx.insert(ChatMessage(role="user", content=["what's the weather in Paris?"]))
    ctx.insert(FunctionCall(call_id="c1", name="get_weather", arguments='{"location":"Paris"}'))
    ctx.insert(FunctionCallOutput(call_id="c1", output="sunny, 22C", is_error=False))

    copied = ctx.copy(tools=["get_weather"])
    assert [item.type for item in copied.items] == [
        "message",
        "function_call",
        "function_call_output",
    ]


def test_copy_drops_a_tool_output_whose_call_is_dropped():
    """An output goes with its call, so the filter never leaves an orphan behind."""
    ctx = ChatContext.empty()
    ctx.insert(FunctionCall(call_id="c1", name="removed_tool", arguments="{}"))
    ctx.insert(FunctionCallOutput(call_id="c1", name="removed_tool", output="ok", is_error=False))
    ctx.insert(FunctionCallOutput(call_id="c2", name="removed_tool", output="ok", is_error=False))

    assert ctx.copy(tools=["get_weather"]).items == []


def test_copy_keeps_a_call_that_has_no_output_yet():
    """A call whose reply is still in flight survives the filter."""
    ctx = ChatContext.empty()
    ctx.insert(FunctionCall(call_id="c1", name="get_weather", arguments="{}"))

    copied = ctx.copy(tools=["get_weather"])
    assert [item.type for item in copied.items] == ["function_call"]


def test_copy_keeps_a_named_tool_output_whose_call_is_not_in_the_context():
    """A realtime provider holds the call, so the output keeps its own name as the test."""
    ctx = ChatContext.empty()
    ctx.insert(FunctionCallOutput(call_id="c1", name="get_weather", output="ok", is_error=False))

    copied = ctx.copy(tools=["get_weather"])
    assert [item.type for item in copied.items] == ["function_call_output"]


def test_copy_drops_a_name_less_tool_output_whose_call_is_not_in_the_context():
    """With no call and no name there is nothing that proves the output eligible."""
    ctx = ChatContext.empty()
    ctx.insert(FunctionCallOutput(call_id="c1", output="ok", is_error=False))

    assert ctx.copy(tools=["get_weather"]).items == []


def test_readonly_chat_ctx_blocks_every_mutation_path():
    """`Agent.chat_ctx` is a `_ReadOnlyChatContext`; every mutation path must raise."""
    ctx = ChatContext.empty()
    ctx.add_message(role="system", content="sys")

    ro = _ReadOnlyChatContext(ctx.items)  # what Agent.chat_ctx returns
    assert ro.readonly is True

    with pytest.raises(RuntimeError):
        ro.items.append(ChatMessage(role="user", content=["x"]))

    with pytest.raises(RuntimeError):
        ro.insert(ChatMessage(role="user", content=["via insert()"]))

    with pytest.raises(RuntimeError):
        ro.add_message(role="user", content="x", created_at=1.0)

    other = ChatContext.empty()
    other.add_message(role="user", content="other")
    with pytest.raises(RuntimeError):
        ro.merge(other)

    with pytest.raises(RuntimeError):
        ro.items = [ChatMessage(role="user", content=["replaced"])]

    # a blocked setter must not have swapped the immutable list for a mutable one
    with pytest.raises(RuntimeError):
        ro.items.append(ChatMessage(role="user", content=["x"]))

    with pytest.raises(RuntimeError):
        ro.items.pop()

    # nothing above may reach the context the read-only view was built from
    assert len(ro.items) == 1
    assert len(ctx.items) == 1


def _equivalence_items() -> list[Any]:
    return [
        ChatMessage(id="m", role="user", content=["hi"]),
        FunctionCall(id="c", call_id="1", name="f", arguments="{}"),
        FunctionCallOutput(id="o", call_id="1", name="f", output="ok", is_error=False),
        AgentHandoff(id="h", new_agent_id="b"),
    ]


@pytest.mark.parametrize(
    ("index", "update"),
    [
        (0, {"role": "assistant"}),
        (0, {"content": ["hello"]}),
        (0, {"interrupted": True}),
        (1, {"arguments": '{"x": 1}'}),
        (1, {"name": "g"}),
        (2, {"output": "nope"}),
        (2, {"is_error": True}),
    ],
)
def test_essential_field_change_breaks_equivalence_and_fingerprint(
    index: int, update: dict[str, Any]
) -> None:
    items = _equivalence_items()
    changed = list(items)
    changed[index] = items[index].model_copy(update=update)

    assert not ChatContext(items).is_equivalent(ChatContext(changed))
    assert items[index]._fingerprint() != changed[index]._fingerprint()


def test_metadata_does_not_affect_equivalence_or_fingerprint() -> None:
    items = _equivalence_items()
    changed = [
        items[0].model_copy(
            update={"created_at": 0.0, "transcript_confidence": 0.5, "extra": {"k": 1}}
        ),
        items[1].model_copy(update={"created_at": 0.0, "group_id": "g"}),
        items[2].model_copy(update={"created_at": 0.0, "reply_required": False}),
        # handoffs are compared by id and type only
        items[3].model_copy(update={"new_agent_id": "c"}),
    ]

    assert ChatContext(items).is_equivalent(ChatContext(changed))
    assert [i._fingerprint() for i in items] == [i._fingerprint() for i in changed]


def test_fingerprint_detects_replaced_image_without_hashing_payload() -> None:
    a = ChatMessage(role="user", content=[ImageContent(id="img", image="data:image/png;base64,AA")])
    b = a.model_copy(update={"content": [ImageContent(id="img", image="data:image/png;base64,BB")]})
    other = a.model_copy(
        update={"content": [ImageContent(id="img2", image="data:image/png;base64,AA")]}
    )
    assert a._fingerprint() != b._fingerprint()
    assert a._fingerprint() != other._fingerprint()


def test_fingerprint_tracks_image_url_and_inference_settings() -> None:
    a = ChatMessage(role="user", content=[ImageContent(id="img", image="https://a.example/img")])
    changed_url = a.model_copy(
        update={"content": [ImageContent(id="img", image="https://b.example/img")]}
    )
    changed_detail = a.model_copy(
        update={
            "content": [
                ImageContent(id="img", image="https://a.example/img", inference_detail="high")
            ]
        }
    )
    assert a._fingerprint() != changed_url._fingerprint()
    assert a._fingerprint() != changed_detail._fingerprint()


def test_compute_lcs_returns_a_longest_common_subsequence() -> None:
    def dp_lcs_length(old_ids: list[str], new_ids: list[str]) -> int:
        n, m = len(old_ids), len(new_ids)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if old_ids[i - 1] == new_ids[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[n][m]

    def is_subsequence(sub: list[str], seq: list[str]) -> bool:
        it = iter(seq)
        return all(item in it for item in sub)

    for seed in range(300):
        rng = random.Random(seed)
        pool = [f"id_{i}" for i in range(rng.randint(0, 40))]
        old_ids = rng.sample(pool, rng.randint(0, len(pool)))
        new_ids = rng.sample(pool, rng.randint(0, len(pool)))
        if seed % 3 == 0 and old_ids:
            # repeated IDs take the dynamic-programming path
            new_ids += rng.choices(old_ids, k=3)

        lcs_ids = utils._compute_lcs(old_ids, new_ids)

        assert len(lcs_ids) == dp_lcs_length(old_ids, new_ids), seed
        assert is_subsequence(lcs_ids, old_ids), seed
        assert is_subsequence(lcs_ids, new_ids), seed


def test_compute_chat_ctx_diff_on_a_long_context_does_not_build_an_n_by_m_table() -> None:
    n = 2000
    old = ChatContext()
    for i in range(n):
        old.add_message(role="user", content=f"message {i}", id=f"item_{i}")
    new = old.copy()
    new.add_message(role="system", content="appended", id="appended")

    tracemalloc.start()
    try:
        diff = utils.compute_chat_ctx_diff(old, new)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert diff.to_create == [(f"item_{n - 1}", "appended")]
    assert diff.to_remove == []
    assert diff.to_update == []
    # the O(n*m) table for 2000 x 2001 items takes about 32 MB
    assert peak < 1_000_000


def test_to_dict_encodes_bytes_in_extra_and_hash() -> None:
    """extra (provider-specific data) may hold non-UTF-8 bytes (e.g. a Google
    thought signature); to_dict() must not crash and must JSON-encode them."""
    import json

    ctx = ChatContext(
        [
            ChatMessage(
                role="assistant",
                content=["ok"],
                extra={"google": {"thought_signature": b"\x00\x01\xff"}},
                hash=b"\x00\x01",
            )
        ]
    )
    data = ctx.to_dict()
    assert data["items"][0]["extra"]["google"]["thought_signature"] == base64.b64encode(
        b"\x00\x01\xff"
    ).decode("ascii")
    assert data["items"][0]["hash"] == base64.b64encode(b"\x00\x01").decode("ascii")
    # The whole dict must be JSON-serializable (no bytes remain).
    json.dumps(data)


def test_to_dict_encodes_bytes_in_function_call_extra() -> None:
    """FunctionCall.extra bytes must also be base64-encoded, not crash."""
    import json

    ctx = ChatContext(
        [FunctionCall(call_id="c1", arguments="{}", name="tool", extra={"sig": b"\xff\xfe"})]
    )
    data = ctx.to_dict()
    assert data["items"][0]["extra"]["sig"] == base64.b64encode(b"\xff\xfe").decode("ascii")
    json.dumps(data)
