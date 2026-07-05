import base64
import os
from typing import Any

import pytest

from livekit.agents import inference
from livekit.agents.llm import AgentHandoff, ChatContext, FunctionCall, FunctionCallOutput, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]

_IMAGE_BYTES = b"fake image bytes"


def ai_function1(a: int, b: str = "default") -> None:
    """
    This is a test function
    Args:
        a: First argument
        b: Second argument
    """
    pass


def skip_if_no_credentials():
    required_vars = ["LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
    missing = [var for var in required_vars if not os.getenv(var)]
    return pytest.mark.skipif(
        bool(missing), reason=f"Missing environment variables: {', '.join(missing)}"
    )


def test_args_model():
    from docstring_parser import parse_from_object

    docstring = parse_from_object(ai_function1)
    print(docstring.description)

    model = utils.function_arguments_to_pydantic_model(ai_function1)
    print(model.model_json_schema())


def test_dict():
    from livekit import rtc
    from livekit.agents.beta import Instructions
    from livekit.agents.llm import ChatContext, ImageContent

    chat_ctx = ChatContext()
    chat_ctx.add_message(
        role="system",
        content=Instructions(
            "You are a helpful assistant in audio mode.",
            text="You are a helpful assistant in text mode.",
        ),
    )
    chat_ctx.add_message(
        role="user",
        content="Hello, world!",
    )
    chat_ctx.add_message(
        role="assistant",
        content="Hello, world!",
    )
    chat_ctx.add_message(
        role="user",
        content=[
            ImageContent(
                image=rtc.VideoFrame(64, 64, rtc.VideoBufferType.RGB24, b"0" * 64 * 64 * 3)
            )
        ],
    )
    print(chat_ctx.to_dict())
    print(chat_ctx.items)
    print(ChatContext.from_dict(chat_ctx.to_dict()).items)


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
    from livekit.agents.llm import ChatContext, ImageContent

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
    from livekit.agents.llm import ChatContext, ImageContent

    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(
        role="user",
        content=[ImageContent(image="https://example.com/image.png", mime_type="image/png")],
    )

    with pytest.raises(ValueError, match="external_url is not supported by AWS Bedrock"):
        chat_ctx.to_provider_format(format="aws")


def test_chat_ctx_can_be_serialized_and_deserialized_with_defaults():
    from livekit.agents.llm import AgentHandoff, ChatContext, ChatMessage

    items = [
        AgentHandoff(new_agent_id="default_agent", old_agent_id=None),
        ChatMessage(role="user", content=["Hello, world!"]),
        ChatMessage(role="assistant", content=["Hi there!"]),
    ]
    chat_ctx = ChatContext(items)
    assert chat_ctx.is_equivalent(ChatContext.from_dict(chat_ctx.to_dict()))


@skip_if_no_credentials()
async def test_summarize():
    from livekit.agents import ChatContext

    chat_ctx = ChatContext()
    chat_ctx.add_message(
        role="system",
        content=(
            "You are SupportGPT, a customer service agent for Acme Audio. "
            "Gather identifying info first, then troubleshoot. "
            "Only promise replacements if the device is under warranty. "
            "Use the provided tools for order lookup, warranty checks, and RMA creation. "
            "If a return is required, hand off to ReturnsAgent for shipping label logistics."
        ),
    )
    chat_ctx.add_message(
        role="user",
        content=(
            "Hi, I need help with an order I placed last week. The earbuds I got "
            "keep disconnecting and the left side sounds crackly."
        ),
    )
    chat_ctx.add_message(
        role="assistant",
        content=(
            "I can help with that! First, could you share your full name and the email "
            "you used at checkout, so I can locate your order?"
        ),
    )
    chat_ctx.add_message(
        role="user", content=("Sure—I'm Maya Chen, and I used maya.chen+shop@gmail.com.")
    )
    chat_ctx.add_message(
        role="assistant",
        content=("Thanks, Maya. Do you also have the order number and approximate purchase date?"),
    )
    chat_ctx.add_message(role="user", content=("Order #LK-4821936. I bought them on October 7."))

    chat_ctx.items.append(
        FunctionCall(
            name="lookup_order",
            call_id="call_lookup_order_1",
            arguments='{"order_number": "LK-4821936", "customer_email": "maya.chen+shop@gmail.com"}',
        )
    )
    chat_ctx.items.append(
        FunctionCallOutput(
            name="lookup_order",
            call_id="call_lookup_order_1",
            output=(
                '{"order_number":"LK-4821936","customer_name":"Maya Chen","'
                'items":[{"sku":"AC-EBD-PRO","name":"Acme Buds Pro","qty":1}],'
                '"purchase_date":"2025-10-07","status":"delivered","serial":"ACB-PRO-7F29D4"}'
            ),
            is_error=False,
        )
    )
    chat_ctx.add_message(
        role="assistant",
        content=(
            "I found your order LK-4821936 for Acme Buds Pro, delivered October 8. "
            "To check warranty and next steps, which device are you pairing with and what OS version?"
        ),
    )
    chat_ctx.add_message(role="user", content="iPhone 14 Pro, iOS 18.0.1.")
    chat_ctx.add_message(
        role="assistant",
        content=(
            "Thanks. Have you tried any troubleshooting—resetting the buds, forgetting/re-pairing Bluetooth, "
            "or testing another device?"
        ),
    )
    chat_ctx.add_message(
        role="user",
        content=(
            "I tried forgetting and re-pairing twice. I also tested on my iPad and the left ear still crackles."
        ),
    )
    chat_ctx.add_message(
        role="assistant",
        content=("Understood. Any visible damage or signs of moisture? And when did it start?"),
    )
    chat_ctx.add_message(
        role="user",
        content=("No damage or moisture. It started the day after I received them—October 9."),
    )

    chat_ctx.items.append(
        FunctionCall(
            name="check_warranty",
            call_id="call_check_warranty_1",
            arguments='{"serial":"ACB-PRO-7F29D4","purchase_date":"2025-10-07"}',
        )
    )
    chat_ctx.items.append(
        FunctionCallOutput(
            name="check_warranty",
            call_id="call_check_warranty_1",
            output='{"eligible":true,"warranty_expires":"2026-10-07"}',
            is_error=False,
        )
    )

    chat_ctx.add_message(
        role="assistant",
        content=(
            "This appears to be a hardware defect and you’re under warranty until 2026-10-07. "
            "I can set up a free replacement. Could you confirm your shipping address and a contact number?"
        ),
    )
    chat_ctx.add_message(
        role="user",
        content=("Ship to 2150 Grove St, Apt 4B, Oakland, CA 94612. Phone is (510) 555-0136."),
    )

    chat_ctx.items.append(
        FunctionCall(
            name="create_rma",
            call_id="call_create_rma_1",
            arguments=(
                '{"order_number":"LK-4821936","serial":"ACB-PRO-7F29D4","reason":"left bud crackling / disconnects",'
                '"customer":{"name":"Maya Chen","email":"maya.chen+shop@gmail.com","phone":"(510) 555-0136","'
                'address":"2150 Grove St, Apt 4B, Oakland, CA 94612"}}'
            ),
        )
    )
    chat_ctx.items.append(
        FunctionCallOutput(
            name="create_rma",
            call_id="call_create_rma_1",
            output='{"rma_id":"RMA-90721","replacement_eta_days":2}',
            is_error=False,
        )
    )

    chat_ctx.items.append(AgentHandoff(old_agent_id="SupportGPT", new_agent_id="ReturnsAgent"))

    chat_ctx.items.append(
        FunctionCall(
            name="generate_return_label",
            call_id="call_label_1",
            arguments='{"rma_id":"RMA-90721","email":"maya.chen+shop@gmail.com"}',
        )
    )
    chat_ctx.items.append(
        FunctionCallOutput(
            name="generate_return_label",
            call_id="call_label_1",
            output='{"label_url":"https://example.invalid/label/RMA-90721","due_in_days":14}',
            is_error=False,
        )
    )

    chat_ctx.add_message(
        role="assistant",
        content=(
            "All set! I’ve created RMA #RMA-90721 linked to order LK-4821936. "
            "You’ll receive the prepaid return label and instructions at maya.chen+shop@gmail.com. "
            "Please ship the defective pair within 14 days; your replacement will ship within 48 hours."
        ),
    )

    import json

    async with inference.LLM(model="openai/gpt-4.1-mini") as llm:
        summary = await chat_ctx._summarize(llm, keep_last_turns=1)
        print("\n=== Summary ===\n")
        print(json.dumps(summary.to_dict(), indent=2))


# --- summarize unit tests (no credentials required) ---


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
    from livekit.agents.llm import ChatContext

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
    from livekit.agents.llm import ChatContext

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
    from livekit.agents.llm import ChatContext

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
    from livekit.agents.llm import ChatContext

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


# --- truncate tests ---


def _make_ctx(*roles: str):
    """Build a ChatContext with messages of the given roles."""
    from livekit.agents.llm import ChatContext

    ctx = ChatContext()
    for role in roles:
        if role == "function_call":
            ctx.items.append(FunctionCall(name="fn", call_id="c1", arguments="{}"))
        elif role == "function_call_output":
            ctx.items.append(
                FunctionCallOutput(name="fn", call_id="c1", output="{}", is_error=False)
            )
        else:
            ctx.add_message(role=role, content=f"msg-{role}")
    return ctx


def test_truncate_noop_when_under_limit():
    ctx = _make_ctx("system", "user", "assistant")
    original_ids = [item.id for item in ctx.items]
    ctx.truncate(max_items=5)
    assert [item.id for item in ctx.items] == original_ids


def test_truncate_basic():
    ctx = _make_ctx("user", "assistant", "user", "assistant")
    ctx.truncate(max_items=2)
    assert len(ctx.items) == 2
    assert ctx.items[0].role == "user"
    assert ctx.items[1].role == "assistant"


def test_truncate_preserves_system_instruction():
    ctx = _make_ctx("system", "user", "assistant", "user", "assistant")
    ctx.truncate(max_items=2)
    # system should be re-inserted at the front
    assert ctx.items[0].role == "system"
    assert len(ctx.items) == 3  # system + last 2


def test_truncate_preserves_developer_instruction():
    ctx = _make_ctx("developer", "user", "assistant", "user", "assistant")
    ctx.truncate(max_items=2)
    assert ctx.items[0].role == "developer"
    assert len(ctx.items) == 3


def test_truncate_no_duplication():
    """When the instruction is already in the truncated tail, don't insert it again."""
    ctx = _make_ctx("system", "user", "assistant")
    ctx.truncate(max_items=3)
    # system is already within the last 3 items, so no duplication
    system_items = [item for item in ctx.items if getattr(item, "role", None) == "system"]
    assert len(system_items) == 1
    assert len(ctx.items) <= 3


def test_truncate_multiple_instructions():
    """Only the first instruction by position is preserved."""
    from livekit.agents.llm import ChatContext

    ctx = ChatContext()
    ctx.add_message(role="system", content="first")
    ctx.add_message(role="developer", content="second")
    ctx.add_message(role="user", content="u1")
    ctx.add_message(role="user", content="u2")
    ctx.add_message(role="user", content="u3")

    ctx.truncate(max_items=2)
    # first instruction is the system msg
    assert ctx.items[0].role == "system"
    assert ctx.items[0].content == ["first"]


# --- remove tests ---


def test_remove_by_id():
    ctx = _make_ctx("system", "user", "assistant")
    target = ctx.items[1]
    ctx.remove(target.id)
    assert ctx.get_by_id(target.id) is None


def test_remove_by_item():
    ctx = _make_ctx("user", "assistant", "user")
    target = ctx.items[0]
    ctx.remove(target)
    assert len(ctx.items) == 2


def test_remove_nonexistent_raises():
    ctx = _make_ctx("user", "assistant")
    with pytest.raises(ValueError):
        ctx.remove("nonexistent_id")


def test_instructions_serialization():
    """Instructions is resolved to str before storage in ChatMessage content."""
    from livekit.agents.llm import ChatContext, ChatMessage
    from livekit.agents.llm.chat_context import Instructions

    # Instructions is no longer a valid ChatContent type; it must be resolved to str first
    instr = Instructions("common text", audio="audio addition", text="text addition")

    # str(instr) returns the common text
    assert str(instr) == "common text"

    # render resolves to a plain string
    assert instr.render(modality="audio") == "common text\n\naudio addition"
    assert instr.render(modality="text") == "common text\n\ntext addition"

    # ChatMessage content must be str (Instructions is resolved before storage)
    resolved = instr.render(modality="audio")
    msg = ChatMessage(role="system", content=[resolved])
    assert isinstance(msg.content[0], str)
    assert msg.content[0] == "common text\n\naudio addition"

    # add_message with Instructions resolves to str(instr) = common
    ctx = ChatContext()
    ctx.add_message(role="system", content=instr)
    assert isinstance(ctx.items[0].content[0], str)
    assert ctx.items[0].content[0] == "common text"

    # to_dict / from_dict round-trip with resolved str content
    ctx2 = ChatContext([ChatMessage(role="system", content=[resolved])])
    data = ctx2.to_dict()
    serialized = data["items"][0]["content"][0]
    assert isinstance(serialized, str)
    assert serialized == "common text\n\naudio addition"

    restored = ChatContext.from_dict(ctx2.to_dict())
    restored_content = restored.items[0].content[0]
    assert isinstance(restored_content, str)
    assert restored_content == "common text\n\naudio addition"

    # Plain str content stays as str after round-trip
    plain_ctx = ChatContext([ChatMessage(role="user", content=["hello"])])
    plain_restored = ChatContext.from_dict(plain_ctx.to_dict())
    assert type(plain_restored.items[0].content[0]) is str


def test_instructions_render():
    """render() returns a plain str combining common + modality-specific additions."""
    from livekit.agents.llm.chat_context import Instructions

    instr = Instructions(
        "You are a helpful assistant.",
        audio="Keep responses short for voice.",
        text="Use markdown formatting.",
    )

    # render('audio') returns common + audio addition
    resolved_audio = instr.render(modality="audio")
    assert isinstance(resolved_audio, str)
    assert resolved_audio == "You are a helpful assistant.\n\nKeep responses short for voice."

    # render('text') returns common + text addition
    resolved_text = instr.render(modality="text")
    assert isinstance(resolved_text, str)
    assert resolved_text == "You are a helpful assistant.\n\nUse markdown formatting."

    # str(instr) returns just the common text
    assert str(instr) == "You are a helpful assistant."

    # Instructions without modality additions returns just common for both
    common_only = Instructions("common only")
    assert common_only.render(modality="audio") == "common only"
    assert common_only.render(modality="text") == "common only"

    # Instructions with only one modality addition
    audio_only = Instructions("base", audio="audio extra")
    assert audio_only.render(modality="audio") == "base\n\naudio extra"
    assert audio_only.render(modality="text") == "base"

    text_only = Instructions("base", text="text extra")
    assert text_only.render(modality="audio") == "base"
    assert text_only.render(modality="text") == "base\n\ntext extra"

    # Empty common with additions
    empty_common = Instructions("", audio="audio only", text="text only")
    assert empty_common.render(modality="audio") == "audio only"
    assert empty_common.render(modality="text") == "text only"


def test_resolve_template_no_double_render():
    """resolve_template with Instructions kwargs renders each variant exactly once."""
    from livekit.agents.llm.chat_context import Instructions

    instr = Instructions.resolve_template(
        "{persona}\n\n{modality_specific}",
        persona="You are a helpful assistant.",
        modality_specific=Instructions(
            audio="Handle noisy voice input.", text="Handle typed input."
        ),
    )

    # each modality resolves to a single copy of the template, not two
    assert instr.render(modality="audio") == (
        "You are a helpful assistant.\n\nHandle noisy voice input."
    )
    assert instr.render(modality="text") == ("You are a helpful assistant.\n\nHandle typed input.")
    # persona (and everything else) appears exactly once per modality
    assert instr.render(modality="audio").count("You are a helpful assistant.") == 1


def test_resolve_template_plain_kwargs():
    """resolve_template without Instructions kwargs is a plain common-only render."""
    from livekit.agents.llm.chat_context import Instructions

    instr = Instructions.resolve_template("Hello {name}", name="Alex")
    assert instr.common == "Hello Alex"
    assert instr.audio is None
    assert instr.text is None
    assert instr.render() == "Hello Alex"
    assert instr.render(modality="audio") == "Hello Alex"


def test_resolve_template_identical_variants_collapse():
    """When modality variants are identical, resolve_template collapses to common-only."""
    from livekit.agents.llm.chat_context import Instructions

    # an Instructions kwarg with no modality-specific parts yields identical variants
    instr = Instructions.resolve_template(
        "{persona}\n\n{note}",
        persona="You are a helpful assistant.",
        note=Instructions("shared note"),
    )
    assert instr.audio is None
    assert instr.text is None
    assert instr.render() == "You are a helpful assistant.\n\nshared note"
    assert instr.render(modality="audio") == "You are a helpful assistant.\n\nshared note"


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
def test_to_provider_format_tolerates_unparseable_tool_arguments(fmt: str):
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

    messages, _ = ctx.to_provider_format(format=fmt)
    assert _tool_call_input(fmt, messages) == {}


@pytest.mark.parametrize("fmt", _JSON_OBJECT_FORMATS)
def test_to_provider_format_preserves_valid_tool_arguments(fmt: str):
    """Valid JSON-object arguments are still passed through unchanged."""
    ctx = ChatContext.empty()
    ctx.insert(FunctionCall(call_id="c1", name="lookup", arguments='{"order": "123"}'))
    ctx.insert(FunctionCallOutput(call_id="c1", name="lookup", output="ok", is_error=False))

    messages, _ = ctx.to_provider_format(format=fmt)
    assert _tool_call_input(fmt, messages) == {"order": "123"}


@pytest.mark.parametrize("fmt", _JSON_OBJECT_FORMATS)
@pytest.mark.parametrize("arguments", ["", "[1, 2, 3]", "42", "null"])
def test_to_provider_format_non_object_tool_arguments(fmt: str, arguments: str):
    """Empty or non-object arguments degrade to `{}`; a call's arguments are always a
    named-parameter object, so an array/scalar/null is treated as no arguments."""
    ctx = ChatContext.empty()
    ctx.insert(FunctionCall(call_id="c1", name="lookup", arguments=arguments))
    ctx.insert(FunctionCallOutput(call_id="c1", name="lookup", output="ok", is_error=False))

    messages, _ = ctx.to_provider_format(format=fmt)
    assert _tool_call_input(fmt, messages) == {}
