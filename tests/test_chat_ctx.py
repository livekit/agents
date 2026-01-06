import os

import pytest

from livekit.agents.llm import AgentHandoff, FunctionCall, FunctionCallOutput, utils
from livekit.plugins import openai


def ai_function1(a: int, b: str = "default") -> None:
    """
    This is a test function
    Args:
        a: First argument
        b: Second argument
    """
    pass


def skip_if_no_credentials():
    required_vars = ["OPENAI_API_KEY"]
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
    from livekit.agents.llm import ChatContext, ImageContent

    chat_ctx = ChatContext()
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

    async with openai.LLM(model="gpt-4o") as llm:
        summary = await chat_ctx._summarize(llm, keep_last_turns=1)
        print("\n=== Summary ===\n")
        print(json.dumps(summary.to_dict(), indent=2))
