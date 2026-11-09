"""Provider wires and byte materialization against the Cargo-linked module."""

import pytest

from livekit.agents import _core
from livekit.agents._generated.provider_data import ProviderFormat, ToolSchemaRaw
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
)
from livekit.agents.llm.utils import SerializedImage

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("provider", ["google", "aws"])
async def test_provider_image_bytes_are_materialized_for_the_sdk(provider: ProviderFormat) -> None:
    image = ImageContent(id="image", image="https://example.test/image.png")
    formatted = await _core.to_provider_format(
        provider,
        [ChatMessage(role="user", content=[image])],
        images={
            "image": SerializedImage(
                data_bytes=b"\x00\xffimage", mime_type="image/png", inference_detail="auto"
            )
        },
    )
    messages = formatted.messages
    if provider == "google":
        from google.genai import types

        accepted = types.Content.model_validate(messages[0])
        assert accepted.parts[0].inline_data.data == b"\x00\xffimage"
    else:
        assert messages[0]["content"][0]["image"]["source"]["bytes"] == b"\x00\xffimage"


async def test_gemini_signatures_are_bytes_and_results_do_not_mix_with_user_text() -> None:
    from google.genai import types

    context = ChatContext.empty()
    context.insert(FunctionCall(call_id="call", name="lookup", arguments="{}"))
    context.insert(
        FunctionCallOutput(call_id="call", name="lookup", output="found", is_error=False)
    )
    context.add_message(role="user", content="Thank you.")
    messages, _ = await context.to_provider_format(
        "google", thought_signatures={"call": b"\x00\xffsignature"}
    )
    accepted = [types.Content.model_validate(message) for message in messages]
    assert accepted[0].parts[0].thought_signature == b"\x00\xffsignature"
    assert [message.role for message in accepted] == ["model", "user", "user"]
    assert accepted[1].parts[0].function_response.response == {"output": "found"}
    assert accepted[2].parts[0].text == "Thank you."


async def test_openai_encrypted_reasoning_survives_saved_history() -> None:
    context = ChatContext.empty()
    reasoning = {"reasoning": [{"type": "reasoning", "encrypted_content": "ciphertext"}]}
    context.insert(ChatMessage(role="assistant", content=["answer"], extra={"openai": reasoning}))
    restored = ChatContext.from_dict(context.to_dict())
    messages, _ = await restored.to_provider_format("openai")
    assert messages[0]["extra_content"]["openai"] == reasoning


async def test_bedrock_failed_tool_status_and_raw_tool_validate_with_botocore() -> None:
    from botocore.session import Session
    from botocore.validate import validate_parameters

    context = ChatContext.empty()
    context.insert(FunctionCall(call_id="call", name="lookup", arguments="{}"))
    context.insert(
        FunctionCallOutput(call_id="call", name="lookup", output="denied", is_error=True)
    )
    messages, _ = await context.to_provider_format("aws")
    tools = _core.to_fnc_ctx("aws", [ToolSchemaRaw(name="lookup", raw_schema={})])
    service = Session().get_service_model("bedrock-runtime")
    validate_parameters(
        {"modelId": "test", "messages": messages, "toolConfig": {"tools": tools}},
        service.operation_model("ConverseStream").input_shape,
    )
    assert messages[-1]["content"][0]["toolResult"]["status"] == "error"
    assert "description" not in tools[0]["toolSpec"]
