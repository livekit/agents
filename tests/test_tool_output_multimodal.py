from google.genai import types as gemini_types

from livekit.agents.llm import ChatContext, FunctionCall, FunctionCallOutput, ImageContent, utils
from livekit.plugins.google.utils import get_tool_results_for_realtime
from livekit.plugins.openai.realtime.utils import livekit_item_to_openai_item


def _fnc_call(call_id: str = "call_1") -> FunctionCall:
    return FunctionCall(name="capture", call_id=call_id, arguments="{}")


def test_make_function_call_output_supports_image() -> None:
    image = ImageContent(image="https://example.com/image.jpg")
    result = utils.make_function_call_output(
        fnc_call=_fnc_call(),
        output=image,
        exception=None,
    )

    assert result.fnc_call_out is not None
    assert isinstance(result.fnc_call_out.output, ImageContent)
    assert result.fnc_call_out.output.image == "https://example.com/image.jpg"


def test_make_function_call_output_supports_text_and_image_list() -> None:
    image = ImageContent(image="https://example.com/image.jpg")
    result = utils.make_function_call_output(
        fnc_call=_fnc_call(),
        output=["before", image, "after"],
        exception=None,
    )

    assert result.fnc_call_out is not None
    assert result.fnc_call_out.output == ["before", image, "after"]


def test_make_function_call_output_keeps_legacy_types_as_text() -> None:
    result = utils.make_function_call_output(
        fnc_call=_fnc_call(),
        output={"k": "v"},
        exception=None,
    )

    assert result.fnc_call_out is not None
    assert result.fnc_call_out.output == "{'k': 'v'}"


def test_tool_output_to_text_uses_placeholder_for_images() -> None:
    image = ImageContent(image="https://example.com/image.jpg")
    text = utils.tool_output_to_text(["before", image, "after"])
    assert text == "before\n[Image omitted: unsupported by provider]\nafter"


def test_tool_output_to_text_for_pure_image_uses_placeholder() -> None:
    image = ImageContent(image="https://example.com/image.jpg")
    text = utils.tool_output_to_text(image)
    assert text == "[Image omitted: unsupported by provider]"


def test_normalize_function_output_value_preserves_image_in_mixed_list() -> None:
    image = ImageContent(image="https://example.com/image.jpg")
    normalized = utils.normalize_function_output_value([image, 42, {"k": "v"}, "tail"])
    assert normalized == [image, "42", "{'k': 'v'}", "tail"]


def test_chat_context_to_dict_excludes_images_in_tool_output() -> None:
    image = ImageContent(image="https://example.com/image.jpg")
    chat_ctx = ChatContext(
        items=[
            _fnc_call(),
            FunctionCallOutput(
                name="capture",
                call_id="call_1",
                output=["before", image, "after"],
                is_error=False,
            ),
        ]
    )

    data = chat_ctx.to_dict(exclude_image=True)
    output = data["items"][1]["output"]
    assert output == ["before", "after"]


def test_chat_context_to_dict_excludes_images_to_empty_string() -> None:
    image = ImageContent(image="https://example.com/image.jpg")
    chat_ctx = ChatContext(
        items=[
            _fnc_call(),
            FunctionCallOutput(
                name="capture",
                call_id="call_1",
                output=[image],
                is_error=False,
            ),
        ]
    )

    data = chat_ctx.to_dict(exclude_image=True)
    output = data["items"][1]["output"]
    assert output == ""


def test_provider_formats_tool_output_with_images() -> None:
    image = ImageContent(image="data:image/png;base64,aW1n")
    chat_ctx = ChatContext(
        items=[
            _fnc_call(),
            FunctionCallOutput(
                name="capture",
                call_id="call_1",
                output=["caption", image],
                is_error=False,
            ),
        ]
    )

    openai_messages, _ = chat_ctx.to_provider_format("openai", inject_dummy_user_message=False)
    assert openai_messages[1]["role"] == "tool"
    assert openai_messages[1]["content"] == "caption\n[Image omitted: unsupported by provider]"

    openai_responses, _ = chat_ctx.to_provider_format(
        "openai.responses", inject_dummy_user_message=False
    )
    output = openai_responses[1]["output"]
    assert isinstance(output, list)
    assert output[0] == {"type": "input_text", "text": "caption"}
    assert output[1]["type"] == "input_image"

    anthropic_messages, _ = chat_ctx.to_provider_format("anthropic", inject_dummy_user_message=False)
    tool_result = next(
        block
        for message in anthropic_messages
        for block in message["content"]
        if block["type"] == "tool_result"
    )
    assert isinstance(tool_result["content"], list)
    assert tool_result["content"][0] == {"type": "text", "text": "caption"}
    assert tool_result["content"][1]["type"] == "image"

    google_turns, _ = chat_ctx.to_provider_format("google", inject_dummy_user_message=False)
    function_response = next(
        part["function_response"]
        for turn in google_turns
        for part in turn["parts"]
        if "function_response" in part
    )
    assert function_response["response"]["output"] == "caption"
    assert len(function_response["parts"]) == 1
    for turn in google_turns:
        gemini_types.Content.model_validate(turn)

    aws_messages, _ = chat_ctx.to_provider_format("aws", inject_dummy_user_message=False)
    tool_result_content = aws_messages[1]["content"][0]["toolResult"]["content"]
    assert tool_result_content[0] == {"text": "caption"}
    assert tool_result_content[1]["image"]["format"] == "png"


def test_openai_chat_supports_multimodal_tool_output_when_enabled() -> None:
    image = ImageContent(image="data:image/png;base64,aW1n")
    chat_ctx = ChatContext(
        items=[
            _fnc_call(),
            FunctionCallOutput(
                name="capture",
                call_id="call_1",
                output=["caption", image],
                is_error=False,
            ),
        ]
    )

    openai_messages, _ = chat_ctx.to_provider_format(
        "openai",
        inject_dummy_user_message=False,
        supports_tool_image_output=True,
    )
    assert openai_messages[1]["role"] == "tool"
    content = openai_messages[1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "caption"}
    assert content[1]["type"] == "image_url"


def test_openai_chat_keeps_text_tool_output_as_string_when_enabled() -> None:
    chat_ctx = ChatContext(
        items=[
            _fnc_call(),
            FunctionCallOutput(
                name="capture",
                call_id="call_1",
                output="done",
                is_error=False,
            ),
        ]
    )

    openai_messages, _ = chat_ctx.to_provider_format(
        "openai",
        inject_dummy_user_message=False,
        supports_tool_image_output=True,
    )
    assert openai_messages[1]["content"] == "done"


def test_google_provider_pure_image_tool_output_omits_empty_output_key() -> None:
    image = ImageContent(image="data:image/png;base64,aW1n")
    chat_ctx = ChatContext(
        items=[
            _fnc_call(),
            FunctionCallOutput(
                name="capture",
                call_id="call_1",
                output=image,
                is_error=False,
            ),
        ]
    )

    google_turns, _ = chat_ctx.to_provider_format("google", inject_dummy_user_message=False)
    function_response = next(
        part["function_response"]
        for turn in google_turns
        for part in turn["parts"]
        if "function_response" in part
    )
    assert function_response["response"] == {}
    assert len(function_response["parts"]) == 1
    for turn in google_turns:
        gemini_types.Content.model_validate(turn)


def test_aws_provider_falls_back_for_external_url_tool_image() -> None:
    image = ImageContent(image="https://example.com/image.jpg")
    chat_ctx = ChatContext(
        items=[
            _fnc_call(),
            FunctionCallOutput(
                name="capture",
                call_id="call_1",
                output=["caption", image],
                is_error=False,
            ),
        ]
    )

    aws_messages, _ = chat_ctx.to_provider_format("aws", inject_dummy_user_message=False)
    tool_result_content = aws_messages[1]["content"][0]["toolResult"]["content"]
    assert tool_result_content[0] == {"text": "caption"}
    assert tool_result_content[1] == {"text": "[Image omitted: unsupported by provider]"}


def test_openai_realtime_falls_back_to_text_tool_output() -> None:
    image = ImageContent(image="https://example.com/image.jpg")
    item = FunctionCallOutput(
        name="capture",
        call_id="call_1",
        output=["caption", image],
        is_error=False,
    )

    converted = livekit_item_to_openai_item(item)
    assert converted.output == "caption\n[Image omitted: unsupported by provider]"


def test_google_realtime_falls_back_to_text_tool_output() -> None:
    image = ImageContent(image="https://example.com/image.jpg")
    chat_ctx = ChatContext(
        items=[
            FunctionCallOutput(
                name="capture",
                call_id="call_1",
                output=["caption", image],
                is_error=False,
            )
        ]
    )

    tool_results = get_tool_results_for_realtime(chat_ctx)
    assert tool_results is not None
    assert tool_results.function_responses is not None
    assert tool_results.function_responses[0].response == {
        "output": "caption\n[Image omitted: unsupported by provider]"
    }
