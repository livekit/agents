from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from livekit.agents import llm

from .utils import (
    convert_mid_conversation_instructions,
    group_tool_calls,
    parse_tool_call_arguments,
)

_AWS_IMAGE_FORMATS = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}


@dataclass
class BedrockFormatData:
    system_messages: list[str] | None


def to_chat_ctx(
    chat_ctx: llm.ChatContext,
    *,
    inject_dummy_user_message: bool = True,
    cache_messages: bool = False,
) -> tuple[list[dict], BedrockFormatData]:
    """Convert to Bedrock Converse ``messages`` and system texts.

    With ``cache_messages``, a ``cachePoint`` block follows the last block of conversation
    content in the serialized messages: the latest user message, tool call or tool result
    that produced content. Tool results are serialized next to their calls, so this is
    not always the last item of ``chat_ctx``. System and developer messages after the
    preamble, such as ``generate_reply(instructions=...)``, are rewritten into user turns
    for this request only and never end the cached prefix: a cachePoint after their text
    would end it on content the next request no longer sends, so it would never be read.
    """
    # ids survive convert_mid_conversation_instructions, so the rewritten instructions
    # can still be told apart from the conversation
    instruction_ids = {
        item.id
        for item in chat_ctx.items
        if item.type == "message" and item.role in ("system", "developer")
    }
    chat_ctx = convert_mid_conversation_instructions(chat_ctx)

    messages: list[dict] = []
    system_messages: list[str] = []
    current_role: str | None = None
    current_content: list[dict] = []
    # (content list, index) right after the last conversation block serialized so far
    cache_at: tuple[list[dict], int] | None = None

    for msg in itertools.chain(*(group.flatten() for group in group_tool_calls(chat_ctx))):
        if msg.type == "message" and msg.role == "system" and (text := msg.raw_text_content):
            system_messages.append(text)
            continue

        if msg.type == "message":
            role = "assistant" if msg.role == "assistant" else "user"
        elif msg.type == "function_call":
            role = "assistant"
        elif msg.type == "function_call_output":
            role = "user"

        # if the effective role changed, finalize the previous turn.
        if role != current_role:
            if current_content and current_role is not None:
                messages.append({"role": current_role, "content": current_content})
            current_content = []
            current_role = role

        blocks_before = len(current_content)
        if msg.type == "message":
            for content in msg.content:
                if isinstance(content, llm.ImageContent):
                    current_content.append(_build_image(content))
                elif isinstance(content, (llm.AudioContent, llm.CacheBreakpoint)):
                    pass
                elif content:
                    # str or Instructions
                    current_content.append({"text": str(content)})
        elif msg.type == "function_call":
            current_content.append(
                {
                    "toolUse": {
                        "toolUseId": msg.call_id,
                        "name": msg.name,
                        "input": parse_tool_call_arguments(msg),
                    }
                }
            )
        elif msg.type == "function_call_output":
            current_content.append(
                {
                    "toolResult": {
                        "toolUseId": msg.call_id,
                        "content": [
                            {"json": msg.output}
                            if isinstance(msg.output, dict)
                            else {"text": msg.output}
                        ],
                        "status": "error" if msg.is_error else "success",
                    }
                }
            )

        # an item that added no block (e.g. audio only) leaves the boundary where it was
        if (
            cache_messages
            and msg.id not in instruction_ids
            and len(current_content) > blocks_before
        ):
            cache_at = (current_content, len(current_content))

    # Finalize the last message if there’s any content left
    if current_role is not None and current_content:
        messages.append({"role": current_role, "content": current_content})

    # the content list is the one stored in `messages`, so this edits the request
    if cache_at is not None:
        cache_content, cache_index = cache_at
        cache_content.insert(cache_index, {"cachePoint": {"type": "default"}})

    # Ensure the message list starts with a "user" message
    if inject_dummy_user_message and (not messages or messages[0]["role"] != "user"):
        messages.insert(0, {"role": "user", "content": [{"text": "(empty)"}]})

    return messages, BedrockFormatData(system_messages=system_messages)


def _build_image(image: llm.ImageContent) -> dict:
    cache_key = "serialized_image"
    if cache_key not in image._cache:
        image._cache[cache_key] = llm.utils.serialize_image(image)
    img: llm.utils.SerializedImage = image._cache[cache_key]

    if img.external_url:
        raise ValueError("external_url is not supported by AWS Bedrock.")

    assert img.data_bytes is not None
    return {
        "image": {
            "format": _image_format(img.mime_type),
            "source": {"bytes": img.data_bytes},
        }
    }


def _image_format(mime_type: str | None) -> str:
    if not mime_type:
        return "jpeg"

    try:
        return _AWS_IMAGE_FORMATS[mime_type]
    except KeyError:
        raise ValueError(f"Unsupported mime_type {mime_type} for AWS Bedrock images") from None


def to_fnc_ctx(tool_ctx: llm.ToolContext) -> list[dict[str, Any]]:
    return [_build_tool_spec(tool) for tool in tool_ctx.function_tools.values()]


def _build_tool_spec(tool: llm.FunctionTool | llm.RawFunctionTool) -> dict:
    if isinstance(tool, llm.FunctionTool):
        fnc = llm.utils.build_legacy_openai_schema(tool, internally_tagged=True)
        return {
            "toolSpec": _strip_nones(
                {
                    "name": fnc["name"],
                    "description": fnc["description"] if fnc["description"] else None,
                    "inputSchema": {"json": fnc["parameters"] if fnc["parameters"] else {}},
                }
            )
        }
    elif isinstance(tool, llm.RawFunctionTool):
        info = tool.info
        return {
            "toolSpec": _strip_nones(
                {
                    "name": info.name,
                    "description": info.raw_schema.get("description", ""),
                    "inputSchema": {"json": info.raw_schema.get("parameters", {})},
                }
            )
        }
    else:
        raise ValueError("Invalid function tool")


def _strip_nones(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}
