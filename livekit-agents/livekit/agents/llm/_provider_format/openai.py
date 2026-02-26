from __future__ import annotations

import base64
from typing import Any, Literal

from livekit.agents import llm

from .utils import group_tool_calls


def to_chat_ctx(
    chat_ctx: llm.ChatContext, *, inject_dummy_user_message: bool = True
) -> tuple[list[dict], Literal[None]]:
    item_groups = group_tool_calls(chat_ctx)
    messages = []
    for group in item_groups:
        if not group.message and not group.tool_calls and not group.tool_outputs:
            continue

        # one message can contain zero or more tool calls
        msg = _to_chat_item(group.message) if group.message else {"role": "assistant"}
        tool_calls = []
        for tool_call in group.tool_calls:
            tc: dict[str, Any] = {
                "id": tool_call.call_id,
                "type": "function",
                "function": {"name": tool_call.name, "arguments": tool_call.arguments},
            }
            # Include provider-specific extra content (e.g., Google thought signatures)
            if tool_call.extra.get("google"):
                tc["extra_content"] = {"google": tool_call.extra["google"]}
            tool_calls.append(tc)
        if tool_calls:
            msg["tool_calls"] = tool_calls
        messages.append(msg)

        # append tool outputs following the tool calls
        for tool_output in group.tool_outputs:
            messages.append(_to_chat_item(tool_output))

    return messages, None


def _to_chat_item(msg: llm.ChatItem) -> dict[str, Any]:
    if msg.type == "message":
        list_content: list[dict[str, Any]] = []
        text_content = ""
        for content in msg.content:
            if isinstance(content, str):
                if text_content:
                    text_content += "\n"
                text_content += content
            elif isinstance(content, llm.ImageContent):
                list_content.append(_to_image_content(content))

        if not list_content:
            # certain providers require text-only content in a string vs a list.
            # for max-compatibility, we will combine all text content into a single string.
            result: dict[str, Any] = {"role": msg.role, "content": text_content}
        else:
            if text_content:
                list_content.append({"type": "text", "text": text_content})
            result = {"role": msg.role, "content": list_content}

        # Include provider-specific extra content (e.g., Google thought signatures)
        if msg.extra.get("google"):
            result["extra_content"] = {"google": msg.extra["google"]}
        return result

    elif msg.type == "function_call":
        tc: dict[str, Any] = {
            "id": msg.call_id,
            "type": "function",
            "function": {
                "name": msg.name,
                "arguments": msg.arguments,
            },
        }
        # Include provider-specific extra content (e.g., Google thought signatures)
        if msg.extra.get("google"):
            tc["extra_content"] = {"google": msg.extra["google"]}
        return {
            "role": "assistant",
            "tool_calls": [tc],
        }

    elif msg.type == "function_call_output":
        return {
            "role": "tool",
            "tool_call_id": msg.call_id,
            "content": llm.utils.tool_output_to_text(msg.output),
        }

    raise ValueError(f"unsupported message type: {msg.type}")


def _to_image_content(image: llm.ImageContent) -> dict[str, Any]:
    img = llm.utils.serialize_image(image)
    if img.external_url:
        return {
            "type": "image_url",
            "image_url": {
                "url": img.external_url,
                "detail": img.inference_detail,
            },
        }
    assert img.data_bytes is not None
    b64_data = base64.b64encode(img.data_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{img.mime_type};base64,{b64_data}",
            "detail": img.inference_detail,
        },
    }


def _to_responses_image_content(image: llm.ImageContent) -> dict[str, Any]:
    img = llm.utils.serialize_image(image)
    if img.external_url:
        return {
            "type": "input_image",
            "image_url": img.external_url,
            "detail": img.inference_detail,
        }
    assert img.data_bytes is not None
    b64_data = base64.b64encode(img.data_bytes).decode("utf-8")
    return {
        "type": "input_image",
        "image_url": f"data:{img.mime_type};base64,{b64_data}",
        "detail": img.inference_detail,
    }


def to_responses_chat_ctx(
    chat_ctx: llm.ChatContext, *, inject_dummy_user_message: bool = True
) -> tuple[list[dict], Literal[None]]:
    item_groups = group_tool_calls(chat_ctx)
    items = []
    for group in item_groups:
        if not group.message and not group.tool_calls and not group.tool_outputs:
            continue

        if group.message:
            msg = _to_responses_chat_item(group.message)
            items.append(msg)

        for tool_call in group.tool_calls:
            call = {
                "call_id": tool_call.call_id,
                "type": "function_call",
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
            items.append(call)

        for tool_output in group.tool_outputs:
            items.append(_to_responses_chat_item(tool_output))

    return items, None


def _to_responses_chat_item(msg: llm.ChatItem) -> dict[str, Any]:
    if msg.type == "message":
        list_content: list[dict[str, Any]] = []
        text_content = ""
        for content in msg.content:
            if isinstance(content, str):
                if text_content:
                    text_content += "\n"
                text_content += content
            elif isinstance(content, llm.ImageContent):
                list_content.append(_to_responses_image_content(content))

        if not list_content:
            return {"role": msg.role, "content": text_content}

        if text_content:
            list_content.append({"type": "input_text", "text": text_content})

        return {"role": msg.role, "content": list_content}

    elif msg.type == "function_call_output":
        return {
            "type": "function_call_output",
            "call_id": msg.call_id,
            "output": _to_responses_tool_output(msg.output),
        }

    raise ValueError(f"unsupported message type: {msg.type}")


def to_fnc_ctx(tool_ctx: llm.ToolContext, *, strict: bool = True) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for tool in tool_ctx.function_tools.values():
        if isinstance(tool, llm.RawFunctionTool):
            schemas.append(
                {
                    "type": "function",
                    "function": tool.info.raw_schema,
                }
            )

        elif isinstance(tool, llm.FunctionTool):
            schema = (
                llm.utils.build_strict_openai_schema(tool)
                if strict
                else llm.utils.build_legacy_openai_schema(tool)
            )
            schemas.append(schema)

    return schemas


def _to_responses_tool_output(output: Any) -> str | list[dict[str, Any]]:
    normalized = llm.utils.normalize_function_output_value(output)
    if isinstance(normalized, str):
        return normalized

    parts: list[dict[str, Any]] = []
    for part in llm.utils.tool_output_parts(normalized):
        if isinstance(part, str):
            parts.append({"type": "input_text", "text": part})
        else:
            parts.append(_to_responses_image_content(part))

    return parts or ""


def to_responses_fnc_ctx(tool_ctx: llm.ToolContext, *, strict: bool = True) -> list[dict[str, Any]]:
    from livekit.plugins import openai

    schemas: list[dict[str, Any]] = []
    for tool in tool_ctx.flatten():
        if isinstance(tool, llm.RawFunctionTool):
            schema = tool.info.raw_schema
            schema["type"] = "function"
            schemas.append(schema)
        elif isinstance(tool, llm.FunctionTool):
            schema = llm.utils.build_legacy_openai_schema(tool, internally_tagged=True)
            schemas.append(schema)
        elif isinstance(tool, openai.tools.OpenAITool):
            schemas.append(tool.to_dict())

    return schemas
