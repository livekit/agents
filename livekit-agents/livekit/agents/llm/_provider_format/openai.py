from __future__ import annotations

import base64
from typing import Any, Literal

from livekit.agents import llm

from .utils import group_tool_calls

_EXTRA_CONTENT_KEYS = ("google", "livekit", "xai")


def _filter_extra(extra: dict[str, Any]) -> dict[str, Any]:
    return {k: extra[k] for k in _EXTRA_CONTENT_KEYS if extra.get(k)}


def to_chat_ctx(
    chat_ctx: llm.ChatContext,
    *,
    inject_dummy_user_message: bool = True,
    prompt_cache_breakpoints: bool = False,
) -> tuple[list[dict], Literal[None]]:
    """Convert to Chat Completions messages.

    With ``prompt_cache_breakpoints``, text before each :class:`llm.CacheBreakpoint`
    is sent as its own part tagged with ``prompt_cache_breakpoint``. Without it the
    markers are dropped and the output matches a context that never had them.
    """
    item_groups = group_tool_calls(chat_ctx)
    messages = []
    for group in item_groups:
        if not group.message and not group.tool_calls and not group.tool_outputs:
            continue

        # one message can contain zero or more tool calls
        msg = (
            _to_chat_item(group.message, prompt_cache_breakpoints=prompt_cache_breakpoints)
            if group.message
            else {"role": "assistant"}
        )
        tool_calls = []
        for tool_call in group.tool_calls:
            tc: dict[str, Any] = {
                "id": tool_call.call_id,
                "type": "function",
                "function": {"name": tool_call.name, "arguments": tool_call.arguments},
            }
            extra_content = _filter_extra(tool_call.extra) if tool_call.extra else {}
            if extra_content:
                tc["extra_content"] = extra_content
            tool_calls.append(tc)
        if tool_calls:
            msg["tool_calls"] = tool_calls  # type: ignore[assignment]
        messages.append(msg)

        # append tool outputs following the tool calls
        for tool_output in group.tool_outputs:
            messages.append(_to_chat_item(tool_output))

    return messages, None


def _to_chat_item(msg: llm.ChatItem, *, prompt_cache_breakpoints: bool = False) -> dict[str, Any]:
    if msg.type == "message":
        images = [_to_image_content(c) for c in msg.content if isinstance(c, llm.ImageContent)]
        text, cuts = _text_and_cuts(msg, breakpoints=prompt_cache_breakpoints)
        result: dict[str, Any] = {
            "role": msg.role,
            "content": _message_content(images, text, cuts, text_type="text"),
        }

        extra_content = _filter_extra(msg.extra)
        if extra_content:
            result["extra_content"] = extra_content
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
        extra_content = _filter_extra(msg.extra)
        if extra_content:
            tc["extra_content"] = extra_content
        return {
            "role": "assistant",
            "tool_calls": [tc],
        }

    elif msg.type == "function_call_output":
        return {
            "role": "tool",
            "tool_call_id": msg.call_id,
            "content": msg.output,
        }

    raise ValueError(f"unsupported message type: {msg.type}")


def _text_and_cuts(msg: llm.ChatMessage, *, breakpoints: bool) -> tuple[str, list[int]]:
    """Join the message's text and record where cache breakpoints split it.

    The text is joined exactly as it was before breakpoints existed, so cutting it
    never changes what the model reads. A breakpoint with no new text since the last
    cut (leading, or repeated) is ignored.
    """
    text = ""
    cuts: list[int] = []
    for content in msg.content:
        if isinstance(content, (llm.ImageContent, llm.AudioContent)):
            continue
        if isinstance(content, llm.CacheBreakpoint):
            if breakpoints and len(text) > (cuts[-1] if cuts else 0):
                cuts.append(len(text))
            continue
        # str or Instructions
        if text:
            text += "\n"
        text += str(content)
    return text, cuts


def _text_parts(text: str, cuts: list[int], *, text_type: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    start = 0
    for cut in cuts:
        parts.append(
            {
                "type": text_type,
                "text": text[start:cut],
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        )
        start = cut
    if start < len(text):
        parts.append({"type": text_type, "text": text[start:]})
    return parts


def _message_content(
    images: list[dict[str, Any]], text: str, cuts: list[int], *, text_type: str
) -> str | list[dict[str, Any]]:
    if not images and not cuts:
        # certain providers require text-only content in a string vs a list.
        # for max-compatibility, we will combine all text content into a single string.
        return text
    return images + _text_parts(text, cuts, text_type=text_type)


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
    chat_ctx: llm.ChatContext,
    *,
    inject_dummy_user_message: bool = True,
    prompt_cache_breakpoints: bool = False,
) -> tuple[list[dict], Literal[None]]:
    """Convert to Responses API input items.

    ``prompt_cache_breakpoints`` behaves as in :func:`to_chat_ctx`, except that
    assistant messages never carry one.
    """
    item_groups = group_tool_calls(chat_ctx)
    items = []
    for group in item_groups:
        if not group.message and not group.tool_calls and not group.tool_outputs:
            continue

        if group.message:
            msg = _to_responses_chat_item(
                group.message, prompt_cache_breakpoints=prompt_cache_breakpoints
            )
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


def _to_responses_chat_item(
    msg: llm.ChatItem, *, prompt_cache_breakpoints: bool = False
) -> dict[str, Any]:
    if msg.type == "message":
        images = [
            _to_responses_image_content(c) for c in msg.content if isinstance(c, llm.ImageContent)
        ]
        # Responses assistant content is output text, which has no prompt_cache_breakpoint
        breakpoints = prompt_cache_breakpoints and msg.role != "assistant"
        text, cuts = _text_and_cuts(msg, breakpoints=breakpoints)
        item: dict[str, Any] = {
            "role": msg.role,
            "content": _message_content(images, text, cuts, text_type="input_text"),
        }

        # Re-attach the assistant message phase (commentary / final_answer) captured from
        # the Responses API. Dropping it on follow-up requests can degrade performance for
        # models like gpt-5.3-codex.
        if msg.role == "assistant":
            phase = msg.extra.get("openai", {}).get("phase")
            if phase is not None:
                item["phase"] = phase

        return item

    elif msg.type == "function_call_output":
        return {
            "type": "function_call_output",
            "call_id": msg.call_id,
            "output": msg.output,
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


def to_responses_fnc_ctx(
    tool_ctx: llm.ToolContext,
    *,
    strict: bool = True,
    provider_tool_type: type[llm.ProviderTool] | None = None,
) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for tool in tool_ctx.flatten():
        if isinstance(tool, llm.RawFunctionTool):
            schema = {**tool.info.raw_schema, "type": "function"}
            schemas.append(schema)
        elif isinstance(tool, llm.FunctionTool):
            schema = llm.utils.build_legacy_openai_schema(tool, internally_tagged=True)
            schemas.append(schema)
        elif (
            provider_tool_type is not None
            and isinstance(tool, provider_tool_type)
            and hasattr(tool, "to_dict")
        ):
            schemas.append(tool.to_dict())

    return schemas
