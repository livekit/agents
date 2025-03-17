import base64
import json
from typing import Any, Literal

import anthropic
from livekit.agents import llm
from livekit.agents.llm import FunctionTool

CACHE_CONTROL_EPHEMERAL = anthropic.types.CacheControlEphemeralParam(type="ephemeral")

__all__ = ["to_fnc_ctx", "to_chat_ctx"]


def to_fnc_ctx(
    fncs: list[FunctionTool], caching: Literal["ephemeral"] | None
) -> list[anthropic.types.ToolParam]:
    tools: list[anthropic.types.ToolParam] = []
    for i, fnc in enumerate(fncs):
        cache_ctrl = (
            CACHE_CONTROL_EPHEMERAL if (i == len(fncs) - 1) and caching == "ephemeral" else None
        )
        tools.append(_build_anthropic_schema(fnc, cache_ctrl=cache_ctrl))

    return tools


def to_chat_ctx(
    chat_ctx: llm.ChatContext,
    cache_key: Any,
    caching: Literal["ephemeral"] | None,
) -> list[anthropic.types.MessageParam]:
    messages: list[anthropic.types.MessageParam] = []
    system_message: anthropic.types.TextBlockParam | None = None
    current_role: str | None = None
    content: list[anthropic.types.TextBlockParam] = []
    for i, msg in enumerate(chat_ctx.items):
        if msg.type == "message" and msg.role == "system":
            for content in msg.content:
                if isinstance(content, str):
                    system_message = anthropic.types.TextBlockParam(
                        text=content,
                        type="text",
                        cache_control=CACHE_CONTROL_EPHEMERAL if caching == "ephemeral" else None,
                    )
            continue

        cache_ctrl = (
            CACHE_CONTROL_EPHEMERAL
            if (i == len(chat_ctx.items) - 1) and caching == "ephemeral"
            else None
        )
        if msg.type == "message":
            role = "assistant" if msg.role == "assistant" else "user"
        elif msg.type == "function_call":
            role = "assistant"
        elif msg.type == "function_call_output":
            role = "user"

        if role != current_role:
            if current_role is not None and content:
                messages.append(anthropic.types.MessageParam(role=current_role, content=content))
            content = []
            current_role = role

        if msg.type == "message":
            for c in msg.content:
                if isinstance(c, str):
                    content.append(
                        anthropic.types.TextBlockParam(
                            text=c, type="text", cache_control=cache_ctrl
                        )
                    )
                elif isinstance(c, llm.ImageContent):
                    content.append(_to_image_content(c, cache_key, cache_ctrl=cache_ctrl))
        elif msg.type == "function_call":
            content.append(
                anthropic.types.ToolUseBlockParam(
                    id=msg.call_id,
                    type="tool_use",
                    name=msg.name,
                    input=json.loads(msg.arguments or "{}"),
                    cache_control=cache_ctrl,
                )
            )
        elif msg.type == "function_call_output":
            content.append(
                anthropic.types.ToolResultBlockParam(
                    tool_use_id=msg.call_id,
                    type="tool_result",
                    content=msg.output,
                    cache_control=cache_ctrl,
                )
            )

    if current_role is not None and content:
        messages.append(anthropic.types.MessageParam(role=current_role, content=content))

    # ensure the messages starts with a "user" message
    if not messages or messages[0]["role"] != "user":
        messages.insert(
            0,
            anthropic.types.MessageParam(
                role="user", content=[anthropic.types.TextBlockParam(text="(empty)", type="text")]
            ),
        )

    return messages, system_message


def _to_image_content(
    image: llm.ImageContent,
    cache_key: Any,
    cache_ctrl: anthropic.types.CacheControlEphemeralParam | None,
) -> anthropic.types.ImageBlockParam:
    img = llm.utils.serialize_image(image)
    if cache_key not in image._cache:
        image._cache[cache_key] = img.data_bytes
    b64_data = base64.b64encode(image._cache[cache_key]).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "data": f"data:{img.media_type};base64,{b64_data}",
            "media_type": img.media_type,
        },
        "cache_control": cache_ctrl,
    }


def _build_anthropic_schema(
    function_tool: FunctionTool,
    cache_ctrl: anthropic.types.CacheControlEphemeralParam | None = None,
) -> anthropic.types.ToolParam:
    fnc = llm.utils.build_legacy_openai_schema(function_tool, internally_tagged=True)
    return anthropic.types.ToolParam(
        name=fnc["name"],
        description=fnc["description"] or "",
        input_schema=fnc["parameters"],
        cache_control=cache_ctrl,
    )
