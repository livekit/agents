import base64
import json
from typing import Any, Literal

import llama_api_client
from llama_api_client.types.chat.completion_create_params import ToolFunction
from livekit.agents import llm
from livekit.agents.llm import FunctionTool

__all__ = ["to_fnc_ctx", "to_chat_ctx"]


def to_fnc_ctx(fncs: list[FunctionTool]) -> list[ToolFunction]:
    tools: list[ToolFunction] = []
    for _, fnc in enumerate(fncs):
        tools.append(_build_llama_schema(fnc))

    return tools


def to_chat_ctx(chat_ctx: llm.ChatContext) -> list[llama_api_client.types.MessageParam]:
    messages: list[llama_api_client.types.MessageParam] = []
    current_role: str | None = None
    content: list[llama_api_client.types.MessageTextContentItemParam] = []
    for _, msg in enumerate(chat_ctx.items):
        if msg.type == "message" and msg.role == "system":
            role = "system"
        elif msg.type == "message":
            role = "assistant" if msg.role == "assistant" else "user"
        elif msg.type == "function_call":
            role = "assistant"
        elif msg.type == "function_call_output":
            role = "tool"

        if role != current_role and role != "system":
            if current_role is not None and content:
                messages.append(
                    llama_api_client.types.MessageParam(
                        role=current_role, content=content
                    )
                )
            content = []
            current_role = role

        if msg.type == "message":
            for c in msg.content:
                if c and isinstance(c, str):
                    content.append(
                        llama_api_client.types.MessageTextContentItemParam(
                            text=c, type="text"
                        )
                    )
                elif isinstance(c, llm.ImageContent):
                    content.append(_to_image_content(c))
        elif msg.type == "function_call":
            content.append(
                llama_api_client.types.completion_message.ToolCall(
                    tool_call_id=msg.call_id,
                    type="tool_use",
                    name=msg.name,
                    arguments=msg.arguments or "{}",
                )
            )
        elif msg.type == "function_call_output":
            content.append(
                llama_api_client.types.ToolResponseMessageParam(
                    tool_call_id=msg.call_id,
                    role="tool",
                    content=msg.output,
                )
            )

    if current_role is not None and content:
        messages.append(
            llama_api_client.types.MessageParam(role=current_role, content=content)
        )

    # ensure the messages starts with a "user" message
    if not messages or messages[0]["role"] != "user":
        messages.insert(
            0,
            llama_api_client.types.MessageParam(
                role="user",
                content=[
                    llama_api_client.types.MessageTextContentItemParam(
                        text="(empty)", type="text"
                    )
                ],
            ),
        )

    return messages


def _to_image_content(
    image: llm.ImageContent,
) -> llama_api_client.types.MessageImageContentItemParam:
    img = llm.utils.serialize_image(image)
    if img.external_url:
        return llama_api_client.types.MessageImageContentItemParam(
            image_url=img.external_url,
        )
    b64_data = base64.b64encode(img.data_bytes).decode("utf-8")
    return llama_api_client.types.MessageImageContentItemParam(
        image_url=f"data:{img.mime_type};base64,{b64_data}",
    )


def _build_llama_schema(function_tool: FunctionTool) -> ToolFunction:
    fnc = llm.utils.build_legacy_openai_schema(function_tool, internally_tagged=True)
    return ToolFunction(
        name=fnc["name"],
        description=fnc["description"] or "",
        parameters=fnc["parameters"],
    )
