from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any, Literal

from livekit.agents import llm
from livekit.agents.log import logger

from .utils import group_tool_calls


@dataclass
class GoogleFormatData:
    system_messages: list[str] | None


def to_chat_ctx(
    chat_ctx: llm.ChatContext,
    *,
    inject_dummy_user_message: bool = True,
    thought_signatures: dict[str, bytes] | None = None,
) -> tuple[list[dict], GoogleFormatData]:
    turns: list[dict] = []
    system_messages: list[str] = []
    current_role: str | None = None
    parts: list[dict] = []

    for msg in itertools.chain(*(group.flatten() for group in group_tool_calls(chat_ctx))):
        if msg.type == "message" and msg.role == "system" and (text := msg.text_content):
            system_messages.append(text)
            continue

        if msg.type == "message":
            role = "model" if msg.role == "assistant" else "user"
        elif msg.type == "function_call":
            role = "model"
        elif msg.type == "function_call_output":
            # tool output shouldn't be mixed with other messages
            role = "tool"

        # if the effective role changed, finalize the previous turn.
        if role != current_role:
            if current_role is not None and parts:
                turns.append({"role": current_role, "parts": parts})
            parts = []
            current_role = role

        if msg.type == "message":
            for content in msg.content:
                if content and isinstance(content, str):
                    parts.append({"text": content})
                elif content and isinstance(content, dict):
                    parts.append({"text": json.dumps(content)})
                elif isinstance(content, llm.ImageContent):
                    parts.append(_to_image_part(content))
        elif msg.type == "function_call":
            fc_part: dict[str, Any] = {
                "function_call": {
                    "id": msg.call_id,
                    "name": msg.name,
                    "args": json.loads(msg.arguments or "{}"),
                }
            }
            # Inject thought_signature if available (Gemini 3 multi-turn function calling)
            if thought_signatures and (sig := thought_signatures.get(msg.call_id)):
                fc_part["thought_signature"] = sig
            parts.append(fc_part)
        elif msg.type == "function_call_output":
            if msg.is_error:
                response = {"error": llm.utils.tool_output_to_text(msg.output)}
                parts.append(
                    {
                        "function_response": {
                            "id": msg.call_id,
                            "name": msg.name,
                            "response": response,
                        }
                    }
                )
            else:
                text_output = llm.utils.tool_output_to_text(
                    msg.output, include_image_placeholder=False
                )
                _, image_parts = llm.utils.split_tool_output_parts(msg.output)
                response_payload: dict[str, Any] = {}
                if text_output:
                    response_payload["output"] = text_output
                function_response: dict[str, Any] = {
                    "id": msg.call_id,
                    "name": msg.name,
                    "response": response_payload,
                }
                if image_parts:
                    function_response["parts"] = [_to_image_part(image) for image in image_parts]
                parts.append({"function_response": function_response})

    if current_role is not None and parts:
        turns.append({"role": current_role, "parts": parts})

    # convert role tool to user for gemini
    for turn in turns:
        if turn["role"] == "tool":
            turn["role"] = "user"

    # Gemini requires the last message to end with user's turn before they can generate
    # allow tool role since we update it to user in the previous step
    if inject_dummy_user_message and current_role not in ("user", "tool"):
        turns.append({"role": "user", "parts": [{"text": "."}]})

    return turns, GoogleFormatData(system_messages=system_messages)


def _to_image_part(image: llm.ImageContent) -> dict[str, Any]:
    cache_key = "serialized_image"
    if cache_key not in image._cache:
        image._cache[cache_key] = llm.utils.serialize_image(image)
    img: llm.utils.SerializedImage = image._cache[cache_key]

    if img.external_url:
        if img.mime_type:
            mime_type = img.mime_type
        else:
            logger.debug("No media type provided for image, defaulting to image/jpeg.")
            mime_type = "image/jpeg"
        return {"file_data": {"file_uri": img.external_url, "mime_type": mime_type}}

    return {"inline_data": {"data": img.data_bytes, "mime_type": img.mime_type}}


TOOL_BEHAVIOR = Literal["UNSPECIFIED", "BLOCKING", "NON_BLOCKING"]


def to_fnc_ctx(
    tool_ctx: llm.ToolContext,
    *,
    tool_behavior: TOOL_BEHAVIOR | None = None,
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in tool_ctx.function_tools.values():
        if isinstance(tool, llm.RawFunctionTool):
            info = tool.info
            schema = {
                "name": info.name,
                "description": info.raw_schema.get("description", ""),
                "parameters_json_schema": info.raw_schema.get("parameters", {}),
            }
            if tool_behavior is not None:
                schema["behavior"] = tool_behavior
            tools.append(schema)

        elif isinstance(tool, llm.FunctionTool):
            from livekit.plugins.google.utils import _GeminiJsonSchema

            fnc = llm.utils.build_legacy_openai_schema(tool, internally_tagged=True)
            json_schema = _GeminiJsonSchema(fnc["parameters"]).simplify()

            schema = {
                "name": fnc["name"],
                "description": fnc["description"],
                "parameters": json_schema or None,
            }
            if tool_behavior is not None:
                schema["behavior"] = tool_behavior
            tools.append(schema)

    return tools
