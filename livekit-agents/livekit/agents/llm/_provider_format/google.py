from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any, Literal

from livekit.agents import llm
from livekit.agents.log import logger

from .utils import convert_mid_conversation_instructions, group_tool_calls


@dataclass
class GoogleFormatData:
    system_messages: list[str] | None


def to_chat_ctx(
    chat_ctx: llm.ChatContext,
    *,
    inject_dummy_user_message: bool = True,
    thought_signatures: dict[str, bytes] | None = None,
) -> tuple[list[dict], GoogleFormatData]:
    chat_ctx = convert_mid_conversation_instructions(chat_ctx)

    turns: list[dict] = []
    system_messages: list[str] = []
    current_role: str | None = None
    parts: list[dict] = []

    # Build call_id -> function name map for strict function response matching.
    # Gemini requires every function_response to carry the call's id and match
    # the originating function_call's name. We trust the function_call as the
    # source of truth and rewrite mismatched response names.
    call_id_to_name: dict[str, str] = {}
    for item in chat_ctx.items:
        if item.type == "function_call" and item.call_id:
            call_id_to_name[item.call_id] = item.name

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
            # On assistant turns Gemini benefits from receiving the thought
            # signature of the previous response. Attach it to the first part
            # of this assistant message so it travels alongside the text.
            msg_signature: bytes | None = None
            if msg.role == "assistant":
                msg_signature = (
                    (msg.extra or {}).get("google", {}).get("thought_signature")
                )
            first_part_idx = len(parts)
            for content in msg.content:
                if content and isinstance(content, str):
                    parts.append({"text": content})
                elif content and isinstance(content, dict):
                    parts.append({"text": json.dumps(content)})
                elif isinstance(content, llm.ImageContent):
                    parts.append(_to_image_part(content))
            if msg_signature is not None and first_part_idx < len(parts):
                parts[first_part_idx]["thought_signature"] = msg_signature
        elif msg.type == "function_call":
            fc_part: dict[str, Any] = {
                "function_call": {
                    "id": msg.call_id,
                    "name": msg.name,
                    "args": json.loads(msg.arguments or "{}"),
                }
            }
            # Inject thought_signature if available (Gemini 2.5+ multi-turn function calling).
            # Prefer the signature stored on the item itself (extra["google"]), and fall back
            # to the runtime dict for backward compatibility with older callers.
            sig: bytes | None = (
                (msg.extra or {}).get("google", {}).get("thought_signature")
            )
            if sig is None and thought_signatures:
                sig = thought_signatures.get(msg.call_id)
            if sig:
                fc_part["thought_signature"] = sig
            parts.append(fc_part)
        elif msg.type == "function_call_output":
            if not msg.call_id:
                logger.warning(
                    "skipping function_call_output without call_id; "
                    "Gemini requires id on every function_response",
                    extra={"tool_name": msg.name},
                )
                continue

            expected_name = call_id_to_name.get(msg.call_id)
            if expected_name is None:
                logger.warning(
                    "skipping function_call_output with no matching function_call",
                    extra={"call_id": msg.call_id, "tool_name": msg.name},
                )
                continue
            if expected_name != msg.name:
                logger.warning(
                    "function_call_output name does not match originating function_call; "
                    "rewriting to match for strict Gemini function response matching",
                    extra={
                        "call_id": msg.call_id,
                        "output_name": msg.name,
                        "call_name": expected_name,
                    },
                )

            response = {"output": msg.output} if not msg.is_error else {"error": msg.output}
            parts.append(
                {
                    "function_response": {
                        "id": msg.call_id,
                        "name": expected_name,
                        "response": response,
                    }
                }
            )

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
    use_parameters_json_schema: bool = True,
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in tool_ctx.function_tools.values():
        if isinstance(tool, llm.RawFunctionTool):
            info = tool.info
            schema = {
                "name": info.name,
                "description": info.raw_schema.get("description", ""),
            }
            if use_parameters_json_schema:
                schema["parameters_json_schema"] = info.raw_schema.get("parameters", {})
            else:
                # Gemini Live doesn't support parameters_json_schema, use the simplified JSON Schema instead
                # see: https://github.com/googleapis/python-genai/issues/1147
                from livekit.plugins.google.utils import _GeminiJsonSchema

                schema["parameters"] = (
                    _GeminiJsonSchema(info.raw_schema.get("parameters", {})).simplify() or None
                )

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
