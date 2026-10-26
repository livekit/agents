"""Chat and tool contexts in each provider's request shape; the dialects are the core's
(``_core.to_provider_format``, ``_core.to_fnc_ctx``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ... import _core

if TYPE_CHECKING:
    from ..chat_context import ChatContext
    from ..tool_context import ProviderTool, ToolContext

__all__ = [
    "TOOL_BEHAVIOR",
    "AnthropicFormatData",
    "BedrockFormatData",
    "GoogleFormatData",
    "MistralFormatData",
    "to_chat_ctx",
    "to_fnc_ctx",
]

TOOL_BEHAVIOR = Literal["UNSPECIFIED", "BLOCKING", "NON_BLOCKING"]


@dataclass
class AnthropicFormatData:
    system_messages: list[str] | None


@dataclass
class BedrockFormatData:
    system_messages: list[str] | None


@dataclass
class GoogleFormatData:
    system_messages: list[str] | None


@dataclass
class MistralFormatData:
    instructions: str | None


_DATA = {
    "anthropic": AnthropicFormatData,
    "aws": BedrockFormatData,
    "google": GoogleFormatData,
    "mistralai": MistralFormatData,
}


def to_chat_ctx(
    chat_ctx: ChatContext,
    format: str,
    *,
    inject_dummy_user_message: bool = True,
    inject_trailing_user_message: bool = False,
    thought_signatures: dict[str, bytes] | None = None,
) -> tuple[list[dict], Any]:
    from ..._core_view import to_core_items
    from ..chat_context import ImageContent
    from ..utils import serialize_image
    from .utils import convert_mid_conversation_instructions

    if format in _DATA:
        # the dialects that hoist the system preamble keep later instructions where they are
        chat_ctx = convert_mid_conversation_instructions(chat_ctx)
    if format in ("anthropic", "google", "aws"):
        # these send arguments as a JSON object: stored ones are repaired, else emptied
        chat_ctx = chat_ctx.copy()
        chat_ctx.items = [_object_arguments(item) for item in chat_ctx.items]
    images = {
        content.id: serialize_image(content)
        for item in chat_ctx.items
        if item.type == "message"
        for content in item.content
        if isinstance(content, ImageContent)
    }
    messages, data = _core.to_provider_format(
        format,
        to_core_items(chat_ctx.items),
        images=images,
        inject_dummy_user_message=inject_dummy_user_message,
        inject_trailing_user_message=inject_trailing_user_message,
        thought_signatures=thought_signatures,
    )
    if format == "mistralai" and inject_dummy_user_message and messages:
        if messages[-1].get("role") == "assistant":
            messages.append({"type": "message.input", "role": "user", "content": "(empty)"})
    cls = _DATA.get(format)
    return messages, cls(**data) if cls is not None else None


def _object_arguments(item: Any) -> Any:
    from ...log import logger
    from ..utils import parse_function_arguments

    if item.type != "function_call":
        return item
    try:
        arguments = parse_function_arguments(item.arguments) if item.arguments else {}
    except ValueError:
        logger.warning(
            "could not parse stored tool call arguments as a JSON object, using empty arguments",
            extra={"call_id": item.call_id, "tool_name": item.name},
        )
        arguments = {}
    return item.model_copy(update={"arguments": json.dumps(arguments)})


def to_fnc_ctx(
    tool_ctx: ToolContext,
    format: str,
    *,
    strict: bool = True,
    provider_tool_type: type[ProviderTool] | None = None,
    tool_behavior: TOOL_BEHAVIOR | None = None,
    use_parameters_json_schema: bool = True,
) -> list[dict[str, Any]]:
    from ..tool_context import FunctionTool, RawFunctionTool
    from ..utils import build_legacy_openai_schema, build_strict_openai_schema

    responses = format == "openai.responses"
    specs: list[list[Any]] = []
    for tool in tool_ctx.flatten() if responses else tool_ctx.function_tools.values():
        if isinstance(tool, RawFunctionTool):
            specs.append(["raw", tool.info.raw_schema, tool.info.name])
        elif isinstance(tool, FunctionTool):
            if strict and format in ("openai", "anthropic"):
                specs.append(["strict", build_strict_openai_schema(tool)])
                continue
            fnc = build_legacy_openai_schema(tool, internally_tagged=format != "openai")
            if format == "google":
                # Gemini takes the schema as written, or simplified where it rejects that (Live)
                params = fnc["parameters"] if fnc["parameters"].get("properties") else None
                if not use_parameters_json_schema:
                    from livekit.plugins.google.utils import _GeminiJsonSchema

                    params = _GeminiJsonSchema(params or {}).simplify() or None
                raw = {"description": fnc["description"], "parameters": params}
                specs.append(["raw", raw, fnc["name"]])
                continue
            specs.append(["legacy", fnc])
        elif responses and provider_tool_type and isinstance(tool, provider_tool_type):
            if hasattr(tool, "to_dict"):
                specs.append(["provider", tool.to_dict()])
    return _core.to_fnc_ctx(  # type: ignore[no-any-return]
        format,
        specs,
        tool_behavior=None if tool_behavior == "UNSPECIFIED" else tool_behavior,
        use_parameters_json_schema=use_parameters_json_schema,
    )
