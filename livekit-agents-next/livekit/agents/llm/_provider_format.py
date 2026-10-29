"""Chat and tool contexts in each provider's request shape; the dialects are the core's
(``_core.to_provider_format``, ``_core.to_fnc_ctx``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from livekit.agents import _core, llm

if TYPE_CHECKING:
    from anthropic.types import MessageParam as AnthropicMessage
    from google.genai.types import ContentDict as GoogleMessage
    from mistralai.client.models import InputEntriesTypedDict as MistralMessage
    from openai.types.chat import ChatCompletionMessageParam as OpenAIMessage
    from openai.types.responses import ResponseInputItemParam as OpenAIResponseItem
    from types_aiobotocore_bedrock_runtime.type_defs import MessageTypeDef as BedrockMessage

    from .chat_context import ChatContext

    ProviderMessages: TypeAlias = (
        list[OpenAIMessage]
        | list[OpenAIResponseItem]
        | list[GoogleMessage]
        | list[BedrockMessage]
        | list[AnthropicMessage]
        | list[MistralMessage]
    )
    from .tool_context import ProviderTool, ToolContext

__all__ = [
    "TOOL_BEHAVIOR",
    "AnthropicFormatData",
    "BedrockFormatData",
    "GoogleFormatData",
    "MistralFormatData",
    "to_chat_ctx",
    "to_fnc_ctx",
]

_DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE = "<instructions>\n{content}\n</instructions>"


def convert_mid_conversation_instructions(
    chat_ctx: llm.ChatContext,
    *,
    role: llm.ChatRole = "user",
    template: str = _DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE,
) -> llm.ChatContext:
    """Rewrite every system or developer message but a leading preamble as a ``role`` message
    wrapped in ``template``, so it keeps its position; one without text is dropped."""

    plan = _core.convert_mid_conversation_instructions(chat_ctx.items, role=role, template=template)
    return llm.ChatContext([chat_ctx.items[i] if new is None else new for i, new in plan])


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


async def to_chat_ctx(
    chat_ctx: ChatContext,
    format: str,
    *,
    inject_dummy_user_message: bool = True,
    inject_trailing_user_message: bool = False,
    thought_signatures: dict[str, bytes] | None = None,
) -> tuple[
    ProviderMessages,
    GoogleFormatData | BedrockFormatData | AnthropicFormatData | MistralFormatData | None,
]:
    from .chat_context import ImageContent
    from .utils import serialize_image

    images = {
        content.id: await serialize_image(content)
        for item in chat_ctx.items
        if item.type == "message"
        for content in item.content
        if isinstance(content, ImageContent)
    }
    messages, data = await _core.to_provider_format(
        format,
        chat_ctx.items,
        images=images,
        inject_dummy_user_message=inject_dummy_user_message,
        inject_trailing_user_message=inject_trailing_user_message,
        thought_signatures=thought_signatures,
    )
    cls = _DATA.get(format)
    return cast("ProviderMessages", messages), cls(
        **cast("dict[str, Any]", data)
    ) if cls is not None else None


def to_fnc_ctx(
    tool_ctx: ToolContext,
    format: str,
    *,
    strict: bool = True,
    provider_tool_type: type[ProviderTool] | None = None,
    tool_behavior: TOOL_BEHAVIOR | None = None,
    use_parameters_json_schema: bool = True,
) -> list[dict[str, Any]]:
    from .tool_context import FunctionTool, RawFunctionTool
    from .utils import build_legacy_openai_schema, build_strict_openai_schema

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
            specs.append(["legacy", fnc])
        elif responses and provider_tool_type and isinstance(tool, provider_tool_type):
            if hasattr(tool, "to_dict"):
                specs.append(["provider", tool.to_dict()])
    return cast(
        "list[dict[str, Any]]",
        _core.to_fnc_ctx(
            format,
            specs,
            tool_behavior=None if tool_behavior == "UNSPECIFIED" else tool_behavior,
            use_parameters_json_schema=use_parameters_json_schema,
        ),
    )
