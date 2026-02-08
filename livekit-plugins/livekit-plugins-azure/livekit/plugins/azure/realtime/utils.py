from __future__ import annotations

import base64
from typing import Any, cast

from azure.ai.voicelive.models import (
    AssistantMessageItem,
    AudioInputTranscriptionOptions,
    FunctionCallItem,
    FunctionCallOutputItem,
    FunctionTool,
    InputAudioContentPart,
    InputAudioFormat,
    InputTextContentPart,
    MessageContentPart,
    Modality,
    OutputAudioFormat,
    OutputTextContentPart,
    ServerVad,
    SystemMessageItem,
    TurnDetection,
    UserMessageItem,
)
from livekit import rtc
from livekit.agents import llm
from livekit.agents.types import NotGivenOr
from livekit.agents.utils import is_given

from ..log import logger

# Default configurations for Azure Voice Live
DEFAULT_INPUT_AUDIO_TRANSCRIPTION = AudioInputTranscriptionOptions(
    model="whisper-1",
)

DEFAULT_TURN_DETECTION = ServerVad(
    threshold=0.5,
    prefix_padding_ms=300,
    silence_duration_ms=500,
    create_response=True,
)

DEFAULT_MODALITIES = [Modality.TEXT, Modality.AUDIO]
DEFAULT_INPUT_AUDIO_FORMAT = InputAudioFormat.PCM16
DEFAULT_OUTPUT_AUDIO_FORMAT = OutputAudioFormat.PCM16
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_OUTPUT_TOKENS = 4096


def to_turn_detection(
    turn_detection: NotGivenOr[TurnDetection | None],
) -> TurnDetection | None:
    """Convert turn detection configuration to Azure TurnDetection format.

    Accepts any TurnDetection subclass including:
    - ServerVad: Basic server-side VAD
    - AzureSemanticVad: Semantic VAD (multilingual)
    - AzureSemanticVadEn: English-only semantic VAD
    - AzureSemanticVadMultilingual: Explicit multilingual semantic VAD
    """
    if not is_given(turn_detection):
        return DEFAULT_TURN_DETECTION

    if turn_detection is None:
        return None

    return turn_detection


def to_audio_transcription(
    audio_transcription: NotGivenOr[AudioInputTranscriptionOptions | None],
) -> AudioInputTranscriptionOptions | None:
    """Convert audio transcription configuration to Azure AudioInputTranscriptionOptions format.

    Args:
        audio_transcription: Audio transcription options. If NOT_GIVEN, returns default config.
            If None, transcription is disabled. Otherwise, returns the provided config.

    Returns:
        AudioInputTranscriptionOptions or None if transcription is disabled.
    """
    if not is_given(audio_transcription):
        return DEFAULT_INPUT_AUDIO_TRANSCRIPTION

    if audio_transcription is None:
        return None

    return audio_transcription


def livekit_tool_to_azure_tool(tool: llm.Tool) -> FunctionTool:
    """Convert LiveKit Tool to Azure FunctionTool format."""
    from livekit.agents.llm import utils as llm_utils

    # Handle FunctionTool and RawFunctionTool
    if isinstance(tool, llm.FunctionTool):
        # Use the build_legacy_openai_schema to get the schema
        schema = llm_utils.build_legacy_openai_schema(tool, internally_tagged=True)

        parameters = schema.get("parameters", {})

        azure_tool = FunctionTool(
            name=schema["name"],
            description=schema.get("description", ""),
            parameters=parameters,
        )

        logger.info(f"[TOOL_CONVERSION] Converted tool {schema['name']}")
        logger.debug(f"[TOOL_CONVERSION] Schema: {schema}")
        logger.debug(f"[TOOL_CONVERSION] Azure tool parameters: {parameters}")

        return azure_tool
    elif isinstance(tool, llm.RawFunctionTool):
        # For RawFunctionTool, extract schema from info.raw_schema
        raw_schema = tool.info.raw_schema
        azure_tool = FunctionTool(
            name=tool.info.name,
            description=raw_schema.get("description", ""),
            parameters=raw_schema.get("parameters", {}),
        )

        logger.info(f"[TOOL_CONVERSION] Converted raw tool {tool.info.name}")
        return azure_tool
    else:
        raise ValueError(f"Unsupported tool type: {type(tool)}")


# Type alias for Azure conversation items
AzureConversationItem = (
    SystemMessageItem
    | UserMessageItem
    | AssistantMessageItem
    | FunctionCallItem
    | FunctionCallOutputItem
)


def livekit_item_to_azure_item(item: llm.ChatItem) -> AzureConversationItem:
    if item.type == "function_call_output":
        return FunctionCallOutputItem(call_id=item.call_id, output=item.output)

    if item.type == "function_call":
        return FunctionCallItem(
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments,
            id=item.id,
        )

    if item.type == "message":
        if item.role in ("system", "developer"):
            content_parts: list[MessageContentPart] = [
                InputTextContentPart(text=c) for c in item.content if isinstance(c, str)
            ]
            return SystemMessageItem(content=content_parts, id=item.id)

        if item.role == "assistant":
            content_parts = [
                OutputTextContentPart(text=c) for c in item.content if isinstance(c, str)
            ]
            return AssistantMessageItem(content=content_parts, id=item.id)

        if item.role == "user":
            content_parts = []
            for c in item.content:
                if isinstance(c, str):
                    content_parts.append(InputTextContentPart(text=c))
                elif isinstance(c, llm.AudioContent):
                    encoded_audio = base64.b64encode(rtc.combine_audio_frames(c.frame).data).decode(
                        "utf-8"
                    )
                    content_parts.append(
                        InputAudioContentPart(audio=encoded_audio, transcript=c.transcript)
                    )
            return UserMessageItem(content=content_parts, id=item.id)
        raise ValueError(f"Unsupported role: {item.role}")
    raise ValueError(f"Unsupported item type: {item.type}")


def azure_item_to_livekit_item(item: dict[str, Any]) -> llm.ChatItem:
    """Convert Azure conversation item to LiveKit ChatItem."""
    item_id = item.get("id")
    item_type = item.get("type")

    if item_type == "function_call":
        return llm.FunctionCall(
            id=item_id,
            call_id=item["call_id"],
            name=item["name"],
            arguments=item["arguments"],
        )

    if item_type == "function_call_output":
        return llm.FunctionCallOutput(
            id=item_id,
            call_id=item["call_id"],
            output=item["output"],
            is_error=False,
        )

    if item_type == "message":
        role = item.get("role")
        content_list = item.get("content", [])

        chat_content: list[llm.ChatContent] = []
        for c in content_list:
            content_type = c.get("type")
            if content_type in ("input_text", "text") and c.get("text"):
                chat_content.append(c["text"])
            elif content_type == "input_audio" and c.get("transcript"):
                chat_content.append(c["transcript"])

        return llm.ChatMessage(
            id=item_id,
            role=role,
            content=chat_content,
        )

    raise ValueError(f"Unsupported item type: {item_type}")
