from __future__ import annotations

import base64
from collections.abc import Sequence
from enum import Enum

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
    ResponseFunctionCallItem,
    ResponseFunctionCallOutputItem,
    ResponseItem,
    ResponseMessageItem,
    ServerVad,
    SystemMessageItem,
    Tool,
    ToolChoiceFunctionSelection,
    ToolChoiceLiteral,
    ToolChoiceSelection,
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
# default of the models that don't support whisper-1, see `uses_azure_speech`
AZURE_SPEECH_INPUT_AUDIO_TRANSCRIPTION = AudioInputTranscriptionOptions(model="azure-speech")

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


def uses_azure_speech(model: str) -> bool:
    """Whether the input audio of the model is transcribed with azure-speech, not whisper-1.

    Voice Live documents whisper-1 for gpt-realtime and gpt-realtime-mini, and azure-speech for
    the non-multimodal (text) models and phi4-mm-realtime, e.g. gpt-4.1 answers whisper-1 with
    invalid_input_audio_transcription_model. The multimodal models have "realtime" in their
    name, those without documented transcription models (e.g. gpt-realtime-1.5, azure-realtime)
    keep whisper-1.

    See https://learn.microsoft.com/azure/ai-services/speech-service/voice-live-how-to and
    https://learn.microsoft.com/azure/ai-services/speech-service/voice-live-language-support
    """
    name = model.lower()
    return "realtime" not in name or name.startswith("phi")


def to_audio_transcription(
    audio_transcription: NotGivenOr[AudioInputTranscriptionOptions | None],
    *,
    model: str,
) -> AudioInputTranscriptionOptions | None:
    """Convert audio transcription configuration to Azure AudioInputTranscriptionOptions format.

    Args:
        audio_transcription: Audio transcription options. If NOT_GIVEN, returns the default config
            of the model, azure-speech or whisper-1 (see `uses_azure_speech`). If None,
            transcription is disabled. Otherwise, returns the provided config.
        model: The Voice Live model of the session.

    Returns:
        AudioInputTranscriptionOptions or None if transcription is disabled.
    """
    if not is_given(audio_transcription):
        if uses_azure_speech(model):
            return AZURE_SPEECH_INPUT_AUDIO_TRANSCRIPTION
        return DEFAULT_INPUT_AUDIO_TRANSCRIPTION

    if audio_transcription is None:
        return None

    return audio_transcription


DEFAULT_TOOL_CHOICE: ToolChoiceLiteral | ToolChoiceSelection = ToolChoiceLiteral.AUTO


def to_azure_tool_choice(
    tool_choice: llm.ToolChoice | None,
) -> ToolChoiceLiteral | ToolChoiceSelection:
    """Convert a LiveKit ToolChoice to Azure's session-level tool_choice format."""
    if isinstance(tool_choice, str):
        return ToolChoiceLiteral(tool_choice)

    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return ToolChoiceFunctionSelection(name=tool_choice["function"]["name"])

    return DEFAULT_TOOL_CHOICE


def to_azure_response_tool_choice(tool_choice: llm.ToolChoice | None) -> str:
    """Convert a LiveKit ToolChoice to the tool_choice of a single response.

    The response-level field is a string: a mode (``auto``, ``none``, ``required``) or the name of
    the function the model must call.
    """
    if isinstance(tool_choice, str):
        return tool_choice

    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return tool_choice["function"]["name"]

    return ToolChoiceLiteral.AUTO.value


def livekit_tool_to_azure_tool(tool: llm.Tool) -> FunctionTool | None:
    """Convert LiveKit Tool to Azure FunctionTool format.

    Returns None for unsupported tool types (e.g. ProviderTool).
    """
    from livekit.agents.llm import utils as llm_utils

    if isinstance(tool, llm.FunctionTool):
        schema = llm_utils.build_legacy_openai_schema(tool, internally_tagged=True)
        return FunctionTool(
            name=schema["name"],
            description=schema.get("description", ""),
            parameters=schema.get("parameters", {}),
        )

    if isinstance(tool, llm.RawFunctionTool):
        raw_schema = tool.info.raw_schema
        return FunctionTool(
            name=tool.info.name,
            description=raw_schema.get("description", ""),
            parameters=raw_schema.get("parameters", {}),
        )

    logger.warning(
        "Azure Voice Live doesn't support this tool type, skipping it",
        extra={"tool_type": type(tool).__name__},
    )
    return None


def livekit_tools_to_azure_tools(tools: Sequence[llm.Tool]) -> list[Tool]:
    """Convert LiveKit tools to Azure function tools, skipping unsupported tool types."""
    azure_tools: list[Tool] = []
    for tool in tools:
        if (azure_tool := livekit_tool_to_azure_tool(tool)) is not None:
            azure_tools.append(azure_tool)
    return azure_tools


# Type alias for Azure conversation items
AzureConversationItem = (
    SystemMessageItem
    | UserMessageItem
    | AssistantMessageItem
    | FunctionCallItem
    | FunctionCallOutputItem
)

_CHAT_ROLES: dict[str, llm.ChatRole] = {
    "system": "system",
    "developer": "developer",
    "user": "user",
    "assistant": "assistant",
}


def livekit_item_to_azure_item(item: llm.ChatItem) -> AzureConversationItem:
    if item.type == "function_call_output":
        return FunctionCallOutputItem(call_id=item.call_id, output=item.output, id=item.id)

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


def azure_item_to_livekit_item(item: ResponseItem) -> llm.ChatItem:
    """Convert a conversation item created by Azure Voice Live to a LiveKit chat item."""
    if not item.id:
        raise ValueError("conversation item has no id")

    if isinstance(item, ResponseFunctionCallItem):
        return llm.FunctionCall(
            id=item.id,
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments or "",
        )

    if isinstance(item, ResponseFunctionCallOutputItem):
        return llm.FunctionCallOutput(
            id=item.id,
            call_id=item.call_id,
            output=item.output,
            is_error=False,
        )

    if isinstance(item, ResponseMessageItem):
        raw_role = item.role.value if isinstance(item.role, Enum) else item.role
        role = _CHAT_ROLES.get(raw_role)
        if role is None:
            raise ValueError(f"Unsupported role: {raw_role}")

        content: list[llm.ChatContent] = []
        for part in item.content or []:
            # text parts carry `text`, audio parts carry the `transcript` of the audio
            text = getattr(part, "text", None) or getattr(part, "transcript", None)
            if isinstance(text, str) and text:
                content.append(text)

        return llm.ChatMessage(id=item.id, role=role, content=content)

    raise ValueError(f"Unsupported item type: {item.type}")
