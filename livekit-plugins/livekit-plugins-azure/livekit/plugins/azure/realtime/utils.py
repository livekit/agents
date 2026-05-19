from __future__ import annotations

import base64
from typing import Any

from azure.ai.voicelive.models import (
    AudioInputTranscriptionOptions,
    Modality,
    ServerVad,
    TurnDetection,
)
from livekit import rtc
from livekit.agents import llm
from livekit.agents.types import NotGivenOr
from livekit.agents.utils import is_given

from ..log import logger

# Default configurations for Azure Voice Live
DEFAULT_INPUT_AUDIO_TRANSCRIPTION = AudioInputTranscriptionOptions(
    model="gpt-4o-mini-transcribe",
)

DEFAULT_TURN_DETECTION = ServerVad(
    threshold=0.5,
    prefix_padding_ms=300,
    silence_duration_ms=500,
    create_response=True,
)

DEFAULT_MODALITIES: list[str] = [Modality.TEXT, Modality.AUDIO]
DEFAULT_INPUT_AUDIO_FORMAT = "pcm16"
DEFAULT_OUTPUT_AUDIO_FORMAT = "pcm16"
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_API_VERSION = "2025-10-01"

_OPENAI_VOICES = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "sage",
        "shimmer",
        "verse",
        "marin",
        "cedar",
    }
)


def to_turn_detection(
    turn_detection: NotGivenOr[TurnDetection | None],
) -> TurnDetection | None:
    """Resolve the turn-detection configuration, applying defaults.

    Accepts any TurnDetection subclass (ServerVad, AzureSemanticVad,
    AzureSemanticVadEn, AzureSemanticVadMultilingual). Returns None to
    disable server-side turn detection.
    """
    if not is_given(turn_detection):
        return DEFAULT_TURN_DETECTION

    if turn_detection is None:
        return None

    return turn_detection


def to_audio_transcription(
    audio_transcription: NotGivenOr[AudioInputTranscriptionOptions | None],
) -> AudioInputTranscriptionOptions | None:
    """Resolve input audio transcription options, applying defaults.

    NOT_GIVEN -> default whisper-1; None -> transcription disabled.
    """
    if not is_given(audio_transcription):
        return DEFAULT_INPUT_AUDIO_TRANSCRIPTION

    if audio_transcription is None:
        return None

    return audio_transcription


def to_azure_tool_choice(tool_choice: llm.ToolChoice | None) -> str | dict[str, Any]:
    """Convert a LiveKit ToolChoice into the wire format Azure expects."""
    if isinstance(tool_choice, str):
        return tool_choice

    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {"type": "function", "name": tool_choice["function"]["name"]}

    return "auto"


def build_voice_config(voice: Any, language: str | None) -> str | dict[str, Any]:
    """Convert a voice option into the wire format.

    A bare string that looks like an Azure neural voice (contains a hyphen and
    is not a known OpenAI voice) is wrapped in an azure-standard voice object.
    SDK voice models are serialized via as_dict().
    """
    if isinstance(voice, str):
        if "-" in voice and voice not in _OPENAI_VOICES:
            cfg: dict[str, Any] = {"name": voice, "type": "azure-standard"}
            if language:
                cfg["locale"] = language
            return cfg
        return voice

    if hasattr(voice, "as_dict"):
        return voice.as_dict()  # type: ignore[no-any-return]

    raise TypeError(f"Unsupported voice configuration: {type(voice)!r}")


def livekit_tool_to_azure_tool(tool: llm.Tool) -> dict[str, Any] | None:
    """Convert a LiveKit Tool to the Azure Voice Live wire dict for a tool.

    Returns None for unsupported tool types (e.g. ProviderTool).
    """
    from livekit.agents.llm import utils as llm_utils

    if isinstance(tool, llm.FunctionTool):
        schema = llm_utils.build_legacy_openai_schema(tool, internally_tagged=True)
        return {
            "type": "function",
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema.get("parameters", {}),
        }
    if isinstance(tool, llm.RawFunctionTool):
        raw_schema = tool.info.raw_schema
        return {
            "type": "function",
            "name": tool.info.name,
            "description": raw_schema.get("description", ""),
            "parameters": raw_schema.get("parameters", {}),
        }

    logger.warning(f"[TOOL_CONVERSION] Skipping unsupported tool type: {type(tool)}")
    return None


def livekit_item_to_azure_item(item: llm.ChatItem) -> dict[str, Any]:
    """Convert a LiveKit ChatItem to the Azure Voice Live wire dict for a conversation item."""
    if item.type == "function_call_output":
        return {
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": item.output,
        }

    if item.type == "function_call":
        return {
            "type": "function_call",
            "id": item.id,
            "call_id": item.call_id,
            "name": item.name,
            "arguments": item.arguments,
        }

    if item.type == "message":
        if item.role in ("system", "developer"):
            content = [
                {"type": "input_text", "text": c} for c in item.content if isinstance(c, str)
            ]
            return {"type": "message", "role": "system", "id": item.id, "content": content}

        if item.role == "assistant":
            content = [{"type": "text", "text": c} for c in item.content if isinstance(c, str)]
            return {
                "type": "message",
                "role": "assistant",
                "id": item.id,
                "content": content,
            }

        if item.role == "user":
            user_content: list[dict[str, Any]] = []
            for c in item.content:
                if isinstance(c, str):
                    user_content.append({"type": "input_text", "text": c})
                elif isinstance(c, llm.AudioContent):
                    encoded_audio = base64.b64encode(rtc.combine_audio_frames(c.frame).data).decode(
                        "utf-8"
                    )
                    user_content.append(
                        {
                            "type": "input_audio",
                            "audio": encoded_audio,
                            "transcript": c.transcript,
                        }
                    )
            return {"type": "message", "role": "user", "id": item.id, "content": user_content}

        raise ValueError(f"Unsupported role: {item.role}")

    raise ValueError(f"Unsupported item type: {item.type}")
