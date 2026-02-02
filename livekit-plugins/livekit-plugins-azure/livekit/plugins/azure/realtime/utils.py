from __future__ import annotations

import base64
from typing import Any

from azure.ai.voicelive.models import (
    FunctionCallOutputItem,
    FunctionTool,
    InputAudioFormat,
    Modality,
    OutputAudioFormat,
    ServerVad,
)
from livekit import rtc
from livekit.agents import llm
from livekit.agents.types import NotGivenOr
from livekit.agents.utils import is_given

from ..log import logger

# Default configurations for Azure Voice Live
DEFAULT_TURN_DETECTION = ServerVad(
    type="server_vad",
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
    turn_detection: NotGivenOr[ServerVad | None],
) -> ServerVad | None:
    """Convert turn detection configuration to Azure ServerVad format."""
    if not is_given(turn_detection):
        return DEFAULT_TURN_DETECTION

    if turn_detection is None:
        return None

    return turn_detection


def livekit_tool_to_azure_tool(tool: llm.Tool) -> FunctionTool:
    """Convert LiveKit Tool to Azure FunctionTool format."""
    from livekit.agents.llm import utils as llm_utils

    # Handle FunctionTool and RawFunctionTool
    if isinstance(tool, (llm.FunctionTool, llm.RawFunctionTool)):
        # Use the build_legacy_openai_schema to get the schema
        schema = llm_utils.build_legacy_openai_schema(tool, internally_tagged=True)

        parameters = schema.get("parameters", {})

        azure_tool = FunctionTool(
            type="function",
            name=schema["name"],
            description=schema.get("description", ""),
            parameters=parameters,
        )

        logger.info(f"[TOOL_CONVERSION] Converted tool {schema['name']}")
        logger.debug(f"[TOOL_CONVERSION] Schema: {schema}")
        logger.debug(f"[TOOL_CONVERSION] Azure tool parameters: {parameters}")

        return azure_tool
    else:
        raise ValueError(f"Unsupported tool type: {type(tool)}")


def livekit_item_to_azure_item(item: llm.ChatItem) -> dict[str, Any] | FunctionCallOutputItem:
    """Convert LiveKit ChatItem to Azure conversation item format."""

    # FunctionCallOutput needs to use FunctionCallOutputItem class
    if item.type == "function_call_output":
        return FunctionCallOutputItem(
            call_id=item.call_id,
            output=item.output,
        )

    # Other items use dict format
    azure_item: dict[str, Any] = {"id": item.id}

    if item.type == "function_call":
        azure_item.update(
            {
                "type": "function_call",
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
            }
        )

    elif item.type == "message":
        content_list = []

        if item.role in ("system", "developer"):
            for c in item.content:
                if isinstance(c, str):
                    content_list.append(
                        {
                            "type": "input_text",
                            "text": c,
                        }
                    )
            azure_item.update(
                {
                    "type": "message",
                    "role": "system",
                    "content": content_list,
                }
            )

        elif item.role == "assistant":
            for c in item.content:
                if isinstance(c, str):
                    content_list.append(
                        {
                            "type": "text",
                            "text": c,
                        }
                    )
            azure_item.update(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": content_list,
                }
            )

        elif item.role == "user":
            for c in item.content:
                if isinstance(c, str):
                    content_list.append(
                        {
                            "type": "input_text",
                            "text": c,
                        }
                    )
                elif isinstance(c, llm.AudioContent):
                    encoded_audio = base64.b64encode(rtc.combine_audio_frames(c.frame).data).decode(
                        "utf-8"
                    )
                    content_list.append(
                        {
                            "type": "input_audio",
                            "audio": encoded_audio,
                            "transcript": c.transcript,
                        }
                    )
            azure_item.update(
                {
                    "type": "message",
                    "role": "user",
                    "content": content_list,
                }
            )
        else:
            raise ValueError(f"Unsupported role: {item.role}")

    else:
        raise ValueError(f"Unsupported item type: {item.type}")

    return azure_item


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
