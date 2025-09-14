from __future__ import annotations

import base64
from typing import cast

from livekit import rtc
from livekit.agents import llm
from livekit.agents.types import (
    NotGivenOr,
)
from livekit.agents.utils import is_given
from openai.types import realtime
from openai.types.beta.realtime.session import (
    InputAudioNoiseReduction,
    InputAudioTranscription,
    TurnDetection,
)
from openai.types.realtime import (
    AudioTranscription,
    NoiseReductionType,
    RealtimeAudioInputTurnDetection,
)

from ..log import logger


def to_noise_reduction(
    noise_reduction: NotGivenOr[InputAudioNoiseReduction | NoiseReductionType | None],
) -> NoiseReductionType | None:
    if not is_given(noise_reduction):
        return None
    if noise_reduction is None:
        return None
    if isinstance(noise_reduction, InputAudioNoiseReduction):
        return cast(NoiseReductionType, noise_reduction.type)
    return noise_reduction


def to_audio_transcription(
    audio_transcription: InputAudioTranscription | AudioTranscription | None,
) -> AudioTranscription | None:
    if audio_transcription is None:
        return None
    if isinstance(audio_transcription, InputAudioTranscription):
        return AudioTranscription(
            model=audio_transcription.model,
            prompt=audio_transcription.prompt,
            language=audio_transcription.language,
        )
    return audio_transcription


def to_turn_detection(
    turn_detection: RealtimeAudioInputTurnDetection | TurnDetection | None,
) -> RealtimeAudioInputTurnDetection | None:
    if turn_detection is None:
        return None
    if isinstance(turn_detection, TurnDetection):
        if turn_detection.type == "server_vad":
            return realtime.realtime_audio_input_turn_detection.ServerVad(
                type="server_vad",
                threshold=turn_detection.threshold,
                prefix_padding_ms=turn_detection.prefix_padding_ms,
                silence_duration_ms=turn_detection.silence_duration_ms,
                create_response=turn_detection.create_response,
            )
        elif turn_detection.type == "semantic_vad":
            return realtime.realtime_audio_input_turn_detection.SemanticVad(
                type="semantic_vad",
                create_response=turn_detection.create_response,
                eagerness=turn_detection.eagerness,
                interrupt_response=turn_detection.interrupt_response,
            )
        else:
            raise ValueError(f"unsupported turn detection type: {turn_detection.type}")
    return turn_detection


def livekit_item_to_openai_item(item: llm.ChatItem) -> realtime.ConversationItem:
    conversation_item: realtime.ConversationItem

    if item.type == "function_call":
        conversation_item = realtime.RealtimeConversationItemFunctionCall(
            id=item.id,
            type="function_call",
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments,
        )

    elif item.type == "function_call_output":
        conversation_item = realtime.RealtimeConversationItemFunctionCallOutput(
            id=item.id,
            type="function_call_output",
            call_id=item.call_id,
            output=item.output,
        )
        conversation_item.type = "function_call_output"
        conversation_item.call_id = item.call_id
        conversation_item.output = item.output

    elif item.type == "message":
        if item.role == "system" or item.role == "developer":
            conversation_item = realtime.RealtimeConversationItemSystemMessage(
                role="system",
            )
            content_list: list[realtime.realtime_conversation_item_system_message.Content] = []
            for c in item.content:
                if isinstance(c, str):
                    content_list.append(
                        realtime.realtime_conversation_item_system_message.Content(
                            type="input_text",
                            text=c,
                        )
                    )
            conversation_item.content = content_list
        elif item.role == "assistant":
            conversation_item = realtime.RealtimeConversationItemAssistantMessage(
                role="assistant",
            )
            content_list: list[realtime.realtime_conversation_item_assistant_message.Content] = []
            for c in item.content:
                if isinstance(c, str):
                    content_list.append(
                        realtime.realtime_conversation_item_assistant_message.Content(
                            type="output_text",
                            text=c,
                        )
                    )
            conversation_item.content = content_list
        elif item.role == "user":
            conversation_item = realtime.RealtimeConversationItemUserMessage(
                role="user",
            )
            content_list: list[realtime.realtime_conversation_item_user_message.Content] = []
            # only user messages could be a list of content
            for c in item.content:
                if isinstance(c, str):
                    content_list.append(
                        realtime.realtime_conversation_item_user_message.Content(
                            type="input_text",
                            text=c,
                        )
                    )
                elif isinstance(c, llm.ImageContent):
                    img = llm.utils.serialize_image(c)
                    if img.external_url:
                        logger.warning("External URL is not supported for input_image")
                        continue
                    content_list.append(
                        realtime.realtime_conversation_item_user_message.Content(
                            type="input_image",
                            image_url=f"data:{img.mime_type};base64,{base64.b64encode(img.data_bytes).decode('utf-8')}",
                        )
                    )
                elif isinstance(c, llm.AudioContent):
                    encoded_audio = base64.b64encode(rtc.combine_audio_frames(c.frame).data).decode(
                        "utf-8"
                    )
                    content_list.append(
                        realtime.realtime_conversation_item_user_message.Content(
                            type="input_audio",
                            audio=encoded_audio,
                            transcript=c.transcript,
                        )
                    )
            conversation_item.content = content_list
        else:
            raise ValueError(f"unsupported role: {item.role}")

        conversation_item.type = "message"

    conversation_item.id = item.id
    return conversation_item


def openai_item_to_livekit_item(item: realtime.ConversationItem) -> llm.ChatItem:
    assert item.id is not None, "id is None"

    if item.type == "function_call":
        assert item.call_id is not None, "call_id is None"
        assert item.name is not None, "name is None"
        assert item.arguments is not None, "arguments is None"

        return llm.FunctionCall(
            id=item.id,
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments,
        )

    if item.type == "function_call_output":
        assert item.call_id is not None, "call_id is None"
        assert item.output is not None, "output is None"

        return llm.FunctionCallOutput(
            id=item.id,
            call_id=item.call_id,
            output=item.output,
            is_error=False,
        )

    if item.type == "message":
        assert item.role is not None, "role is None"
        assert item.content is not None, "content is None"

        content: list[llm.ChatContent] = []
        if isinstance(item, realtime.RealtimeConversationItemSystemMessage):
            for c in item.content:
                if c.text:
                    content.append(c.text)
        elif isinstance(item, realtime.RealtimeConversationItemAssistantMessage):
            for c in item.content:
                if c.text:
                    content.append(c.text)
        elif isinstance(item, realtime.RealtimeConversationItemUserMessage):
            for c in item.content:
                if c.type == "input_text":
                    content.append(c.text)
                # intentially ignore image and audio output
                # this function is used to convert changes to previous chat context

        return llm.ChatMessage(
            id=item.id,
            role=item.role,
            content=content,
        )

    raise ValueError(f"unsupported item type: {item.type}")


def to_oai_tool_choice(tool_choice: llm.ToolChoice | None) -> str:
    if isinstance(tool_choice, str):
        return tool_choice

    elif isinstance(tool_choice, dict) and tool_choice["type"] == "function":
        return tool_choice["function"]["name"]

    return "auto"
