from __future__ import annotations

import base64
from typing import Any

from livekit import rtc
from livekit.agents import llm
from livekit.agents.types import (
    NotGivenOr,
)
from livekit.agents.utils import is_given
from openai.types import realtime, responses
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
from openai.types.realtime.realtime_audio_config_input import NoiseReduction

from ..log import logger

# default values got from a "default" session from their API
DEFAULT_TURN_DETECTION = realtime.realtime_audio_input_turn_detection.SemanticVad(
    type="semantic_vad",
    create_response=True,
    eagerness="medium",
    interrupt_response=True,
)
DEFAULT_TOOL_CHOICE: responses.ToolChoiceOptions = "auto"
DEFAULT_MAX_RESPONSE_OUTPUT_TOKENS = "inf"

DEFAULT_INPUT_AUDIO_TRANSCRIPTION = AudioTranscription(
    model="gpt-4o-mini-transcribe",
)

# use beta version TurnDetection and InputAudioTranscription for compatibility
AZURE_DEFAULT_TURN_DETECTION = TurnDetection(
    type="server_vad",
    threshold=0.5,
    prefix_padding_ms=300,
    silence_duration_ms=200,
    create_response=True,
)

AZURE_DEFAULT_INPUT_AUDIO_TRANSCRIPTION = InputAudioTranscription(
    model="whisper-1",
)

DEFAULT_MAX_SESSION_DURATION = 20 * 60  # 20 minutes


def to_noise_reduction(
    noise_reduction: NotGivenOr[
        InputAudioNoiseReduction | NoiseReduction | NoiseReductionType | None
    ],
) -> NoiseReduction | None:
    if not is_given(noise_reduction) or noise_reduction is None:
        return None
    if isinstance(noise_reduction, NoiseReduction):
        return noise_reduction
    if isinstance(noise_reduction, InputAudioNoiseReduction):
        return NoiseReduction(type=noise_reduction.type)
    return NoiseReduction(type=noise_reduction)


def to_audio_transcription(
    audio_transcription: NotGivenOr[InputAudioTranscription | AudioTranscription | None],
) -> AudioTranscription | None:
    if not is_given(audio_transcription):
        return DEFAULT_INPUT_AUDIO_TRANSCRIPTION

    if audio_transcription is None:
        return None

    if isinstance(audio_transcription, InputAudioTranscription):
        return AudioTranscription.model_construct(
            **audio_transcription.model_dump(
                by_alias=True, exclude_unset=True, exclude_defaults=True
            )
        )
    return audio_transcription


def to_turn_detection(
    turn_detection: NotGivenOr[RealtimeAudioInputTurnDetection | TurnDetection | None],
) -> RealtimeAudioInputTurnDetection | None:
    if not is_given(turn_detection):
        return DEFAULT_TURN_DETECTION

    if turn_detection is None:
        return None

    if isinstance(turn_detection, TurnDetection):
        kwargs: dict[str, Any] = {}
        if turn_detection.type == "server_vad":
            kwargs["type"] = "server_vad"
            if turn_detection.threshold is not None:
                kwargs["threshold"] = turn_detection.threshold
            if turn_detection.prefix_padding_ms is not None:
                kwargs["prefix_padding_ms"] = turn_detection.prefix_padding_ms
            if turn_detection.silence_duration_ms is not None:
                kwargs["silence_duration_ms"] = turn_detection.silence_duration_ms
            if turn_detection.create_response is not None:
                kwargs["create_response"] = turn_detection.create_response
            return realtime.realtime_audio_input_turn_detection.ServerVad(**kwargs)
        elif turn_detection.type == "semantic_vad":
            kwargs["type"] = "semantic_vad"
            if turn_detection.create_response is not None:
                kwargs["create_response"] = turn_detection.create_response
            if turn_detection.eagerness is not None:
                kwargs["eagerness"] = turn_detection.eagerness
            if turn_detection.interrupt_response is not None:
                kwargs["interrupt_response"] = turn_detection.interrupt_response
            return realtime.realtime_audio_input_turn_detection.SemanticVad(**kwargs)
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
            system_content: list[realtime.realtime_conversation_item_system_message.Content] = []
            for c in item.content:
                if isinstance(c, str):
                    system_content.append(
                        realtime.realtime_conversation_item_system_message.Content(
                            type="input_text",
                            text=c,
                        )
                    )
            conversation_item = realtime.RealtimeConversationItemSystemMessage(
                type="message",
                role="system",
                content=system_content,
            )
        elif item.role == "assistant":
            assistant_content: list[
                realtime.realtime_conversation_item_assistant_message.Content
            ] = []
            for c in item.content:
                if isinstance(c, str):
                    assistant_content.append(
                        realtime.realtime_conversation_item_assistant_message.Content(
                            type="output_text",
                            text=c,
                        )
                    )
            conversation_item = realtime.RealtimeConversationItemAssistantMessage(
                type="message",
                role="assistant",
                content=assistant_content,
            )
        elif item.role == "user":
            user_content: list[realtime.realtime_conversation_item_user_message.Content] = []
            # only user messages could be a list of content
            for c in item.content:
                if isinstance(c, str):
                    user_content.append(
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
                    assert img.data_bytes is not None
                    user_content.append(
                        realtime.realtime_conversation_item_user_message.Content(
                            type="input_image",
                            image_url=f"data:{img.mime_type};base64,{base64.b64encode(img.data_bytes).decode('utf-8')}",
                        )
                    )
                elif isinstance(c, llm.AudioContent):
                    encoded_audio = base64.b64encode(rtc.combine_audio_frames(c.frame).data).decode(
                        "utf-8"
                    )
                    user_content.append(
                        realtime.realtime_conversation_item_user_message.Content(
                            type="input_audio",
                            audio=encoded_audio,
                            transcript=c.transcript,
                        )
                    )
            conversation_item = realtime.RealtimeConversationItemUserMessage(
                type="message",
                role="user",
                content=user_content,
            )
        else:
            raise ValueError(f"unsupported role: {item.role}")

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
            for sc in item.content:
                if sc.text:
                    content.append(sc.text)
        elif isinstance(item, realtime.RealtimeConversationItemAssistantMessage):
            for ac in item.content:
                if ac.text:
                    content.append(ac.text)
        elif isinstance(item, realtime.RealtimeConversationItemUserMessage):
            for uc in item.content:
                if uc.type == "input_text" and uc.text is not None:
                    content.append(uc.text)
                elif uc.type == "input_image" and uc.image_url is not None:
                    content.append(llm.ImageContent(image=uc.image_url))
                elif uc.type == "input_audio" and uc.transcript is not None:
                    content.append(uc.transcript)
        return llm.ChatMessage(
            id=item.id,
            role=item.role,
            content=content,
        )

    raise ValueError(f"unsupported item type: {item.type}")


def to_oai_tool_choice(tool_choice: llm.ToolChoice | None) -> realtime.RealtimeToolChoiceConfig:
    if isinstance(tool_choice, str):
        return tool_choice

    elif isinstance(tool_choice, dict) and tool_choice["type"] == "function":
        return responses.ToolChoiceFunction(
            name=tool_choice["function"]["name"],
            type="function",
        )

    return DEFAULT_TOOL_CHOICE
