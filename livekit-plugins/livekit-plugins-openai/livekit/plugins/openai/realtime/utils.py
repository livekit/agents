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

# default values got from a "default" session from their API
DEFAULT_TURN_DETECTION = realtime.realtime_audio_input_turn_detection.SemanticVad(
    type="semantic_vad",
    create_response=True,
    eagerness="medium",
    interrupt_response=True,
)
DEFAULT_TOOL_CHOICE = "auto"
DEFAULT_MAX_RESPONSE_OUTPUT_TOKENS = "inf"

DEFAULT_INPUT_AUDIO_TRANSCRIPTION = AudioTranscription(
    model="gpt-4o-mini-transcribe",
)

AZURE_DEFAULT_TURN_DETECTION = realtime.realtime_audio_input_turn_detection.ServerVad(
    type="server_vad",
    threshold=0.5,
    prefix_padding_ms=300,
    silence_duration_ms=200,
    create_response=True,
)

AZURE_DEFAULT_INPUT_AUDIO_TRANSCRIPTION = AudioTranscription(
    model="whisper-1",
)

DEFAULT_MAX_SESSION_DURATION = 20 * 60  # 20 minutes


def to_noise_reduction(
    noise_reduction: NotGivenOr[InputAudioNoiseReduction | NoiseReductionType | None],
) -> NoiseReductionType | None:
    if not is_given(noise_reduction) or noise_reduction is None:
        return None
    if isinstance(noise_reduction, InputAudioNoiseReduction):
        return cast(NoiseReductionType, noise_reduction.type)
    return cast(NoiseReductionType, noise_reduction)


def to_audio_transcription(
    audio_transcription: NotGivenOr[InputAudioTranscription | AudioTranscription | None],
) -> AudioTranscription | None:
    if not is_given(audio_transcription):
        return DEFAULT_INPUT_AUDIO_TRANSCRIPTION

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
    turn_detection: NotGivenOr[RealtimeAudioInputTurnDetection | TurnDetection | None],
) -> RealtimeAudioInputTurnDetection | None:
    if not is_given(turn_detection):
        return DEFAULT_TURN_DETECTION

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


def to_oai_tool_choice(tool_choice: llm.ToolChoice | None) -> str:
    if isinstance(tool_choice, str):
        return tool_choice

    elif isinstance(tool_choice, dict) and tool_choice["type"] == "function":
        return tool_choice["function"]["name"]

    return DEFAULT_TOOL_CHOICE
