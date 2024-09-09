from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict


API_URL = "wss://api.openai.com/v1/realtime"
SAMPLE_RATE = 24000
NUM_CHANNELS = 1

IN_FRAME_SIZE = 2400  # 100ms
OUT_FRAME_SIZE = 1200  # 50ms

Voices = Literal["alloy", "echo", "shimmer"]


ClientRequest = Literal[
    "set_inference_config",
    "add_item",
    "delete_item",
    "add_user_audio",
    "commit_pending_audio",
    "client_turn_finished",
    "client_interrupted",
    "generate",
    "create_conversation",
    "delete_conversation",
    "subscribe_to_user_audio",
    "unsubscribe_from_user_audio",
    "truncate_content",
]

ServerResponse = Literal[
    "start_session",
    "error",
    "add_item",
    "add_content",
    "item_added",
    "turn_finished",
    "vad_speech_started",
    "vad_speech_stopped",
    "input_transcribed",
    "model_listening",
]


# Client Requests


class ClientMessage:
    class SetInferenceConfig(TypedDict):
        event: Literal["set_inference_config"]
        system_message: str
        turn_end_type: Literal["client_decision", "server_detection"]
        voice: Voices
        audio_format: Literal["pcm16", "g711-ulaw", "g711-alaw"]
        tools: dict | None
        tool_choice: Literal["auto", "none", "required"] | None
        temperature: float  # [0.6, 1.2]
        max_tokens: int  # [1, 4096]
        disable_audio: bool
        transcribe_input: bool

    class ItemContent(TypedDict):
        type: Literal["text", "audio"]
        text: str | None
        audio: str | None  # b64

    class AddItem(TypedDict):
        event: Literal["add_item"]
        type: Literal["message", "tool_response", "tool_call"]
        id: str
        previous_id: str
        conversation_label: str
        role: Literal["system", "assistant", "user"]
        tool_call_id: str | None
        content: str | list[ClientMessage.ItemContent]

    class DeleteItem(TypedDict):
        event: Literal["delete_item"]
        id: str

    class AddUserAudio(TypedDict):
        data: str

    class CommitPendingAudio(TypedDict):
        event: Literal["commit_pending_audio"]

    class ClientTurnFinished(TypedDict):
        event: Literal["client_turn_finished"]

    class ClientInterrupted(TypedDict):
        event: Literal["client_interrupted"]

    class Generate(TypedDict):
        event: Literal["generate"]
        conversation_label: str

    class CreateConversation(TypedDict):
        event: Literal["create_conversation"]
        label: str

    class DeleteConversation(TypedDict):
        event: Literal["delete_conversation"]
        label: str

    class SubscribeToUserAudio(TypedDict):
        event: Literal["subscribe_to_user_audio"]
        label: str

    class UnsubscribeFromUserAudio(TypedDict):
        event: Literal["unsubscribe_from_user_audio"]
        label: str

    class TruncateContent(TypedDict):
        event: Literal["truncate_content"]
        message_id: str
        index: int  # TODO(theomonnom): this is ignore?
        text_chars: int
        audio_samples: int


# Server Responses


class ServerMessage:
    class StartSession(TypedDict):
        event: Literal["start_session"]
        session_id: str
        model: str
        system_fingerprint: str

    class Error(TypedDict):
        event: Literal["error"]
        message: str

    class AddItem(TypedDict):
        event: Literal["add_item"]
        item: ClientMessage.AddItem
        id: str
        previous_id: str
        conversation_label: Literal["tool_call"]
        name: str

    class AddContent(TypedDict):
        event: Literal["add_content"]
        type: Literal["text", "audio", "tool_call_arguments"]
        item_id: str
        data: str

    class ItemAdded(TypedDict):
        event: Literal["item_added"]
        type: Literal["tool_call"]
        id: str
        previous_id: str
        conversation_label: str
        arguments: str
        tool_call_id: str

    class TurnFinished(TypedDict):
        event: Literal["turn_finished"]
        reason: Literal["stop", "max_tokens", "content_filter", "interrupt"]
        conversation_label: str
        item_ids: list[str]

    class VadSpeechStarted(TypedDict):
        event: Literal["vad_speech_started"]
        sample_index: int
        item_id: str

    class VadSpeechStopped(TypedDict):
        event: Literal["vad_speech_stopped"]
        sample_index: int
        item_id: str

    class InputTranscribed(TypedDict):
        type: Literal["input_transcribed"]
        item_id: str
        transcript: str

    class ModelListening(TypedDict):
        event: Literal["model_listening"]
