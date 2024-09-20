from __future__ import annotations

from typing import Literal, Union
from typing_extensions import TypedDict, NotRequired

API_URL = "wss://api.openai.com/v1/realtime"

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

IN_FRAME_SIZE = 2400  # 100ms
OUT_FRAME_SIZE = 1200  # 50ms


TurnDetectionType = Literal["disabled", "server_vad"]
Voices = Literal["alloy", "echo", "shimmer"]
ToolChoice = Literal["auto", "none", "required"]
Role = Literal["system", "assistant", "user", "tool"]
GenerationFinishedReason = Literal["stop", "max_tokens", "content_filter", "interrupt"]


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class AudioContent(TypedDict):
    type: Literal["audio"]
    audio: str  # b64


class ToolCallContent(TypedDict):
    type: Literal["tool_call"]
    name: str
    arguments: str  # json
    tool_call_id: str


MessageContent = Union[TextContent, AudioContent, ToolCallContent]


class Message(TypedDict):
    id: NotRequired[str]
    role: Role
    tool_call_id: NotRequired[str]
    content: list[MessageContent]


class VadConfig(TypedDict, total=False):
    threshold: float
    prefix_padding_ms: float
    silence_duration_ms: float


class ClientEvent:
    class UpdateSessionConfig(TypedDict):
        event: Literal["update_session_config"]
        turn_detection: TurnDetectionType
        input_audio_format: Literal["pcm16", "g711-ulaw", "g711-alaw"]
        transcribe_input: bool
        vad: NotRequired[VadConfig]

    class UpdateConversationConfig(TypedDict):
        event: Literal["update_conversation_config"]
        system_message: str
        voice: Voices
        subscribe_to_user_audio: bool
        output_audio_format: Literal["pcm16", "g711-ulaw", "g711-alaw"]
        tools: NotRequired[list]
        tool_choice: NotRequired[ToolChoice]
        temperature: float  # [0.6, 1.2]
        max_tokens: int  # [1, 4096]
        disable_audio: bool
        conversation_label: NotRequired[str]

    class AddMessage(TypedDict):
        event: Literal["add_message"]
        previous_id: NotRequired[str]
        conversation_label: NotRequired[str]
        message: list[Message]

    class DeleteMessage(TypedDict):
        event: Literal["delete_message"]
        id: str
        conversation_label: NotRequired[str]

    class AddUserAudio(TypedDict):
        event: Literal["add_user_audio"]
        data: str  # b64

    class CommitUserAudio(TypedDict):
        event: Literal["commit_user_audio"]

    class Generate(TypedDict):
        event: Literal["generate"]
        conversation_label: NotRequired[str]

    class CancelGeneration(TypedDict):
        event: Literal["cancel_generation"]
        conversation_label: NotRequired[str]

    class CreateConversation(TypedDict):
        event: Literal["create_conversation"]
        label: str

    class DeleteConversation(TypedDict):
        event: Literal["delete_conversation"]
        label: str

    class TruncateContent(TypedDict):
        event: Literal["truncate_content"]
        message_id: str
        index: int  # TODO(theomonnom): this is ignore?
        text_chars: int
        audio_samples: int


class ServerEvent:
    class StartSession(TypedDict):
        event: Literal["start_session"]
        session_id: str
        model: str
        system_fingerprint: str

    class Error(TypedDict):
        event: Literal["error"]
        message: str

    class AddMessage(TypedDict):
        event: Literal["add_message"]
        previous_id: str
        conversation_label: str
        message: Message

    class AddContent(TypedDict):
        event: Literal["add_content"]
        type: Literal["text", "audio", "tool_call"]
        message_id: str
        data: str

    class MessageAdded(TypedDict):
        event: Literal["message_added"]
        id: str
        previous_id: str
        conversation_label: str
        content: list[MessageContent]

    class GenerationFinished(TypedDict):
        event: Literal["generation_finished"]
        reason: GenerationFinishedReason
        conversation_label: str
        message_ids: list[str]

    class GenerationCanceled(TypedDict):
        event: Literal["generation_canceled"]
        conversation_label: str

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


ClientEventType = Literal[
    "update_session_config",
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

ServerEventType = Literal[
    "start_session",
    "error",
    "add_message",
    "add_content",
    "message_added",
    "generation_finished",
    "generation_canceled",
    "vad_speech_started",
    "vad_speech_stopped",
    "input_transcribed",
]

ClientEvents = Union[
    ClientEvent.UpdateSessionConfig,
    ClientEvent.UpdateConversationConfig,
    ClientEvent.AddMessage,
    ClientEvent.DeleteMessage,
    ClientEvent.AddUserAudio,
    ClientEvent.CommitUserAudio,
    ClientEvent.Generate,
    ClientEvent.CancelGeneration,
    ClientEvent.CreateConversation,
    ClientEvent.DeleteConversation,
    ClientEvent.TruncateContent,
]

ServerEvents = Union[
    ServerEvent.StartSession,
    ServerEvent.Error,
    ServerEvent.AddMessage,
    ServerEvent.AddContent,
    ServerEvent.MessageAdded,
    ServerEvent.GenerationFinished,
    ServerEvent.GenerationCanceled,
    ServerEvent.VadSpeechStarted,
    ServerEvent.VadSpeechStopped,
    ServerEvent.InputTranscribed,
]
