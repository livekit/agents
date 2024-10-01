from __future__ import annotations

from typing import Literal, Union

from typing_extensions import NotRequired, TypedDict

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

IN_FRAME_SIZE = 2400  # 100ms
OUT_FRAME_SIZE = 1200  # 50ms


class FunctionToolChoice(TypedDict):
    type: Literal["function"]
    name: str


Voice = Literal["alloy", "echo", "shimmer"]
ToolChoice = Union[Literal["auto", "none", "required"], FunctionToolChoice]
Role = Literal["system", "assistant", "user", "tool"]
GenerationFinishedReason = Literal["stop", "max_tokens", "content_filter", "interrupt"]
AudioFormat = Literal["pcm16", "g711-ulaw", "g711-alaw"]
InputTranscriptionModel = Literal["whisper-1"]
Modality = Literal["text", "audio"]
ResponseStatus = Literal[
    "in_progress", "completed", "incomplete", "cancelled", "failed"
]


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class InputTextContent(TypedDict):
    type: Literal["input_text"]
    text: str


class AudioContent(TypedDict):
    type: Literal["audio"]
    audio: str  # b64


class InputAudioContent(TypedDict):
    type: Literal["input_audio"]
    audio: str  # b64


Content = Union[InputTextContent, TextContent, AudioContent, InputAudioContent]


class ContentPart(TypedDict):
    type: Literal["text", "audio"]
    audio: NotRequired[str]  # b64
    transcript: NotRequired[str]


class InputAudioTranscription(TypedDict):
    model: InputTranscriptionModel | str


class ServerVad(TypedDict):
    type: Literal["server_vad"]
    threshold: NotRequired[float]
    prefix_padding_ms: NotRequired[int]
    silence_duration_ms: NotRequired[int]


class FunctionTool(TypedDict):
    type: Literal["function"]
    name: str
    description: NotRequired[str | None]
    parameters: dict


class SystemItem(TypedDict):
    id: str
    object: Literal["realtime.item"]
    type: Literal["message"]
    role: Literal["system"]
    content: list[InputTextContent]


class UserItem(TypedDict):
    id: str
    object: Literal["realtime.item"]
    type: Literal["message"]
    role: Literal["user"]
    content: list[InputTextContent | InputAudioContent]


class AssistantItem(TypedDict):
    id: str
    object: Literal["realtime.item"]
    type: Literal["message"]
    role: Literal["assistant"]
    content: list[TextContent | AudioContent]


class FunctionCallItem(TypedDict):
    id: str
    object: Literal["realtime.item"]
    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str


class FunctionCallOutputItem(TypedDict):
    id: str
    object: Literal["realtime.item"]
    type: Literal["function_call_output"]
    call_id: str
    output: str


class CancelledStatusDetails(TypedDict):
    type: Literal["cancelled"]
    reason: Literal["turn_detected", "client_cancelled"]


class IncompleteStatusDetails(TypedDict):
    type: Literal["incomplete"]
    reason: Literal["max_output_tokens", "content_filter"]


class Error(TypedDict):
    code: str
    message: str


class FailedStatusDetails(TypedDict):
    type: Literal["failed"]
    error: NotRequired[Error | None]


ResponseStatusDetails = Union[
    CancelledStatusDetails, IncompleteStatusDetails, FailedStatusDetails
]


class Usage(TypedDict):
    total_tokens: int
    input_tokens: int
    output_tokens: int


class Resource:
    class Session(TypedDict):
        id: str
        object: Literal["realtime.session"]
        expires_at: int
        model: str
        modalities: list[Literal["text", "audio"]]
        instructions: str
        voice: Voice
        input_audio_format: AudioFormat
        output_audio_format: AudioFormat
        input_audio_transcription: InputAudioTranscription | None
        turn_detection: ServerVad | None
        tools: list[FunctionTool]
        tool_choice: ToolChoice
        temperature: float
        max_response_output_tokens: int | Literal["inf"]

    class Conversation(TypedDict):
        id: str
        object: Literal["realtime.conversation"]

    Item = Union[SystemItem, UserItem, FunctionCallItem, FunctionCallOutputItem]

    class Response(TypedDict):
        id: str
        object: Literal["realtime.response"]
        status: ResponseStatus
        status_details: NotRequired[ResponseStatusDetails | None]
        output: list[Resource.Item]
        usage: NotRequired[Usage | None]


class ClientEvent:
    class SessionUpdateData(TypedDict):
        modalities: list[Literal["text", "audio"]]
        instructions: str
        voice: Voice
        input_audio_format: AudioFormat
        output_audio_format: AudioFormat
        input_audio_transcription: InputAudioTranscription | None
        turn_detection: ServerVad | None
        tools: list[FunctionTool]
        tool_choice: ToolChoice
        temperature: float
        max_response_output_tokens: int | Literal["inf"]

    class SessionUpdate(TypedDict):
        event_id: NotRequired[str]
        type: Literal["session.update"]
        session: ClientEvent.SessionUpdateData

    class InputAudioBufferAppend(TypedDict):
        event_id: NotRequired[str]
        type: Literal["input_audio_buffer.append"]
        audio: str  # b64

    class InputAudioBufferCommit(TypedDict):
        event_id: NotRequired[str]
        type: Literal["input_audio_buffer.commit"]

    class InputAudioBufferClear(TypedDict):
        event_id: NotRequired[str]
        type: Literal["input_audio_buffer.clear"]

    class UserItemCreate(TypedDict):
        type: Literal["message"]
        role: Literal["user"]
        content: list[InputTextContent | InputAudioContent]

    class AssistantItemCreate(TypedDict):
        type: Literal["message"]
        role: Literal["assistant"]
        content: list[TextContent]

    class SystemItemCreate(TypedDict):
        type: Literal["message"]
        role: Literal["system"]
        content: list[InputTextContent]

    class FunctionCallOutputItemCreate(TypedDict):
        type: Literal["function_call_output"]
        call_id: str
        output: str

    ConversationItemCreateContent = Union[
        UserItemCreate,
        AssistantItemCreate,
        SystemItemCreate,
        FunctionCallOutputItemCreate,
    ]

    class ConversationItemCreate(TypedDict):
        event_id: NotRequired[str]
        type: Literal["conversation.item.create"]
        previous_item_id: NotRequired[str | None]
        item: ClientEvent.ConversationItemCreateContent

    class ConversationItemTruncate(TypedDict):
        event_id: NotRequired[str]
        type: Literal["conversation.item.truncate"]
        item_id: str
        content_index: int
        audio_end_ms: int

    class ConversationItemDelete(TypedDict):
        event_id: NotRequired[str]
        type: Literal["conversation.item.delete"]
        item_id: str

    class ResponseCreateData(TypedDict, total=False):
        modalities: list[Literal["text", "audio"]]
        instructions: str
        voice: Voice
        output_audio_format: AudioFormat
        tools: list[FunctionTool]
        tool_choice: ToolChoice
        temperature: float
        max_output_tokens: int | Literal["inf"]

    class ResponseCreate(TypedDict):
        event_id: NotRequired[str]
        type: Literal["response.create"]
        response: NotRequired[ClientEvent.ResponseCreateData]

    class ResponseCancel(TypedDict):
        event_id: NotRequired[str]
        type: Literal["response.cancel"]


class ServerEvent:
    class ErrorContent(TypedDict):
        type: str
        code: NotRequired[str]
        message: str
        param: NotRequired[str]
        event_id: NotRequired[str]

    class Error(TypedDict):
        event_id: str
        type: Literal["error"]
        error: ServerEvent.ErrorContent

    class SessionCreated(TypedDict):
        event_id: str
        type: Literal["session.created"]
        session: Resource.Session

    class SessionUpdated(TypedDict):
        event_id: str
        type: Literal["session.updated"]
        session: Resource.Session

    class ConversationCreated(TypedDict):
        event_id: str
        type: Literal["conversation.created"]
        conversation: Resource.Conversation

    class InputAudioBufferCommitted(TypedDict):
        event_id: str
        type: Literal["input_audio_buffer.committed"]
        item_id: str

    class InputAudioBufferCleared(TypedDict):
        event_id: str
        type: Literal["input_audio_buffer.cleared"]

    class InputAudioBufferSpeechStarted(TypedDict):
        event_id: str
        type: Literal["input_audio_buffer.speech_started"]
        item_id: str
        audio_start_ms: int

    class InputAudioBufferSpeechStopped(TypedDict):
        event_id: str
        type: Literal["input_audio_buffer.speech_stopped"]
        item_id: str
        audio_end_ms: int

    class ConversationItemCreated(TypedDict):
        event_id: str
        type: Literal["conversation.item.created"]
        item: Resource.Item

    class ConversationItemInputAudioTranscriptionCompleted(TypedDict):
        event_id: str
        type: Literal["conversation.item.input_audio_transcription.completed"]
        item_id: str
        content_index: int
        transcript: str

    class InputAudioTranscriptionError(TypedDict):
        type: str
        code: NotRequired[str]
        message: str
        param: NotRequired[str]

    class ConversationItemInputAudioTranscriptionFailed(TypedDict):
        event_id: str
        type: Literal["conversation.item.input_audio_transcription.failed"]
        item_id: str
        content_index: int
        error: ServerEvent.InputAudioTranscriptionError

    class ConversationItemTruncated(TypedDict):
        event_id: str
        type: Literal["conversation.item.truncated"]
        item_id: str
        content_index: int
        audio_end_ms: int

    class ConversationItemDeleted(TypedDict):
        event_id: str
        type: Literal["conversation.item.deleted"]
        item_id: str

    class ResponseCreated(TypedDict):
        event_id: str
        type: Literal["response.created"]
        response: Resource.Response

    class ResponseDone(TypedDict):
        event_id: str
        type: Literal["response.done"]
        response: Resource.Response

    class ResponseOutputItemAdded(TypedDict):
        event_id: str
        type: Literal["response.output_item.added"]
        response_id: str
        output_index: int
        item: Resource.Item

    class ResponseOutputItemDone(TypedDict):
        event_id: str
        type: Literal["response.output.done"]
        response_id: str
        output_index: int
        item: Resource.Item

    class ResponseContentPartAdded(TypedDict):
        event_id: str
        type: Literal["response.content_part.added"]
        item_id: str
        response_id: str
        output_index: int
        content_index: int
        part: ContentPart

    class ResponseContentPartDone(TypedDict):
        event_id: str
        type: Literal["response.content.done"]
        response_id: str
        output_index: int
        content_index: int
        part: ContentPart

    class ResponseTextDeltaAdded(TypedDict):
        event_id: str
        type: Literal["response.text.delta"]
        response_id: str
        output_index: int
        content_index: int
        delta: str

    class ResponseTextDone(TypedDict):
        event_id: str
        type: Literal["response.text.done"]
        response_id: str
        output_index: int
        content_index: int
        text: str

    class ResponseAudioTranscriptDelta(TypedDict):
        event_id: str
        type: Literal["response.audio_transcript.delta"]
        response_id: str
        output_index: int
        content_index: int
        delta: str

    class ResponseAudioTranscriptDone(TypedDict):
        event_id: str
        type: Literal["response.audio_transcript.done"]
        response_id: str
        output_index: int
        content_index: int
        transcript: str

    class ResponseAudioDelta(TypedDict):
        event_id: str
        type: Literal["response.audio.delta"]
        response_id: str
        output_index: int
        content_index: int
        delta: str  # b64

    class ResponseAudioDone(TypedDict):
        event_id: str
        type: Literal["response.audio.done"]
        response_id: str
        output_index: int
        content_index: int

    class ResponseFunctionCallArgumentsDelta(TypedDict):
        event_id: str
        type: Literal["response.function_call_arguments.delta"]
        response_id: str
        output_index: int
        delta: str

    class ResponseFunctionCallArgumentsDone(TypedDict):
        event_id: str
        type: Literal["response.function_call_arguments.done"]
        response_id: str
        output_index: int
        arguments: str

    class RateLimitsData(TypedDict):
        name: Literal["requests", "tokens", "input_tokens", "output_tokens"]
        limit: int
        remaining: int
        reset_seconds: float

    class RateLimitsUpdated:
        event_id: str
        type: Literal["rate_limits.updated"]
        limits: list[ServerEvent.RateLimitsData]


ClientEvents = Union[
    ClientEvent.SessionUpdate,
    ClientEvent.InputAudioBufferAppend,
    ClientEvent.InputAudioBufferCommit,
    ClientEvent.InputAudioBufferClear,
    ClientEvent.ConversationItemCreate,
    ClientEvent.ConversationItemTruncate,
    ClientEvent.ConversationItemDelete,
    ClientEvent.ResponseCreate,
    ClientEvent.ResponseCancel,
]

ServerEvents = Union[
    ServerEvent.Error,
    ServerEvent.SessionCreated,
    ServerEvent.SessionUpdated,
    ServerEvent.ConversationCreated,
    ServerEvent.InputAudioBufferCommitted,
    ServerEvent.InputAudioBufferCleared,
    ServerEvent.InputAudioBufferSpeechStarted,
    ServerEvent.InputAudioBufferSpeechStopped,
    ServerEvent.ConversationItemCreated,
    ServerEvent.ConversationItemInputAudioTranscriptionCompleted,
    ServerEvent.ConversationItemInputAudioTranscriptionFailed,
    ServerEvent.ConversationItemTruncated,
    ServerEvent.ConversationItemDeleted,
    ServerEvent.ResponseCreated,
    ServerEvent.ResponseDone,
    ServerEvent.ResponseOutputItemAdded,
    ServerEvent.ResponseOutputItemDone,
    ServerEvent.ResponseContentPartAdded,
    ServerEvent.ResponseContentPartDone,
    ServerEvent.ResponseTextDeltaAdded,
    ServerEvent.ResponseTextDone,
    ServerEvent.ResponseAudioTranscriptDelta,
    ServerEvent.ResponseAudioTranscriptDone,
    ServerEvent.ResponseAudioDelta,
    ServerEvent.ResponseAudioDone,
    ServerEvent.ResponseFunctionCallArgumentsDelta,
    ServerEvent.ResponseFunctionCallArgumentsDone,
    ServerEvent.RateLimitsUpdated,
]

ClientEventType = Literal[
    "session.update",
    "input_audio_buffer.append",
    "input_audio_buffer.commit",
    "input_audio_buffer.clear",
    "conversation.item.create",
    "conversation.item.truncate",
    "conversation.item.delete",
    "response.create",
    "response.cancel",
]

ServerEventType = Literal[
    "error",
    "session.created",
    "session.updated",
    "conversation.created",
    "input_audio_buffer.committed",
    "input_audio_buffer.cleared",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "conversation.item.created",
    "conversation.item.input_audio_transcription.completed",
    "conversation.item.input_audio_transcription.failed",
    "conversation.item.truncated",
    "conversation.item.deleted",
    "response.created",
    "response.done",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.text.delta",
    "response.text.done",
    "response.audio_transcript.delta",
    "response.audio_transcript.done",
    "response.audio.delta",
    "response.audio.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "rate_limits.updated",
]
