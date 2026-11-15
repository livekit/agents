"""Wire types of the OpenAI GPT-Live API.

Client events are the messages the session sends, as JSON. Server events declare only the fields
the session reads, each optional: an event type not declared here is skipped, and a field or a
value the service adds never breaks a live session.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from pydantic import JsonValue

from openai.types.responses import ResponseTextConfigParam
from openai.types.shared_params import Reasoning

DelegationTarget = Literal["responses", "client"]
"""``responses`` hands delegated work to a backend model; ``client`` hands it to the application."""
InputRole = Literal["developer", "user", "assistant"]
"""The roles startup history accepts; there is no ``system``."""
ServiceTier = Literal["auto", "default", "flex", "priority", "ultrafast"]
"""Processing tier the session asks the service for, sent as the ``OpenAI-Service-Tier`` header."""


class CustomVoice(TypedDict):
    """An authorized custom voice."""

    id: str


# session configuration


class TextPart(TypedDict):
    """The assistant's own words are ``output_text``; every other role speaks in ``input_text``."""

    type: Literal["input_text", "output_text"]
    text: str


class InputItem(TypedDict):
    """One message of the startup history."""

    type: Literal["message"]
    role: InputRole
    content: list[TextPart]


class AudioFormat(TypedDict):
    type: Literal["audio/pcm", "audio/pcmu", "audio/pcma"]
    rate: int


class AudioOutput(TypedDict):
    voice: str | CustomVoice


class AudioConfig(TypedDict):
    format: AudioFormat
    output: AudioOutput


class FunctionToolChoice(TypedDict):
    type: Literal["function"]
    name: str


class ResponsesConfig(TypedDict, total=False):
    """The backend Responses model a delegation is handed to; sparse after startup."""

    model: str
    instructions: str
    tools: list[dict[str, JsonValue]]
    tool_choice: str | FunctionToolChoice
    parallel_tool_calls: bool
    reasoning: Reasoning
    text: ResponseTextConfigParam
    service_tier: ServiceTier
    max_output_tokens: int


class Delegation(TypedDict):
    """Chosen at startup; the type is immutable, the responses settings are not."""

    type: DelegationTarget
    responses: NotRequired[ResponsesConfig]


class SessionConfig(TypedDict):
    """The whole startup configuration; ``model`` is the only required field."""

    model: str
    instructions: NotRequired[str]
    input: NotRequired[list[InputItem]]
    audio: NotRequired[AudioConfig]
    delegation: NotRequired[Delegation]


class SessionUpdateConfig(TypedDict):
    """What a running session still accepts: delegation settings within the startup mode."""

    delegation: Delegation


class FunctionCallOutput(TypedDict):
    """A backend function call's result."""

    type: Literal["function_call_output"]
    call_id: str
    output: str


# client events


class SessionStartEvent(TypedDict):
    type: Literal["session.start"]
    event_id: str
    session: SessionConfig


class SessionUpdateEvent(TypedDict):
    type: Literal["session.update"]
    event_id: str
    session: SessionUpdateConfig


class InputAudioAppendEvent(TypedDict):
    type: Literal["session.input_audio.append"]
    audio: str


class InputAudioMuteEvent(TypedDict):
    type: Literal["session.input_audio.mute"]
    event_id: str


class InputAudioUnmuteEvent(TypedDict):
    type: Literal["session.input_audio.unmute"]
    event_id: str


class InstructionsAppendEvent(TypedDict):
    """Developer instructions the model follows from now on; startup instructions stay."""

    type: Literal["session.instructions.append"]
    event_id: str
    delegation_id: str | None
    """Required even when null."""
    content: str


class ThinkingAppendEvent(TypedDict):
    """Silent context: informs later replies without being spoken when appended."""

    type: Literal["session.thinking.append"]
    event_id: str
    delegation_id: str | None
    content: str


class CommentaryAppendEvent(TypedDict):
    """Speakable context: the model paraphrases it aloud."""

    type: Literal["session.commentary.append"]
    event_id: str
    delegation_id: str | None
    content: str


class ResponseItemCreateEvent(TypedDict):
    """Queues an input item for the backend; nothing runs until ``response.create``."""

    type: Literal["response.item.create"]
    event_id: str
    item: FunctionCallOutput


class ResponseCreateEvent(TypedDict):
    """Starts or continues delegated Responses work, with no body of its own."""

    type: Literal["response.create"]
    event_id: str


class SessionCloseEvent(TypedDict):
    """Asks the service to end the session; it answers with ``session.closed``."""

    type: Literal["session.close"]
    event_id: str


ClientEvent = (
    SessionStartEvent
    | SessionUpdateEvent
    | InputAudioAppendEvent
    | InputAudioMuteEvent
    | InputAudioUnmuteEvent
    | InstructionsAppendEvent
    | ThinkingAppendEvent
    | CommentaryAppendEvent
    | ResponseItemCreateEvent
    | ResponseCreateEvent
    | SessionCloseEvent
)


# server events


class SessionResource(TypedDict):
    id: NotRequired[str | None]


class SessionStartedEvent(TypedDict):
    type: Literal["session.started"]
    session: NotRequired[SessionResource]


class OutputAudioDeltaEvent(TypedDict):
    """One frame of the continuous output stream; it carries no timing of its own."""

    type: Literal["session.output_audio.delta"]
    delta: NotRequired[str]


class OutputTranscriptDeltaEvent(TypedDict):
    """A fragment of the model's speech, over a half-open span of the session timeline."""

    type: Literal["session.output_transcript.delta"]
    delta: NotRequired[str]
    start_ms: NotRequired[int | None]
    end_ms: NotRequired[int | None]


class InputTranscriptDeltaEvent(TypedDict):
    """A fragment of the caller's speech, over a half-open span of the session timeline."""

    type: Literal["session.input_transcript.delta"]
    delta: NotRequired[str]
    start_ms: NotRequired[int | None]
    end_ms: NotRequired[int | None]


class DelegationInfo(TypedDict):
    """Metadata only: the task itself is whatever the conversation says."""

    id: NotRequired[str | None]
    target: NotRequired[str | None]


class SessionDelegationCreatedEvent(TypedDict):
    type: Literal["session.delegation.created"]
    delegation: NotRequired[DelegationInfo]


class InputTokensDetails(TypedDict):
    cached_tokens: NotRequired[int]
    cache_write_tokens: NotRequired[int]


class OutputTokensDetails(TypedDict):
    reasoning_tokens: NotRequired[int]


class ResponseUsage(TypedDict):
    input_tokens: NotRequired[int]
    input_tokens_details: NotRequired[InputTokensDetails]
    output_tokens: NotRequired[int]
    output_tokens_details: NotRequired[OutputTokensDetails]


class ResponseSnapshot(TypedDict):
    """A reduced Responses object: ``output`` is always empty here, so never read calls from it."""

    id: NotRequired[str | None]
    model: NotRequired[str | None]
    usage: NotRequired[ResponseUsage | None]
    error: NotRequired[JsonValue]
    incomplete_details: NotRequired[JsonValue]


class OutputItem(TypedDict):
    """Fields needed to dispatch a completed backend function call."""

    id: NotRequired[str | None]
    type: NotRequired[str | None]
    status: NotRequired[str | None]
    call_id: NotRequired[str | None]
    name: NotRequired[str | None]
    arguments: NotRequired[str | None]


class ResponsesEvent(TypedDict):
    """The Responses streaming event nested in a ``response.event`` envelope: any of its types,
    of which the session reads a few."""

    type: str
    response: NotRequired[ResponseSnapshot | None]
    item: NotRequired[OutputItem | None]


class ResponseEventEnvelope(TypedDict):
    type: Literal["response.event"]
    delegation_id: NotRequired[str | None]
    event: ResponsesEvent


class Usage(TypedDict):
    """Cumulative voice usage for the whole session, in seconds."""

    seconds: NotRequired[float]


class ContextWindow(TypedDict):
    usage_ratio: NotRequired[float | None]


class SessionUsageUpdatedEvent(TypedDict):
    type: Literal["session.usage.updated"]
    usage: NotRequired[Usage]
    context_window: NotRequired[ContextWindow | None]


class SessionClosedEvent(TypedDict):
    type: Literal["session.closed"]
    reason: NotRequired[str | None]
    usage: NotRequired[Usage]


class ErrorBody(TypedDict):
    type: NotRequired[str | None]
    code: NotRequired[str | None]
    message: NotRequired[str | None]
    param: NotRequired[str | None]
    client_event_id: NotRequired[str | None]


class ErrorEvent(TypedDict):
    type: Literal["error"]
    error: NotRequired[ErrorBody]


ServerEvent = (
    SessionStartedEvent
    | OutputAudioDeltaEvent
    | OutputTranscriptDeltaEvent
    | InputTranscriptDeltaEvent
    | SessionDelegationCreatedEvent
    | ResponseEventEnvelope
    | SessionUsageUpdatedEvent
    | SessionClosedEvent
    | ErrorEvent
)
