"""Wire types of the OpenAI GPT-Live API.

Server events are read with ``Event.construct(**payload)``, which does not validate, so a field the
service reshapes cannot break a live session; client events serialise with ``exclude_none``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import model_serializer

from openai import BaseModel
from openai.types.responses import ResponseInputItem, ResponseTextConfigParam
from openai.types.shared_params import Reasoning

DelegationTarget = Literal["responses", "client"]
"""``responses`` hands delegated work to a backend model; ``client`` hands it to the application."""
InputRole = Literal["developer", "user", "assistant"]
"""The roles startup history accepts; there is no ``system``."""

# shared parts


class InputTextPart(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str = ""


class OutputTextPart(BaseModel):
    """The assistant's own words; every other role speaks in an input part."""

    type: Literal["output_text"] = "output_text"
    text: str = ""


class InputItem(BaseModel):
    """One message of the startup history."""

    type: Literal["message"] = "message"
    role: InputRole = "user"
    content: list[InputTextPart | OutputTextPart] = []


# session configuration


class AudioFormat(BaseModel):
    type: Literal["audio/pcm", "audio/pcmu", "audio/pcma"] = "audio/pcm"
    rate: int = 24000


class AudioOutput(BaseModel):
    voice: str | dict[str, Any] | None = None
    """A named voice, or ``{"id": "voice_..."}`` for an authorized custom voice."""


class AudioConfig(BaseModel):
    format: AudioFormat | None = None
    output: AudioOutput | None = None


class ResponsesConfig(BaseModel):
    """The backend Responses model a delegation is handed to; sparse after startup."""

    model: str | None = None
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning: Reasoning | None = None
    text: ResponseTextConfigParam | None = None
    service_tier: Literal["auto", "default", "flex", "priority"] | None = None
    max_output_tokens: int | None = None


class Delegation(BaseModel):
    """Chosen at startup; the type is immutable, the responses settings are not."""

    type: DelegationTarget = "responses"
    responses: ResponsesConfig | None = None


class SessionConfig(BaseModel):
    """The whole startup configuration; ``model`` is the only required field."""

    model: str
    instructions: str | None = None
    input: list[InputItem] | None = None
    audio: AudioConfig | None = None
    delegation: Delegation | None = None


class SessionUpdateConfig(BaseModel):
    """What a running session still accepts: delegation settings within the startup mode."""

    delegation: Delegation


class SessionResource(BaseModel):
    id: str | None = None


# client events


class SessionStartEvent(BaseModel):
    type: Literal["session.start"] = "session.start"
    event_id: str | None = None
    session: SessionConfig


class SessionUpdateEvent(BaseModel):
    type: Literal["session.update"] = "session.update"
    event_id: str | None = None
    session: SessionUpdateConfig


class InputAudioAppendEvent(BaseModel):
    type: Literal["session.input_audio.append"] = "session.input_audio.append"
    audio: str


class InputAudioMuteEvent(BaseModel):
    type: Literal["session.input_audio.mute"] = "session.input_audio.mute"
    event_id: str | None = None


class InputAudioUnmuteEvent(BaseModel):
    type: Literal["session.input_audio.unmute"] = "session.input_audio.unmute"
    event_id: str | None = None


class _ContextAppendEvent(BaseModel):
    """Shared shape of the three appends, whose ``delegation_id`` is required even when null."""

    event_id: str | None = None
    delegation_id: str | None
    content: str

    @model_serializer(mode="wrap")
    def _keep_delegation_id(self, handler: Any) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        data["delegation_id"] = self.delegation_id
        return data


class InstructionsAppendEvent(_ContextAppendEvent):
    """Developer instructions the model follows from now on; startup instructions stay."""

    type: Literal["session.instructions.append"] = "session.instructions.append"


class ThinkingAppendEvent(_ContextAppendEvent):
    """Silent context: informs later replies without being spoken when appended."""

    type: Literal["session.thinking.append"] = "session.thinking.append"


class CommentaryAppendEvent(_ContextAppendEvent):
    """Speakable context: the model paraphrases it aloud."""

    type: Literal["session.commentary.append"] = "session.commentary.append"


class ResponseItemCreateEvent(BaseModel):
    """Queues any Responses API input item for the backend; nothing runs until ``response.create``."""

    type: Literal["response.item.create"] = "response.item.create"
    event_id: str | None = None
    item: ResponseInputItem


class ResponseCreateEvent(BaseModel):
    """Starts or continues delegated Responses work, with no body of its own."""

    type: Literal["response.create"] = "response.create"
    event_id: str | None = None


class SessionCloseEvent(BaseModel):
    type: Literal["session.close"] = "session.close"
    event_id: str | None = None


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


class SessionStartedEvent(BaseModel):
    type: Literal["session.started"] = "session.started"
    session: SessionResource = SessionResource()


class OutputAudioDeltaEvent(BaseModel):
    """One frame of the continuous output stream; it carries no timing of its own."""

    type: Literal["session.output_audio.delta"] = "session.output_audio.delta"
    delta: str = ""


class TranscriptDeltaEvent(BaseModel):
    """A fragment of user or assistant speech, over a half-open span of the session timeline."""

    type: str = ""
    delta: str = ""
    start_ms: int | None = None
    end_ms: int | None = None


class DelegationInfo(BaseModel):
    """Metadata only: the task itself is whatever the conversation says."""

    id: str | None = None
    target: DelegationTarget | None = None


class SessionDelegationCreatedEvent(BaseModel):
    type: Literal["session.delegation.created"] = "session.delegation.created"
    delegation: DelegationInfo = DelegationInfo()


class ResponseUsage(BaseModel):
    class InputDetails(BaseModel):
        cached_tokens: int = 0
        cache_write_tokens: int = 0

    class OutputDetails(BaseModel):
        reasoning_tokens: int = 0

    input_tokens: int = 0
    input_tokens_details: InputDetails = InputDetails()
    output_tokens: int = 0
    output_tokens_details: OutputDetails = OutputDetails()
    total_tokens: int = 0


class ResponseSnapshot(BaseModel):
    """A reduced Responses object: ``output`` is always empty here, so never read calls from it."""

    id: str | None = None
    model: str | None = None
    usage: ResponseUsage | None = None
    error: dict[str, Any] | None = None
    incomplete_details: dict[str, Any] | None = None


class OutputItem(BaseModel):
    """Only a completed function-call item carries all of name, call id and arguments."""

    id: str | None = None
    type: str | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None


class ResponsesEvent(BaseModel):
    """The Responses streaming event nested in a ``response.event`` envelope."""

    type: str = ""
    response: ResponseSnapshot | None = None
    item: OutputItem | None = None


class ResponseEventEnvelope(BaseModel):
    type: Literal["response.event"] = "response.event"
    delegation_id: str | None = None
    event: ResponsesEvent = ResponsesEvent()


class Usage(BaseModel):
    """Cumulative voice usage for the whole session, in seconds."""

    seconds: float = 0.0


class ContextWindow(BaseModel):
    usage_ratio: float | None = None


class SessionUsageUpdatedEvent(BaseModel):
    type: Literal["session.usage.updated"] = "session.usage.updated"
    usage: Usage = Usage()
    context_window: ContextWindow | None = None


class SessionClosedEvent(BaseModel):
    type: Literal["session.closed"] = "session.closed"
    reason: (
        Literal["close_requested", "expired", "content", "remote_hangup", "connection_lost"] | None
    ) = None
    usage: Usage = Usage()


class ErrorBody(BaseModel):
    type: str | None = None
    code: str | None = None
    message: str = ""
    param: str | None = None
    client_event_id: str | None = None


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error: ErrorBody = ErrorBody()
