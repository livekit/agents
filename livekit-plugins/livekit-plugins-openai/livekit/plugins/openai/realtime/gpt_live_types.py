"""Wire types of the OpenAI GPT-Live API, as of the 2026-08-24 alpha.

Server events are read with ``Event.construct(**payload)``: it builds nested models without
validating, so a field the alpha reshapes cannot break a live session. Client events serialise with
``model_dump(exclude_none=True)``, which keeps a later ``session.update`` sparse.
"""

from __future__ import annotations

from typing import Any, Literal

from openai import BaseModel

TurnRole = Literal["user", "assistant"]
DelegationTarget = Literal["responses", "client"]
"""``responses`` hands delegated work to a backend model; ``client`` hands it to the application."""
InitialItemRole = Literal["system", "developer", "user", "assistant"]
Channel = Literal["speakable", "commentary"]
"""``speakable`` context prompts the model to act on the text now; ``commentary`` is silent."""

# -- shared parts --------------------------------------------------------------------------------


class InputTextPart(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str = ""


class OutputTextPart(BaseModel):
    """The assistant's own words; every other role speaks in an input part."""

    type: Literal["output_text"] = "output_text"
    text: str = ""


class InitialItem(BaseModel):
    type: Literal["message"] = "message"
    role: InitialItemRole = "user"
    content: list[InputTextPart | OutputTextPart] = []


class FunctionCallOutputItem(BaseModel):
    """Sent to answer a call, and echoed back on acceptance."""

    id: str | None = None
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str = ""
    output: str = ""


# -- session configuration -----------------------------------------------------------------------


class AudioFormat(BaseModel):
    type: Literal["audio/pcm", "audio/pcmu", "audio/pcma"] = "audio/pcm"
    rate: int = 24000


class AudioOutput(BaseModel):
    voice: str | None = None


class AudioConfig(BaseModel):
    format: AudioFormat | None = None
    output: AudioOutput | None = None


class ResponsesConfig(BaseModel):
    """The backend Responses model a delegation is handed to."""

    model: str = ""
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    service_tier: str | None = None
    max_output_tokens: int | None = None
    parallel_tool_calls: bool | None = None


class Delegation(BaseModel):
    type: DelegationTarget = "responses"
    responses: ResponsesConfig | None = None


class Opening(BaseModel):
    """A passage the model speaks before microphone input can interrupt it."""

    text: str = ""


class SessionConfig(BaseModel):
    """Startup configuration; a later update carries only the fields it changes."""

    instructions: str | None = None
    opening: Opening | None = None
    audio: AudioConfig | None = None
    delegation: Delegation | None = None
    initial_items: list[InitialItem] | None = None


class SessionResource(BaseModel):
    """The public session as the service reports it back, bounded to its identity."""

    id: str | None = None
    expires_at: int | None = None
    status: str | None = None


# -- client events -------------------------------------------------------------------------------


class SessionUpdateEvent(BaseModel):
    type: Literal["session.update"] = "session.update"
    event_id: str | None = None
    session: SessionConfig


class SessionFeedbackEvent(BaseModel):
    type: Literal["session.feedback"] = "session.feedback"
    text: str


class InputAudioAppendEvent(BaseModel):
    type: Literal["input_audio.append"] = "input_audio.append"
    event_id: str | None = None
    audio: str


class InputAudioPauseEvent(BaseModel):
    type: Literal["input_audio.pause"] = "input_audio.pause"
    event_id: str | None = None


class InputAudioResumeEvent(BaseModel):
    type: Literal["input_audio.resume"] = "input_audio.resume"
    event_id: str | None = None


class SessionContextAppendEvent(BaseModel):
    type: Literal["session.context.append"] = "session.context.append"
    event_id: str | None = None
    channel: Channel | None = None
    content: list[InputTextPart]


class DelegationContextAppendEvent(BaseModel):
    """Answers a client-targeted delegation; unused while delegation runs on Responses."""

    type: Literal["delegation.context.append"] = "delegation.context.append"
    event_id: str | None = None
    delegation_item_id: str
    channel: Channel | None = None
    content: list[InputTextPart]


class DelegationFunctionCallOutputCreateEvent(BaseModel):
    type: Literal["delegation.function_call_output.create"] = (
        "delegation.function_call_output.create"
    )
    event_id: str | None = None
    item: FunctionCallOutputItem


class SessionCloseEvent(BaseModel):
    type: Literal["session.close"] = "session.close"
    event_id: str | None = None


ClientEvent = (
    SessionUpdateEvent
    | SessionFeedbackEvent
    | InputAudioAppendEvent
    | InputAudioPauseEvent
    | InputAudioResumeEvent
    | SessionContextAppendEvent
    | DelegationContextAppendEvent
    | DelegationFunctionCallOutputCreateEvent
    | SessionCloseEvent
)


# -- server events -------------------------------------------------------------------------------


class SessionStartedEvent(BaseModel):
    type: Literal["session.started"] = "session.started"
    session: SessionResource = SessionResource()


class SessionUpdatedEvent(BaseModel):
    """Echoes ``event_id`` where the update carried one, so concurrent updates can be told apart."""

    type: Literal["session.updated"] = "session.updated"
    event_id: str | None = None
    session: SessionResource = SessionResource()


class ContextWindowApproachingEvent(BaseModel):
    type: Literal["session.context_window.approaching"] = "session.context_window.approaching"
    rollover_id: str = ""
    expires_at: int | None = None


class ContextWindowRolledOverEvent(BaseModel):
    type: Literal["session.context_window.rolled_over"] = "session.context_window.rolled_over"
    rollover_id: str = ""


class SessionOpeningStartedEvent(BaseModel):
    type: Literal["session.opening.started"] = "session.opening.started"


class SessionOpeningCompletedEvent(BaseModel):
    """The protected phase is over; it does not certify what the model said."""

    type: Literal["session.opening.completed"] = "session.opening.completed"


class InputAudioPausedEvent(BaseModel):
    type: Literal["input_audio.paused"] = "input_audio.paused"


class InputAudioResumedEvent(BaseModel):
    type: Literal["input_audio.resumed"] = "input_audio.resumed"


class InputAudioDtmfEventReceivedEvent(BaseModel):
    type: Literal["input_audio.dtmf_event_received"] = "input_audio.dtmf_event_received"
    event_id: str | None = None
    event: str = ""


class OutputAudioDeltaEvent(BaseModel):
    """A gap between ranges is silence the service omitted, not time it compressed."""

    type: Literal["output_audio.delta"] = "output_audio.delta"
    audio: str = ""
    start_ms: int | None = None
    end_ms: int | None = None


class SessionContextAppendedEvent(BaseModel):
    type: Literal["session.context.appended"] = "session.context.appended"
    start_ms: int | None = None
    end_ms: int | None = None


class DelegationContextAppendedEvent(BaseModel):
    type: Literal["delegation.context.appended"] = "delegation.context.appended"
    delegation_item_id: str = ""
    start_ms: int | None = None
    end_ms: int | None = None


class DelegationFunctionCallOutputCreatedEvent(BaseModel):
    """The result was accepted; the delegation itself runs on until a terminal ``response.*``."""

    type: Literal["delegation.function_call_output.created"] = (
        "delegation.function_call_output.created"
    )
    item: FunctionCallOutputItem = FunctionCallOutputItem()


class TranscriptItem(BaseModel):
    id: str | None = None
    type: str | None = None
    text: str = ""


class InputTranscriptAddedEvent(BaseModel):
    type: Literal["input_transcript.added"] = "input_transcript.added"
    start_ms: int | None = None
    end_ms: int | None = None
    item: TranscriptItem = TranscriptItem()


class OutputTranscriptAddedEvent(BaseModel):
    """One complete fragment; its boundaries follow cadence, not the turn."""

    type: Literal["output_transcript.added"] = "output_transcript.added"
    start_ms: int | None = None
    end_ms: int | None = None
    item: TranscriptItem = TranscriptItem()


class Turn(BaseModel):
    id: str = ""
    role: TurnRole | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    transcript: str | None = None


class TurnCreatedEvent(BaseModel):
    type: Literal["turn.created"] = "turn.created"
    turn: Turn = Turn()


class TurnDeltaEvent(BaseModel):
    type: Literal["turn.delta"] = "turn.delta"
    turn_id: str = ""
    start_ms: int | None = None
    end_ms: int | None = None
    delta: str = ""


class TurnDoneEvent(BaseModel):
    type: Literal["turn.done"] = "turn.done"
    turn: Turn = Turn()


class DelegationItem(BaseModel):
    id: str | None = None
    type: Literal["delegation"] = "delegation"
    target: DelegationTarget | None = None
    response_id: str | None = None
    content: list[InputTextPart] = []


class DelegationCreatedEvent(BaseModel):
    type: Literal["delegation.created"] = "delegation.created"
    offset_ms: int | None = None
    item: DelegationItem = DelegationItem()


class FunctionCallItem(BaseModel):
    """Only the completed item carries the name and call id; the argument events do not."""

    id: str | None = None
    type: str | None = None
    status: str | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None


class ResponseOutputItemDoneEvent(BaseModel):
    """One of the ``response.*`` family the Responses delegation emits without a wrapper."""

    type: Literal["response.output_item.done"] = "response.output_item.done"
    sequence_number: int | None = None
    output_index: int | None = None
    item: FunctionCallItem = FunctionCallItem()


class InputTokenDetails(BaseModel):
    text_tokens: int = 0
    audio_tokens: int = 0
    image_tokens: int = 0
    cached_tokens: int = 0


class OutputTokenDetails(BaseModel):
    text_tokens: int = 0
    audio_tokens: int = 0
    image_tokens: int = 0
    cached_tokens: int = 0


class BackendInputTokenDetails(BaseModel):
    cached_tokens: int = 0
    cache_write_tokens: int = 0


class BackendOutputTokenDetails(BaseModel):
    reasoning_tokens: int = 0


class BackendModelUsage(BaseModel):
    model: str = ""
    input_tokens: int = 0
    input_tokens_details: BackendInputTokenDetails = BackendInputTokenDetails()
    output_tokens: int = 0
    output_tokens_details: BackendOutputTokenDetails = BackendOutputTokenDetails()
    total_tokens: int = 0


class Usage(BaseModel):
    """Cumulative for the whole session, in either the duration-based or the token-only shape."""

    audio_duration_ms: int = 0
    backend_model_usage: list[BackendModelUsage] | None = None
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    input_token_details: InputTokenDetails = InputTokenDetails()
    output_token_details: OutputTokenDetails = OutputTokenDetails()


class UsageLimit(BaseModel):
    status: str | None = None
    reset_seconds: int | None = None


class SessionUsageUpdatedEvent(BaseModel):
    type: Literal["session.usage.updated"] = "session.usage.updated"
    usage: Usage = Usage()
    usage_limit: UsageLimit | None = None


class SessionClosedEvent(BaseModel):
    type: Literal["session.closed"] = "session.closed"
    reason: str | None = None
    usage: Usage = Usage()


class ErrorBody(BaseModel):
    type: str | None = None
    code: str | None = None
    message: str = ""
    param: str | None = None
    event_id: str | None = None


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error: ErrorBody = ErrorBody()
