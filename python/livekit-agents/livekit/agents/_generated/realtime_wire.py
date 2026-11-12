# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/realtime.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable
from dataclasses import dataclass
from inspect import get_annotations
from typing import TYPE_CHECKING, Literal, TypeAlias

from livekit import rtc

from .chat_items import ChatItem, FunctionCall
from .transcript import TranscriptChunk

if TYPE_CHECKING:
    from livekit.agents.llm.chat_context import FunctionCall
    from livekit.agents.llm.tool_context import ToolChoice

    from .metrics import AgentMetrics
    from .rtc_frames import AudioFrame


@dataclass
class RemoteItemAddedEvent:
    """The mirror gained an item the server created."""

    previous_item_id: str | None
    """The item this one was inserted after; `None` means "appended"."""
    item: ChatItem


@dataclass(kw_only=True, slots=True)
class RealtimeMetrics:
    """What the model measured (each response's `RealtimeModelMetrics`, the connection's, the input
    transcription's `STTMetrics`), for the session's metrics and usage.
    """

    metrics: AgentMetrics


@dataclass(kw_only=True, slots=True)
class FunctionCallReady:
    """A function call the model made."""

    call: FunctionCall
    """The plugin's `llm.FunctionCall`, not the legacy model its derive generates."""


@dataclass(kw_only=True, slots=True)
class AudioDelta:
    """One chunk of assistant audio."""

    message_id: str
    response_id: str | None = None
    """The generation this frame belongs to, when the plugin says."""
    frame: AudioFrame


@dataclass(kw_only=True, slots=True)
class MessageModalities:
    """What a message turned out to contain."""

    message_id: str
    response_id: str | None = None
    """The generation this message belongs to; see `AudioDelta::response_id`."""
    modalities: Modalities


@dataclass(kw_only=True, slots=True)
class GenerationDone:
    """The turn is over. Carries what `response.done` reported."""

    response_id: str


@dataclass(kw_only=True, slots=True)
class ServerGeneration:
    """A generation the server started by itself (`generation_created` with `user_initiated=False`):
    the host leg drains it like a requested one.
    """

    generation: object
    """The plugin's `GenerationCreatedEvent`, handed back to `_core_drain`."""


@dataclass(kw_only=True, slots=True)
class GenerationCreated:
    """A turn has begun."""

    response_id: str
    req_id: str | None = None
    """`Some` exactly when this answers a [`RealtimeCommand::GenerateReply`]."""


@dataclass(kw_only=True, slots=True)
class MessageStarted:
    """A new assistant message inside the current generation."""

    message_id: str
    response_id: str | None = None
    """The generation this message belongs to; lets the driver drop deltas of a refused one."""


@dataclass(kw_only=True, slots=True)
class TextDelta:
    """A transcript or text delta."""

    message_id: str
    response_id: str | None = None
    """The generation this delta belongs to; see `AudioDelta::response_id`."""
    delta: str
    start_time: float | None = None
    """The `io.TranscriptChunk` stamp when the provider sends one; never synthesised."""


@dataclass(kw_only=True, slots=True)
class MessageDone:
    """The message is complete."""

    message_id: str
    response_id: str | None = None
    """The generation this message belongs to; see `AudioDelta::response_id`."""


@dataclass
class InputSpeechStoppedEvent:
    """`InputSpeechStoppedEvent` (`llm/realtime.py:53`)."""

    user_transcription_enabled: bool


@dataclass
class InputTranscriptionCompleted:
    """`InputTranscriptionCompleted` (`llm/realtime.py:152`); also emitted for partials (`is_final`)."""

    item_id: str
    transcript: str
    is_final: bool
    confidence: float | None = None
    """Geometric mean of the per-token probabilities, when the provider returns logprobs."""
    turn_started_at: float | None = None
    """When the turn began, as `time.time()`: final transcripts only, when server VAD reported an
    onset for `item_id`. Wall seconds, whatever clock the session runs on: it may precede the
    session (the message is placed at it, `agent_activity.py:2040`).
    """


@dataclass
class InputSpeechStartedEvent:
    """Server-side VAD heard the caller start speaking."""


@dataclass
class RealtimeSessionReconnectedEvent:
    """The session reconnected to the server."""


@dataclass(kw_only=True, slots=True)
class HostRealtimeError:
    """`error` as a host session reports it, `timestamp` the host's `time.time()`."""

    label: str
    message: str
    recoverable: bool
    timestamp: float
    """With no anchor yet it lands on `Ts::ZERO`, so the error still forwards."""
    req_id: str | None = None
    code: str | None = None


Modalities: TypeAlias = Literal["text", "audio"]
"""What a message contains: audio always comes with its transcript."""


@dataclass(kw_only=True, slots=True)
class PushAudio:
    """One frame of microphone audio."""

    frame: AudioFrame


@dataclass(kw_only=True, slots=True)
class GenerateReply:
    """Ask for a turn."""

    req_id: str
    instructions: str | None
    tool_choice: ToolChoice | None
    """Sent only for sessions reporting `per_response_tool_choice`; `None` when absent."""


@dataclass(kw_only=True, slots=True)
class Say:
    """Speak these exact words."""

    req_id: str
    text: str
    add_to_chat_ctx: bool
    """`say(add_to_chat_ctx=)`; the reference's realtime model ignores `False` (R-10)."""


@dataclass(kw_only=True, slots=True)
class CommitAudio:
    """End the user's turn explicitly."""


@dataclass(kw_only=True, slots=True)
class ClearAudio:
    """Discard audio the server has buffered but not yet turned into a turn."""


@dataclass(kw_only=True, slots=True)
class Interrupt:
    """Cancel the open generation. See [`RealtimeCommand::Interrupt`]."""

    response_id: str | None
    """The generation being cancelled; `None` is "whatever is in progress"."""


@dataclass(kw_only=True, slots=True)
class Truncate:
    """How much of a message the user actually heard."""

    message_id: str
    has_audio: bool
    audio_end_ms: int
    audio_transcript: str | None


@dataclass
class MessageGeneration:
    message_id: str
    text_stream: AsyncIterable[str | TranscriptChunk]
    audio_stream: AsyncIterable[rtc.AudioFrame]
    modalities: Awaitable[list[Literal['text', 'audio']]]


@dataclass
class GenerationCreatedEvent:
    message_stream: AsyncIterable[MessageGeneration]
    function_stream: AsyncIterable[FunctionCall]
    user_initiated: bool
    """True if the message was generated by the user using generate_reply()"""
    response_id: str | None = None
    """The response ID associated with this generation, used for metrics attribution"""


Event: TypeAlias = (
    InputSpeechStartedEvent
    | InputSpeechStoppedEvent
    | InputTranscriptionCompleted
    | GenerationCreated
    | ServerGeneration
    | MessageStarted
    | TextDelta
    | AudioDelta
    | MessageModalities
    | MessageDone
    | FunctionCallReady
    | GenerationDone
    | RemoteItemAddedEvent
    | RealtimeSessionReconnectedEvent
    | RealtimeMetrics
    | HostRealtimeError
)
"""`MessageStarted` through `GenerationDone` are the inside of the host's one `"generation_created"`.
A host session reports one event at a time, the reference's stream of streams flattened by the
generated `RealtimeSession._core_drain`. The native variants are not on that wire.
"""


RealtimeCommand: TypeAlias = (
    PushAudio | GenerateReply | Say | CommitAudio | ClearAudio | Interrupt | Truncate
)
"""What the driver sends a session. A host session is sent the ones it applies at once
(`HostRealtimeSession::send`) and the reply requests; the leg answers the native ones itself or
through the session's update calls.
"""


RemoteItemAddedEvent.__annotations__ = get_annotations(RemoteItemAddedEvent, eval_str=True)
RemoteItemAddedEvent.__module__ = "livekit.agents.llm.realtime"
InputSpeechStoppedEvent.__annotations__ = get_annotations(InputSpeechStoppedEvent, eval_str=True)
InputSpeechStoppedEvent.__module__ = "livekit.agents.llm.realtime"
InputTranscriptionCompleted.__annotations__ = get_annotations(InputTranscriptionCompleted, eval_str=True)
InputTranscriptionCompleted.__module__ = "livekit.agents.llm.realtime"
InputSpeechStartedEvent.__annotations__ = get_annotations(InputSpeechStartedEvent, eval_str=True)
InputSpeechStartedEvent.__module__ = "livekit.agents.llm.realtime"
RealtimeSessionReconnectedEvent.__annotations__ = get_annotations(RealtimeSessionReconnectedEvent, eval_str=True)
RealtimeSessionReconnectedEvent.__module__ = "livekit.agents.llm.realtime"
MessageGeneration.__annotations__ = get_annotations(MessageGeneration, eval_str=True)
MessageGeneration.__module__ = "livekit.agents.llm.realtime"
GenerationCreatedEvent.__annotations__ = get_annotations(GenerationCreatedEvent, eval_str=True)
GenerationCreatedEvent.__module__ = "livekit.agents.llm.realtime"
