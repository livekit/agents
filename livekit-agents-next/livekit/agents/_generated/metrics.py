# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/metrics/base.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, Field

from .. import _bindings
from .usage import TokenUsage


class Metadata(BaseModel):
    model_name: str | None = None
    model_provider: str | None = None


class _BaseMetrics(BaseModel):
    def __repr__(self) -> str:
        return _bindings.metrics_repr(self)


class InterruptionMetrics(_BaseMetrics):
    type: Literal["interruption_metrics"] = "interruption_metrics"
    timestamp: float
    total_duration: float
    """Latest RTT (Round Trip Time) time taken to perform the inference, in seconds."""
    prediction_duration: float
    """Latest time taken to perform the inference from the model side, in seconds."""
    detection_delay: float
    """Latest total time from the onset of the speech to the final prediction, in seconds."""
    num_interruptions: int
    """Number of interruptions detected, incrementally counted."""
    num_backchannels: int
    """Number of backchannels detected, incrementally counted."""
    num_requests: int
    """Number of requests sent to the interruption detection model, incrementally counted."""
    metadata: Metadata | None = None


class EOTInferenceMetrics(_BaseMetrics):
    """Per-inference metrics emitted by the EOT model on each prediction."""

    type: Literal["eot_inference_metrics"] = "eot_inference_metrics"
    timestamp: float
    total_duration: float
    """Earliest audio creation time in an inference to response receive time."""
    detection_delay: float
    """Latest audio creation time in an inference to response receive time."""
    prediction_duration: float
    """Server side model inference time."""
    num_requests: int = 1
    """Number of inference requests made during one inference."""
    metadata: Metadata | None = None


class LLMMetrics(_BaseMetrics):
    type: Literal["llm_metrics"] = "llm_metrics"
    label: str
    request_id: str
    timestamp: float
    duration: float
    ttft: float | None = None
    """Time to first generated token in seconds; ``None`` if the response generated none."""
    cancelled: bool
    tokens: TokenUsage = Field(default_factory=lambda: TokenUsage())
    tokens_per_second: float
    speech_id: str | None = None
    metadata: Metadata | None = None


class STTMetrics(_BaseMetrics):
    type: Literal["stt_metrics"] = "stt_metrics"
    label: str
    request_id: str
    timestamp: float
    duration: float
    """The request duration in seconds, 0.0 if the STT is streaming."""
    audio_duration: float | None = None
    """The duration of the pushed audio in seconds; ``None`` when unknown."""
    streamed: bool
    """Whether the STT is streaming (e.g using websocket)."""
    tokens: TokenUsage = Field(default_factory=lambda: TokenUsage())
    acquire_time: float = 0.0
    """Time in seconds to acquire the connection. (WebSocket only)"""
    connection_reused: bool = False
    """Whether the connection was reused from a pool. (WebSocket only)"""
    metadata: Metadata | None = None


class TTSMetrics(_BaseMetrics):
    type: Literal["tts_metrics"] = "tts_metrics"
    label: str
    request_id: str
    timestamp: float
    ttfb: float | None = None
    """Time to the first audio frame in seconds; ``None`` if no audio arrived."""
    duration: float
    audio_duration: float
    cancelled: bool
    characters_count: int
    """Number of characters synthesized (for character-based billing)."""
    tokens: TokenUsage = Field(default_factory=lambda: TokenUsage())
    """Token-based billing (e.g., OpenAI TTS): text in, audio out."""
    streamed: bool
    acquire_time: float = 0.0
    """Time in seconds to acquire the connection. (WebSocket only)"""
    connection_reused: bool = False
    """Whether the connection was reused from a pool. (WebSocket only)"""
    segment_id: str | None = None
    speech_id: str | None = None
    metadata: Metadata | None = None


class VADMetrics(_BaseMetrics):
    type: Literal["vad_metrics"] = "vad_metrics"
    label: str
    timestamp: float
    idle_time: float
    inference_duration_total: float
    inference_count: int
    metadata: Metadata | None = None


class EOUMetrics(_BaseMetrics):
    type: Literal["eou_metrics"] = "eou_metrics"
    timestamp: float
    end_of_utterance_delay: float
    """Amount of time between the end of speech from VAD and the decision to end the user's turn.
    Set to 0.0 if the end of speech was not detected.
    """
    transcription_delay: float
    """Time taken to obtain the transcript after the end of the user's speech.
    Set to 0.0 if the end of speech was not detected.
    """
    on_user_turn_completed_delay: float
    """Time taken to invoke the user's `Agent.on_user_turn_completed` callback."""
    speech_id: str | None = None
    metadata: Metadata | None = None


class RealtimeModelMetrics(_BaseMetrics):
    type: Literal["realtime_model_metrics"] = "realtime_model_metrics"
    label: str = ""
    request_id: str
    timestamp: float
    """The timestamp of the response creation."""
    duration: float = 0.0
    """The duration of the response from created to done in seconds."""
    connection_duration: float | None = None
    """Metered session-connection seconds (for session-based billing like xAI)."""
    ttft: float | None = None
    """Time to first audio token in seconds; ``None`` if no audio token was sent."""
    cancelled: bool = False
    """Whether the request was cancelled."""
    tokens: TokenUsage = Field(default_factory=lambda: TokenUsage())
    tokens_per_second: float = 0.0
    """The number of tokens per second."""
    acquire_time: float = 0.0
    """Time in seconds to acquire the connection. (WebSocket only)"""
    connection_reused: bool = False
    """Whether the connection was reused from a pool. (WebSocket only)"""
    metadata: Metadata | None = None


class AvatarMetrics(_BaseMetrics):
    type: Literal["avatar_metrics"] = "avatar_metrics"
    timestamp: float
    playback_latency: float = 0
    """Delay between forwarding the first audio frame to the avatar and the playback started."""
    session_started_time: float | None = None
    """Time when the avatar session was started."""
    avatar_joined_time: float | None = None
    """Time when the avatar participant joined and started video track."""
    metadata: Metadata | None = None


AgentMetrics: TypeAlias = (
    STTMetrics
    | LLMMetrics
    | TTSMetrics
    | VADMetrics
    | EOUMetrics
    | EOTInferenceMetrics
    | RealtimeModelMetrics
    | InterruptionMetrics
    | AvatarMetrics
)


Metadata.__module__ = "livekit.agents.metrics"
_BaseMetrics.__module__ = "livekit.agents.metrics"
InterruptionMetrics.__module__ = "livekit.agents.metrics"
EOTInferenceMetrics.__module__ = "livekit.agents.metrics"
LLMMetrics.__module__ = "livekit.agents.metrics"
STTMetrics.__module__ = "livekit.agents.metrics"
TTSMetrics.__module__ = "livekit.agents.metrics"
VADMetrics.__module__ = "livekit.agents.metrics"
EOUMetrics.__module__ = "livekit.agents.metrics"
RealtimeModelMetrics.__module__ = "livekit.agents.metrics"
AvatarMetrics.__module__ = "livekit.agents.metrics"
