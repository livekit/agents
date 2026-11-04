# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from .llm import ChatChunk
from .metrics import AgentMetrics, LLMMetrics, STTMetrics, TTSMetrics
from .stt import SpeechEvent
from .tts import SynthesizedAudio

def runtime_busy_seconds() -> list[float]:
    """Each async worker's total busy seconds so far; empty before the runtime started."""

def metrics_log_fields(metrics: AgentMetrics) -> tuple[str, dict[str, object]] | None:
    """`log_metrics(metrics)` minus the logging: `(message, extra)`, or `None` for do not log
    (`VADMetrics` and `EOTInferenceMetrics` have no branch in the reference).
    """

def stt_batch_metrics(
    event: SpeechEvent,
    start: float,
    now: float,
    audio_durations: Sequence[float],
    timestamp: float,
    label: str,
    model: str,
    provider: str,
) -> STTMetrics:
    """`STT.recognize`'s metrics for one batch recognition, stamped at `timestamp` (unix seconds)
    with the provider's label, model and provider name.
    """

def stt_stream_metrics(
    event: SpeechEvent,
    timestamp: float,
    label: str,
    model: str,
    provider: str,
) -> tuple[STTMetrics | None, bool]:
    """`RecognizeStream`'s metrics for one event, and whether it resets the retry count."""

class LlmMetricState:
    """`LLMStream`'s metric fold as the host drives it."""
    def __new__(cls, start: float, label: str, model: str, provider: str) -> Self:
        """`label`, `model` and `provider`: the provider's, stamped on the record."""
    def observe(self, now: float, chunk: ChatChunk) -> bool:
        """Whether `chunk` is the first with a response."""
    def finish(self, now: float, failed: bool, timestamp: float) -> LLMMetrics | None:
        """`None` for a failed attempt or one that yielded no chunk; `timestamp`: unix seconds."""

class TtsMetricState:
    """The TTS streams' metric fold as the host drives it."""
    def __new__(cls, streamed: bool, label: str, model: str, provider: str) -> Self:
        """`label`, `model` and `provider`: the provider's, stamped on each record."""
    def queue_text(self, text: str) -> None:
        """A streamed segment's text, in the order its audio is finished."""
    def observe(self, now: float, start: float, audio: SynthesizedAudio) -> bool:
        """Whether `audio` ends its segment."""
    def finish(
        self,
        now: float,
        start: float,
        failed: bool,
        batch_text: str,
        timestamp: float,
    ) -> TTSMetrics | None:
        """The record of a finished request or segment, then the figures reset; `timestamp`: unix
        seconds.
        """
