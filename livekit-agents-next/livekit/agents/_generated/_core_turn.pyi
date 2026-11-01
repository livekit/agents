# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Self

from livekit.local_inference import EOT

from .connect import APIConnectOptions
from .keyterms import KeytermDetectionOptions, ResolvedKeytermDetection
from .metrics import EOTInferenceMetrics
from .rtc_frames import AudioFrame
from .turn_events import EotPrediction, ThresholdOverride, TurnDetectorModel

class InferenceEot:
    """The detector behind `inference.TurnDetector`: the user's threshold overrides over the model's
    defaults, shared by every stream it opens, so a stream's cloud-to-local fallback rescales the
    table the detector answers from.
    """
    def __new__(
        cls,
        model: TurnDetectorModel,
        overrides: ThresholdOverride | None,
        backchannel_overrides: ThresholdOverride | None,
        *,
        local_fallback: bool,
        base_url: str,
        api_key: str,
        api_secret: str,
        sample_rate: int,
        local_model: Callable[[], EOT],
        on_metrics: Callable[[EOTInferenceMetrics], None] | None = None,
    ) -> Self:
        """`turn-detector-v1` connects to the gateway at `base_url`; `local_fallback` lets it degrade
        to the mini model, which `local_model` builds when the build links none.
        """
    @property
    def model(self) -> TurnDetectorModel:
        """The active model, which a stream's fallback changes."""
    @property
    def overrides(self) -> ThresholdOverride | None:
        """`None` is `NOT_GIVEN`."""
    @property
    def backchannel_overrides(self) -> ThresholdOverride | None: ...
    def lookup(self, language: str | None = None) -> float | None:
        """`language=None` scores as English, not the global default."""
    def lookup_backchannel(self, language: str | None = None) -> float | None:
        """`None` disables backchannel detection; a `0.0` threshold also means off."""
    def supports(self, language: str | None = None) -> bool: ...
    def update_overrides(self, overrides: ThresholdOverride | None = None) -> None: ...
    def update_backchannel_overrides(self, overrides: ThresholdOverride | None = None) -> None: ...
    def stream(
        self,
        headers: Sequence[tuple[str, str]],
        conn_options: APIConnectOptions,
    ) -> EotStream:
        """One stream of the user's audio. `headers` (`get_inference_headers()`) identify the job on
        each gateway connection.
        """

class EotStream:
    """`TurnDetectorStream`: one stream of the user's audio, answering `predict()` at each pause. The
    session's own detector when handed to it, which implements its stream operations.
    """
    @property
    def model(self) -> TurnDetectorModel:
        """The active model, which changes under the caller after a fallback."""
    @property
    def is_fallback(self) -> bool: ...
    @property
    def is_degraded(self) -> bool:
        """True once the cloud transport is gone with no local model to replace it."""
    async def predict(self) -> EotPrediction:
        """The prediction over the audio so far; supersedes any in flight."""
    def cancel_inference(self, *, timed_out: bool = False) -> None: ...
    def flush(self, reason: str | None = None) -> None:
        """A segment boundary, not a close."""
    def push_audio(self, frame: AudioFrame) -> None: ...
    def end_input(self) -> None: ...
    async def aclose(self) -> None: ...

class InterruptionDetector:
    """`inference.AdaptiveInterruptionDetector`: the gateway model that tells an interruption from a
    backchannel while the agent speaks. Natively the session's interruption source.
    """
    def __new__(
        cls,
        *,
        threshold: float | None = None,
        min_interruption_duration: float = 0.05,
        max_audio_duration: float = 3.0,
        audio_prefix_duration: float = 1.0,
        detection_interval: float = 0.1,
        inference_timeout: float = 0.7,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        conn_options: APIConnectOptions | None = None,
        token_ttl_secs: int = 600,
    ) -> Self:
        """`base_url`, `api_key` and `api_secret` are required; the Python class resolves them from
        the server's settings.
        """
    def update_options(
        self,
        *,
        threshold: float | None = None,
        min_interruption_duration: float | None = None,
    ) -> None:
        """Shared with live streams, which reconnect to pick the options up."""

def resolve_keyterm_detection(config: KeytermDetectionOptions | None) -> ResolvedKeytermDetection:
    """`_resolve_detection`: the defaults, then whatever keys the caller set. `turn_interval`
    comes back clamped to at least 1, and an empty `instructions` resolves to the default prompt.
    """
