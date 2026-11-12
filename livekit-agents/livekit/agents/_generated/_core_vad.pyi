# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from livekit.local_inference import VAD

from ..types import NotGivenOr
from .inference_models import VADModels
from .rtc_frames import AudioFrame
from .vad import VADCapabilities, VADEvent

class InferenceVad:
    """The Silero VAD behind `inference.VAD`."""
    def __new__(
        cls,
        model_factory: Callable[[], VAD],
        *,
        model: VADModels = "silero",
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.25,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.5,
        deactivation_threshold: float | None = None,
    ) -> Self:
        """`model_factory` builds the host's model when the build links no native one;
        `deactivation_threshold` is `max(activation_threshold - 0.15, 0.01)` when `None`. Only the
        `"silero"` model exists.
        """
    @property
    def model(self) -> VADModels: ...
    @property
    def capabilities(self) -> VADCapabilities: ...
    def update_options(
        self,
        *,
        min_speech_duration: NotGivenOr[float] = ...,
        min_silence_duration: NotGivenOr[float] = ...,
        prefix_padding_duration: NotGivenOr[float] = ...,
        max_buffered_speech: NotGivenOr[float] = ...,
        activation_threshold: NotGivenOr[float] = ...,
        deactivation_threshold: NotGivenOr[float] = ...,
    ) -> None:
        """The given options, applied by each running stream before its next frame."""
    @property
    def min_silence_duration(self) -> float: ...
    def stream(self) -> InferenceVadStream:
        """One detection; each event carries the audio it refers to."""

class InferenceVadStream:
    def push(self, item: AudioFrame) -> None: ...
    def flush(self) -> None: ...
    def end_input(self) -> None: ...
    async def recv(self) -> VADEvent | None: ...
    async def aclose(self) -> None: ...
