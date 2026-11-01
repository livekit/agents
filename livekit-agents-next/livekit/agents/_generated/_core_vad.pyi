# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from livekit.local_inference import VAD

from .rtc_frames import AudioFrame
from .vad import VADEvent

class InferenceVad:
    """The Silero VAD behind `inference.VAD`."""
    def __new__(
        cls,
        model_factory: Callable[[], VAD],
        *,
        min_speech_duration: float,
        min_silence_duration: float,
        prefix_padding_duration: float,
        max_buffered_speech: float,
        activation_threshold: float,
        deactivation_threshold: float,
    ) -> Self: ...
    def update_options(
        self,
        *,
        min_speech_duration: float,
        min_silence_duration: float,
        prefix_padding_duration: float,
        max_buffered_speech: float,
        activation_threshold: float,
        deactivation_threshold: float,
    ) -> None:
        """Every option, applied by each running stream before its next frame."""
    def stream(self) -> InferenceVadStream:
        """One detection; each event carries the audio it refers to."""

class InferenceVadStream:
    def push(self, item: AudioFrame) -> None: ...
    def flush(self) -> None: ...
    def end_input(self) -> None: ...
    async def recv(self) -> VADEvent | None: ...
    async def aclose(self) -> None: ...
