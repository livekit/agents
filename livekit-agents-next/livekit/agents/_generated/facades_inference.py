# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .. import _bindings, _core
from .._bindings import silero_model
from ..types import NOT_GIVEN
from ..utils import is_given
from .vad import VAD as BaseVAD, VADStream

if TYPE_CHECKING:
    from .. import inference
    from ..types import NotGivenOr
    from .inference_models import VADModels


class VAD(BaseVAD):
    """`inference.VAD`: Silero voice activity detection, the model native to the core when the build
    links it, else `livekit-local-inference`'s.
    """

    def __init__(
        self,
        *,
        model: VADModels = "silero",
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.25,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.5,
        deactivation_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Silero voice activity detection.

        Args:
            model: The VAD model; only ``"silero"``.
            min_speech_duration: Speech needed before a start of speech.
            min_silence_duration: Silence needed before an end of speech.
            prefix_padding_duration: Audio kept before the start of speech.
            max_buffered_speech: The longest speech kept for the end-of-speech event.
            activation_threshold: The probability above which a window is speech.
            deactivation_threshold: The probability below which speech ends;
                ``activation_threshold - 0.15`` by default.
        """
        self._native = _core.InferenceVad(
            silero_model,
            model=model,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            deactivation_threshold=deactivation_threshold if is_given(deactivation_threshold) else None,
        )
        super().__init__(capabilities=self._native.capabilities)

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, VAD, "stream")
        return self if replaced else self._native

    @property
    def model(self) -> VADModels:
        return self._native.model

    @property
    def min_silence_duration(self) -> float:
        return self._native.min_silence_duration

    def update_options(
        self,
        *,
        min_speech_duration: NotGivenOr[float] = NOT_GIVEN,
        min_silence_duration: NotGivenOr[float] = NOT_GIVEN,
        prefix_padding_duration: NotGivenOr[float] = NOT_GIVEN,
        max_buffered_speech: NotGivenOr[float] = NOT_GIVEN,
        activation_threshold: NotGivenOr[float] = NOT_GIVEN,
        deactivation_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Update the options; running streams apply them before their next frame."""
        self._native.update_options(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            deactivation_threshold=deactivation_threshold,
        )

    def stream(self) -> _VADStream:
        return _VADStream(vad=cast("inference.VAD", self))


class _VADStream(VADStream):
    """`inference.VAD.stream()`: the native detection, frames and flushes in, its events out."""

    _vad: inference.VAD

    def __init__(self, *, vad: inference.VAD) -> None:
        super().__init__(vad=vad)

    async def _core_open(self) -> _core.InferenceVadStream:
        return self._vad._native.stream()

    _run = VADStream._core_run_native


_VADStream.__module__ = "livekit.agents.inference"
