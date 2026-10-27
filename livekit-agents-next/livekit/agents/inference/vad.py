from __future__ import annotations

import asyncio
from concurrent.futures import Executor
from typing import Any

from .. import _bindings, _core
from .._generated import vad
from .._generated.inference_vad import VADModels
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import aio, is_given

__all__ = ["VAD", "VADModels"]


def _silero_model() -> Any:
    # the host's model, which the core asks for only when the build links no native one
    from livekit.local_inference import VAD

    return VAD()


class VAD(vad.VAD):
    """Silero voice activity detection, the detection machine in the core (``_core.InferenceVad``)
    and the model native to the core when the build links it, else ``livekit-local-inference``'s."""

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
        executor: Executor | None = None,
    ) -> None:
        """
        Args:
            model: The VAD model; only ``"silero"``.
            min_speech_duration: Speech needed before a start of speech.
            min_silence_duration: Silence needed before an end of speech.
            prefix_padding_duration: Audio kept before the start of speech.
            max_buffered_speech: The longest speech kept for the end-of-speech event.
            activation_threshold: The probability above which a window is speech.
            deactivation_threshold: The probability below which speech ends;
                ``activation_threshold - 0.15`` by default.
            executor: Unused; the core runs the model.
        """
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.032))
        if model != "silero":
            raise ValueError(f"Unknown VAD model: {model!r}. Supported: 'silero'.")
        if is_given(deactivation_threshold) and deactivation_threshold <= 0:
            raise ValueError("deactivation_threshold must be greater than 0")
        self._model = model
        self._opts = {
            "min_speech_duration": min_speech_duration,
            "min_silence_duration": min_silence_duration,
            "prefix_padding_duration": prefix_padding_duration,
            "max_buffered_speech": max_buffered_speech,
            "activation_threshold": activation_threshold,
            "deactivation_threshold": deactivation_threshold
            if is_given(deactivation_threshold)
            else max(activation_threshold - 0.15, 0.01),
        }
        self._native = _core.InferenceVad(_silero_model, **self._opts)

    def _core_leg(self) -> Any:
        """The core's own detector, unless a subclass replaces its stream."""
        return self if _bindings.overrides(self, VAD, "stream") else self._native

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "livekit-local-inference"

    @property
    def min_silence_duration(self) -> float:
        return self._opts["min_silence_duration"]

    def stream(self) -> vad.VADStream:
        return _VADStream(self)

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
        given = {
            "min_speech_duration": min_speech_duration,
            "min_silence_duration": min_silence_duration,
            "prefix_padding_duration": prefix_padding_duration,
            "max_buffered_speech": max_buffered_speech,
            "activation_threshold": activation_threshold,
            "deactivation_threshold": deactivation_threshold,
        }
        self._opts.update({k: v for k, v in given.items() if is_given(v)})
        self._native.update_options(**self._opts)


class _VADStream(vad.VADStream):
    def __init__(self, parent: VAD) -> None:
        self._native = parent._native.stream()
        super().__init__(parent)

    async def _main_task(self) -> None:
        native = self._native

        async def _push() -> None:
            async for item in self.input:
                if isinstance(item, self.Flush):
                    native.flush()
                else:
                    native.push(item)
            native.end_input()

        push = asyncio.create_task(_push())
        try:
            while (ev := await native.recv()) is not None:
                if ev.pop("slow_inference"):
                    logger.warning(
                        "inference is slower than realtime",
                        extra={"delay": ev["extra_inference_time"]},
                    )
                del ev["extra_inference_time"]
                ev["type"] = vad.VADEventType(ev["type"])
                self.send(vad.VADEvent(**ev))
        finally:
            await aio.cancel_and_wait(push)
            await native.aclose()
