from __future__ import annotations

from .. import _core, tts
from .._generated import facades_inference_tts as _facades, public_inference_tts as _public
from .._generated.inference_tts import FallbackModel as FallbackModel
from .._generated.public_inference_tts import *  # noqa: F403
from ..types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import is_given
from . import get_inference_headers

__all__ = [
    "TTS",
    "FallbackModel",
]
__all__ += _public.__all__


class TTS(_facades.TTS):
    """LiveKit Inference TTS."""

    class Markup(tts.TTS.Markup):
        def _provider_key(self) -> str:
            return _core.tts_markup_provider_key(self._tts.model)

    @classmethod
    def from_model_string(cls, model: str) -> TTS:
        """A TTS from ``"provider/model[:voice]"``."""
        return cls(model)

    @property
    def provider(self) -> str:
        return "livekit"

    def _core_conn_options(self) -> APIConnectOptions | None:
        return self._native.conn_options

    def _core_native(self) -> _core.InferenceTts:
        return self._native.identified(list(get_inference_headers().items()), self)

    def synthesize(
        self, text: str, *, conn_options: NotGivenOr[APIConnectOptions] = NOT_GIVEN
    ) -> tts.ChunkedStream:
        resolved = conn_options if is_given(conn_options) else self._native.conn_options
        return self._synthesize_with_stream(text, conn_options=resolved)
