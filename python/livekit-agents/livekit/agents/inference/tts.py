from __future__ import annotations

from .. import _core, tts
from .._generated import facades_inference_tts as _facades, public_inference_tts as _public
from .._generated.inference_tts import FallbackModel as FallbackModel
from .._generated.public_inference_tts import *  # noqa: F403

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
