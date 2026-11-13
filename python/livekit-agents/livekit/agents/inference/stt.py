from __future__ import annotations

from typing import Any

from .. import stt
from .._generated import facades_inference_stt as _facades, public_inference_stt as _public
from .._generated.inference_stt import FallbackModel as FallbackModel
from .._generated.public_inference_stt import *  # noqa: F403

__all__ = [
    "STT",
    "FallbackModel",
]
__all__ += _public.__all__


class STT(_facades.STT):
    """LiveKit Inference STT."""

    @classmethod
    def from_model_string(cls, model: str) -> STT:
        """An STT from ``"provider/model[:language]"``."""
        return cls(model)

    @property
    def provider(self) -> str:
        return "livekit"

    async def _recognize_impl(self, buffer: Any, **_: Any) -> stt.SpeechEvent:
        raise NotImplementedError(
            "LiveKit Inference STT does not support batch recognition, use stream() instead"
        )

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self._core_update_keyterms(keyterms)

    def _push_conversation_item(self, role: str, text: str) -> None:
        self._core_push_conversation_item(role, text)
