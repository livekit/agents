from __future__ import annotations

from typing import Any

from .. import _core, stt
from .._generated import facades_inference_stt as _facades, public_inference_stt as _public
from .._generated.inference_stt import FallbackModel as FallbackModel
from .._generated.public_inference_stt import *  # noqa: F403
from ..types import APIConnectOptions
from . import get_inference_headers

__all__ = [
    "STT",
    "FallbackModel",
]
__all__ += _public.__all__


class STT(_facades.STT):
    """LiveKit Inference STT over the core's gateway client (``_core.InferenceStt``)."""

    @classmethod
    def from_model_string(cls, model: str) -> STT:
        """An STT from ``"provider/model[:language]"``."""
        return cls(model)

    @property
    def provider(self) -> str:
        return "livekit"

    def _core_conn_options(self) -> APIConnectOptions | None:
        """The policy this provider was given, which a session or run that supplies none
        keeps (models.md)."""
        return self._native.conn_options

    def _core_native(self) -> _core.InferenceStt:
        """The client as a session of the current job holds it: its connections carry that job."""
        return self._native.identified(list(get_inference_headers().items()), self)

    async def _recognize_impl(self, buffer: Any, **_: Any) -> stt.SpeechEvent:
        raise NotImplementedError(
            "LiveKit Inference STT does not support batch recognition, use stream() instead"
        )

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self._native.update_keyterms(keyterms)

    def _push_conversation_item(self, role: str, text: str) -> None:
        self._native.push_conversation_item(role, text)
