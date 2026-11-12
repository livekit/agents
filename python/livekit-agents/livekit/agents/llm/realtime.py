from __future__ import annotations

from .. import _bindings, _core
from .._generated import facades_llm_realtime as _facades, public_llm_realtime as _public
from .._generated.public_llm_realtime import *  # noqa: F403
from .._generated.realtime import EventTypes, RealtimeModelError


class RealtimeModelFallbackAdapter(_facades.RealtimeModelFallbackAdapter):
    """Falls back between realtime models (or restarts one), preserving chat context and handlers.

    A failed model is skipped until its cooldown ends, the conversation is replayed onto the next
    one, and a reply cut by the swap is requested again.
    """

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        raise NotImplementedError(
            "RealtimeModelFallbackAdapter sessions run inside a VoiceSession (the core swaps them)"
        )

    def _core_leg(self) -> _core.RealtimeFallbackModel:
        # re-take each member's leg so inference members identify themselves as the current job's
        _bindings.core_legs(self._members)
        return self._native


__all__ = [
    "EventTypes",
    "RealtimeModelError",
    "RealtimeModelFallbackAdapter",
]
__all__ += _public.__all__
