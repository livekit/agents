from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from livekit import rtc

from .. import _core
from .._generated import public_llm_realtime as _public
from .._generated.chat_items import MetricsMetadata
from .._generated.public_llm_realtime import *  # noqa: F403
from .._generated.realtime import EventTypes, RealtimeModelError

if TYPE_CHECKING:
    from .._generated.llm_fallback import AvailabilityChanged

# Event data and request exceptions share their ABI declarations with the native adapters.


def _merge_capabilities(models: list[RealtimeModel]) -> RealtimeCapabilities:
    """The pipeline-shaping capabilities must agree; the others are the conservative AND."""
    return _core.merge_realtime_capabilities([m.capabilities for m in models])


class RealtimeModelFallbackAdapter(
    RealtimeModel,
    rtc.EventEmitter[Literal["realtime_availability_changed"]],
):
    """Falls back between realtime models (or restarts one), preserving chat context and handlers.

    The swap runs in the core (``_core.RealtimeFallbackModel``): a failed model is skipped until its
    cooldown ends, the conversation is replayed onto the next one, and a reply cut by the swap is
    asked for again.
    """

    def __init__(
        self,
        models: list[RealtimeModel],
        *,
        cooldown: float = 10.0,
        regenerate_on_swap: bool = True,
        attempt_timeout: float | None = None,
    ) -> None:
        """Fall back between realtime models while preserving chat context.

        Args:
            models: Ordered models; the first is primary, the rest fallbacks. All must agree on
                the ``audio_output`` and ``turn_detection`` capabilities.
            cooldown: Seconds a failed model stays unavailable before it can be preferred again.
            regenerate_on_swap: Re-issue the reply on the new session if one was in progress.
            attempt_timeout: Seconds one model's session may take to open before the next is
                tried; ``None`` waits for the model's own connection limits.

        Raises:
            ValueError: If no models are given or their hard capabilities disagree.
        """
        if len(models) < 1:
            raise ValueError("at least one RealtimeModel instance must be provided.")

        RealtimeModel.__init__(self, capabilities=_merge_capabilities(models))
        rtc.EventEmitter.__init__(self)
        self._models = models
        self._native = _core.RealtimeFallbackModel(
            [m._core_leg() for m in models],
            cooldown=cooldown,
            attempt_timeout=attempt_timeout,
            regenerate_on_swap=regenerate_on_swap,
            on_availability=self,
        )

    @property
    def model(self) -> str:
        return "RealtimeModelFallbackAdapter"

    @property
    def provider(self) -> str:
        return "livekit"

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        """Metadata of the model currently serving sessions (the primary until a swap)."""
        return self._models[self._native.active_index].metrics_metadata

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        raise NotImplementedError(
            "RealtimeModelFallbackAdapter sessions run inside a VoiceSession (the core swaps them)"
        )

    def _core_leg(self) -> _core.RealtimeFallbackModel:
        # each member's leg is taken again: an inference member identifies itself as the job's
        for model in self._models:
            model._core_leg()
        return self._native

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None:
        self.emit(
            "realtime_availability_changed",
            RealtimeAvailabilityChangedEvent(
                realtime_model=self._models[ev.index], available=ev.available
            ),
        )

    def _core_all_unavailable(self) -> None:
        pass

    async def restart_session(self, *, switch_model: bool = False) -> None:
        """Bring up a fresh underlying session, preserving chat context and bound handlers.

        Args:
            switch_model: Bring the new session up on the next available model instead of the
                current one.
        """
        await self._native.restart(switch_model=switch_model)

    async def aclose(self) -> None:
        for model in self._models:
            await model.aclose()


__all__ = [
    "EventTypes",
    "RealtimeModelError",
]
__all__ += _public.__all__
