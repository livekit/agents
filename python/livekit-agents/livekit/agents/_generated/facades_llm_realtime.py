# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .. import _bindings, _core
from ..log import logger
from . import realtime_availability
from .realtime import RealtimeModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .chat_items import MetricsMetadata
    from .llm_fallback import AvailabilityChanged


class RealtimeModelFallbackAdapter(_core.RealtimeFallbackModel, RealtimeModel[Literal["realtime_availability_changed"]]):
    """`llm.RealtimeModelFallbackAdapter`: falls back between realtime models (or restarts one),
    preserving chat context and handlers. A failed model is skipped until its cooldown ends, the
    conversation is replayed onto the next one, and a reply cut by the swap is asked for again.
    """

    if TYPE_CHECKING:
        _members: list[RealtimeModel]

        def _core_failed(self, error: BaseException, recoverable: bool) -> None: ...

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _bindings.refuse_overrides(cls, ("session",), "session.llm_node")

    def __init__(
        self,
        models: Sequence[RealtimeModel],
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
        _core.RealtimeFallbackModel.__init__(
            self,
            models,
            cooldown=cooldown,
            attempt_timeout=attempt_timeout,
            regenerate_on_swap=regenerate_on_swap,
        )
        self._members = list(models)
        RealtimeModel.__init__(self, capabilities=self.capabilities)

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None:
        self.emit(
            "realtime_availability_changed",
            realtime_availability.RealtimeAvailabilityChangedEvent(self._members[ev.index], ev.available),
        )

    def _core_all_unavailable(self) -> None:
        logger.error("all RealtimeModels are unavailable, retrying..")

    @property
    def _active(self) -> RealtimeModel:
        return self._members[self._core_active_index]

    @property
    def model(self) -> str:
        return self._active.model

    @property
    def provider(self) -> str:
        return self._active.provider

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        return self._active.metrics_metadata

    async def aclose(self) -> None:
        await RealtimeModel.aclose(self)
        for member in self._members:
            await member.aclose()

    async def restart_session(self, *, switch_model: bool = False) -> None:
        """Bring up a fresh underlying session, preserving chat context and bound handlers.

        Args:
            switch_model: Bring the new session up on the next available model instead of the
                current one.
        """
        await _core.RealtimeFallbackModel._core_restart(self, switch_model=switch_model)
