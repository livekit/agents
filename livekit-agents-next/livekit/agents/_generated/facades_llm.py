# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .. import _bindings, _core
from .._bindings import core_legs
from ..log import logger
from . import llm_availability
from .llm import LLM

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .chat_items import MetricsMetadata
    from .llm_fallback import AvailabilityChanged


class FallbackAdapter(LLM[Literal["llm_availability_changed"]]):
    """`llm.FallbackAdapter`: several LLMs behind one; a failed one is skipped until it recovers."""

    if TYPE_CHECKING:
        _members: list[LLM[Any]]

    def __init__(
        self,
        llm: Sequence[LLM[Any]],
        *,
        attempt_timeout: float = 5.0,
        max_retry_per_llm: int = 0,
        retry_interval: float = 0.5,
        sticky: bool = False,
    ) -> None:
        """Several LLMs behind one: a failed one is skipped until it recovers.

        Args:
            llm: The LLMs, in order of preference.
            attempt_timeout: The timeout of one attempt.
            max_retry_per_llm: Retries of each before moving to the next.
            retry_interval: The wait between retries.
            sticky: Keep the LLM that last served until it fails, even once a preferred one
                recovers; on failure the rest are tried in order.
        """
        super().__init__()
        self._native = _core.FallbackLlm(
            core_legs(llm),
            attempt_timeout=attempt_timeout,
            max_retry_per_llm=max_retry_per_llm,
            retry_interval=retry_interval,
            sticky=sticky,
            on_availability=self,
        )
        self._members = list(llm)

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, FallbackAdapter, "chat")
        return self if replaced else self._native

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None:
        self.emit(
            "llm_availability_changed",
            llm_availability.AvailabilityChangedEvent(self._members[ev.index], ev.available),
        )

    def _core_all_unavailable(self) -> None:
        logger.error("all LLMs are unavailable, retrying..")

    @property
    def _active(self) -> LLM[Any]:
        return self._members[self._native.active_index]

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
        await super().aclose()
        await self._native.aclose()
