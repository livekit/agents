# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core
from .._bindings import core_legs
from . import llm_availability

if TYPE_CHECKING:
    from livekit.agents.llm import LLM

    from .llm_fallback import AvailabilityChanged


class FallbackAdapter:
    """`llm.FallbackAdapter`'s chain, built once per loop from the adapter's attributes; a failed LLM
    is skipped until it recovers.
    """

    if TYPE_CHECKING:
        _members: list[LLM[Any]]
        _attempt_timeout: float
        _max_retry_per_llm: int
        _retry_interval: float
        _sticky: bool

        def _core_all_unavailable(self) -> None: ...

        def emit(self, event: str, *args: Any) -> None: ...

    @property
    def _native(self) -> _core.FallbackLlm:
        return _bindings.on_loop(
            self,
            lambda: _core.FallbackLlm(
                core_legs(self._members),
                attempt_timeout=self._attempt_timeout,
                max_retry_per_llm=self._max_retry_per_llm,
                retry_interval=self._retry_interval,
                sticky=self._sticky,
                on_availability=self,
            ),
        )

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, FallbackAdapter, "chat")
        return self if replaced else self._native

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None:
        self.emit(
            "llm_availability_changed",
            llm_availability.AvailabilityChangedEvent(self._members[ev.index], ev.available),
        )
