# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core
from .._bindings import core_leg, core_legs
from . import stt_availability

if TYPE_CHECKING:
    from livekit.agents import VAD
    from livekit.agents.stt import STT

    from .llm_fallback import AvailabilityChanged


class FallbackAdapter:
    """`stt.FallbackAdapter`'s chain, built once per loop from the adapter's attributes; a failed STT
    is skipped until it recovers.
    """

    if TYPE_CHECKING:
        _members: list[STT[Any]]
        _attempt_timeout: float
        _max_retry_per_stt: int
        _retry_interval: float

        def _core_all_unavailable(self) -> None: ...

        def emit(self, event: str, *args: Any) -> None: ...

    @property
    def _native(self) -> _core.FallbackStt:
        return _bindings.on_loop(
            self,
            lambda: _core.FallbackStt(
                core_legs(self._members),
                attempt_timeout=self._attempt_timeout,
                max_retry_per_stt=self._max_retry_per_stt,
                retry_interval=self._retry_interval,
                on_availability=self,
            ),
        )

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(
            self,
            FallbackAdapter,
            "stream",
            "recognize",
            "_recognize_impl",
        )
        return self if replaced else self._native

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None:
        self.emit(
            "stt_availability_changed",
            stt_availability.AvailabilityChangedEvent(self._members[ev.index], ev.available),
        )


class StreamAdapter:
    """`stt.StreamAdapter`'s engine, built once per loop from the adapter's attributes."""

    if TYPE_CHECKING:
        _stt: STT[Any]
        _vad: VAD[Any]

    @property
    def _native(self) -> _core.SttStreamAdapter:
        return _bindings.on_loop(
            self,
            lambda: _core.SttStreamAdapter(
                core_leg(self._stt),
                core_leg(self._vad),
            ),
        )

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, StreamAdapter, "stream", "recognize")
        return self if replaced else self._native
