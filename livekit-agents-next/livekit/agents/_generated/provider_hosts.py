# GENERATED FILE -- DO NOT EDIT.
# From #[host(new, facade)] declarations in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core
from .._bindings import core_leg, core_legs
from . import llm_availability, stt_availability, tts_availability

if TYPE_CHECKING:
    from livekit.agents import VAD
    from livekit.agents.llm import LLM
    from livekit.agents.stt import STT
    from livekit.agents.tts import TTS

    from .llm_fallback import AvailabilityChanged


class FallbackLlmHost:
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
        replaced = _bindings.overrides(self, FallbackLlmHost, "chat")
        return self if replaced else self._native

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None:
        self.emit(
            "llm_availability_changed",
            llm_availability.AvailabilityChangedEvent(self._members[ev.index], ev.available),
        )


class FallbackTtsHost:
    if TYPE_CHECKING:
        _members: list[TTS[Any]]
        _max_retry_per_tts: int
        _attempt_timeout: float | None
        _sample_rate: int

        def _core_all_unavailable(self) -> None: ...

        def emit(self, event: str, *args: Any) -> None: ...

    @property
    def _native(self) -> _core.FallbackTts:
        return _bindings.on_loop(
            self,
            lambda: _core.FallbackTts(
                core_legs(self._members),
                max_retry_per_tts=self._max_retry_per_tts,
                attempt_timeout=self._attempt_timeout,
                sample_rate=self._sample_rate,
                on_availability=self,
            ),
        )

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, FallbackTtsHost, "synthesize", "stream")
        return self if replaced else self._native

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None:
        self.emit(
            "tts_availability_changed",
            tts_availability.AvailabilityChangedEvent(self._members[ev.index], ev.available),
        )


class FallbackSttHost:
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
            FallbackSttHost,
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


class SttStreamAdapterHost:
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
        replaced = _bindings.overrides(self, SttStreamAdapterHost, "stream", "recognize")
        return self if replaced else self._native
