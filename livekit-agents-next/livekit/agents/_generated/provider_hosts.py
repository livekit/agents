# GENERATED FILE -- DO NOT EDIT.
# From #[host(new, facade)] declarations in lk-agents-abi.

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from .. import _bindings, _core
from .._bindings import core_leg, core_legs
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from . import llm_availability, stt_availability, tts_availability
from .inference_llm import LLMModels
from .llm_provider_options import ChatCompletionOptions

if TYPE_CHECKING:
    from livekit.agents import VAD
    from livekit.agents.llm import LLM
    from livekit.agents.stt import STT
    from livekit.agents.tts import TTS

    from .connect import APIConnectOptions
    from .inference_stt import FallbackModel as HostFallbackModel
    from .inference_tts import FallbackModel
    from .language import LanguageCode
    from .llm_fallback import AvailabilityChanged


class InferenceTtsHost:
    if TYPE_CHECKING:
        _model: str
        _base_url: str
        _api_key: str
        _api_secret: str
        _conn_options: APIConnectOptions
        _voice: str | None
        _language: LanguageCode | None
        _encoding: str
        _sample_rate: int
        _extra_kwargs: dict[str, Any]
        _fallback: list[FallbackModel]

        def _core_native(self) -> _core.InferenceTts: ...

    @functools.cached_property
    def _native(self) -> _core.InferenceTts:
        return _core.InferenceTts(
            self._model,
            base_url=self._base_url,
            api_key=self._api_key,
            api_secret=self._api_secret,
            conn_options=self._conn_options,
            voice=self._voice,
            language=self._language,
            encoding=self._encoding,
            sample_rate=self._sample_rate,
            extra_kwargs=self._extra_kwargs,
            fallback=self._fallback,
        )

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, InferenceTtsHost, "synthesize", "stream")
        return self if replaced else self._core_native()


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


class InferenceLlmHost:
    if TYPE_CHECKING:
        _model: str
        _provider: str | None
        _base_url: str
        _api_key: str
        _api_secret: str | None
        _label: str
        _inference_class: str | None
        _strict_tool_schema: bool
        _gzip_body: bool
        _msgpack_body: bool
        _extra_kwargs: dict[str, object]

        def _core_native(self) -> _core.InferenceLlm: ...

    @functools.cached_property
    def _native(self) -> _core.InferenceLlm:
        return _core.InferenceLlm(
            self._model,
            provider=self._provider,
            base_url=self._base_url,
            api_key=self._api_key,
            api_secret=self._api_secret,
            label=self._label,
            inference_class=self._inference_class,
            strict_tool_schema=self._strict_tool_schema,
            gzip_body=self._gzip_body,
            msgpack_body=self._msgpack_body,
            extra_kwargs=self._extra_kwargs,
        )

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, InferenceLlmHost, "chat")
        return self if replaced else self._core_native()

    @property
    def model(self) -> str:
        return self._native.model

    def update_options(
        self,
        *,
        model: NotGivenOr[LLMModels | str] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[ChatCompletionOptions | dict[str, Any]] = NOT_GIVEN,
    ) -> None:
        """Read by the next ``chat()``. ``extra_kwargs`` replaces the persistent kwargs."""
        return self._native.update_options(
            model=model if is_given(model) else None,
            extra_kwargs=dict(extra_kwargs) if is_given(extra_kwargs) else None,
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


class InferenceSttHost:
    if TYPE_CHECKING:
        _model: str | None
        _language: LanguageCode | None
        _encoding: str
        _sample_rate: int
        _base_url: str
        _api_key: str
        _api_secret: str
        _extra_kwargs: dict[str, Any]
        _fallback: list[HostFallbackModel]
        _conn_options: APIConnectOptions
        _vad: VAD[Any] | None

        def _core_native(self) -> _core.InferenceStt: ...

    @functools.cached_property
    def _native(self) -> _core.InferenceStt:
        return _core.InferenceStt(
            model=self._model,
            language=self._language,
            encoding=self._encoding,
            sample_rate=self._sample_rate,
            base_url=self._base_url,
            api_key=self._api_key,
            api_secret=self._api_secret,
            extra_kwargs=self._extra_kwargs,
            fallback=self._fallback,
            conn_options=self._conn_options,
            vad=self._vad,
        )

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, InferenceSttHost, "stream")
        return self if replaced else self._core_native()


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
