# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from .. import _bindings, _core
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN
from ..utils import is_given
from . import stt_availability
from .stt import STT, RecognizeStream

if TYPE_CHECKING:
    from collections.abc import Sequence

    from livekit.agents.vad import VAD

    from .. import stt
    from ..types import NotGivenOr
    from .chat_items import MetricsMetadata
    from .connect import APIConnectOptions
    from .llm_fallback import AvailabilityChanged


class FallbackAdapter(_core.FallbackStt, STT[Literal["stt_availability_changed"]]):
    """`stt.FallbackAdapter`: several STT behind one; a failed one is skipped until it recovers."""

    if TYPE_CHECKING:
        _members: list[STT[Any]]

        def _core_failed(self, error: BaseException, recoverable: bool) -> None: ...

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _bindings.refuse_overrides(
            cls,
            ("stream", "recognize", "_recognize_impl"),
            "session.stt_node",
        )

    def __init__(
        self,
        stt: Sequence[STT[Any]],
        *,
        vad: VAD[Any] | None = None,
        attempt_timeout: float = 10.0,
        max_retry_per_stt: int = 1,
        retry_interval: float = 5.0,
    ) -> None:
        """Several STT behind one: a failed one is skipped until it recovers.

        Args:
            stt: The STT, in order of preference.
            vad: Wraps the non-streaming ones in a ``StreamAdapter``; required when any is.
            attempt_timeout: The timeout of one attempt.
            max_retry_per_stt: Retries of each before moving to the next.
            retry_interval: The wait between retries.
        """
        _core.FallbackStt.__init__(
            self,
            stt,
            vad=vad,
            attempt_timeout=attempt_timeout,
            max_retry_per_stt=max_retry_per_stt,
            retry_interval=retry_interval,
        )
        self._members = list(stt)
        STT.__init__(self, capabilities=self.capabilities)

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None:
        self.emit(
            "stt_availability_changed",
            stt_availability.AvailabilityChangedEvent(self._members[ev.index], ev.available),
        )

    def _core_all_unavailable(self) -> None:
        logger.error("all STTs are unavailable, retrying..")

    @property
    def _active(self) -> STT[Any]:
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
        await STT.aclose(self)
        await _core.FallbackStt.aclose(self)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return FallbackRecognizeStream(
            stt=cast("stt.FallbackAdapter", self),
            language=language if is_given(language) else None,
            conn_options=conn_options,
        )


class FallbackRecognizeStream(RecognizeStream):
    """`stt.FallbackAdapter.stream()`: the chain's recognition, which reconnects and falls back."""

    _core_native_retries = True
    _stt: stt.FallbackAdapter

    def __init__(
        self,
        *,
        stt: stt.FallbackAdapter,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._language: str | None = language

    async def _core_open(self) -> _core.FallbackRecognizeStream:
        return await self._stt._core_native_stream(
            language=self._language,
            start_time_offset=self._start_time_offset,
            conn_options=self._conn_options,
        )

    _run = RecognizeStream._core_run_native


class StreamAdapter(_core.SttStreamAdapter, STT):
    """`stt.StreamAdapter`: a recognize-only STT as a streaming one."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _bindings.refuse_overrides(
            cls,
            ("stream", "recognize", "_recognize_impl"),
            "session.stt_node",
        )

    def __init__(self, *, stt: STT[Any], vad: VAD[Any]) -> None:
        """A recognize-only STT as a streaming one: the VAD cuts the audio into utterances, each
        recognized on its own.

        Args:
            stt: The recognize-only STT.
            vad: What cuts the audio into utterances.
        """
        _core.SttStreamAdapter.__init__(self, stt=stt, vad=vad)
        self._stt = stt
        self._vad = vad
        STT.__init__(self, capabilities=self.capabilities)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return StreamAdapterWrapper(
            stt=cast("stt.StreamAdapter", self),
            language=language if is_given(language) else None,
            conn_options=conn_options,
        )


class StreamAdapterWrapper(RecognizeStream):
    """`stt.StreamAdapter.stream()`: audio into the VAD, its utterances recognized, each retried on
    its own.
    """

    _core_native_retries = True
    _stt: stt.StreamAdapter

    def __init__(
        self,
        *,
        stt: stt.StreamAdapter,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._language: str | None = language

    async def _core_open(self) -> _core.SttAdapterStream:
        return await self._stt._core_native_stream(
            language=self._language,
            start_time_offset=self._start_time_offset,
            conn_options=self._conn_options,
        )

    _run = RecognizeStream._core_run_native


FallbackRecognizeStream.__module__ = "livekit.agents.stt"
StreamAdapterWrapper.__module__ = "livekit.agents.stt"
