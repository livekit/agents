"""Speech to text: the base classes a provider subclasses (generated from the core's
declarations) and the adapters, whose fallback chain is the core's (``_core.FallbackStt``)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from . import _bindings, _core, utils
from ._generated.llm_fallback import AvailabilityChanged
from ._generated.stt import (
    STT,
    RecognitionUsage,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    SpeechStream,
    STTCapabilities,
    STTError,
)
from ._generated.vad import VAD
from .log import logger
from .types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr

if TYPE_CHECKING:
    from .llm.chat_context import MetricsMetadata

__all__ = [
    "SpeechEventType",
    "SpeechEvent",
    "SpeechData",
    "RecognizeStream",
    "SpeechStream",
    "STT",
    "STTCapabilities",
    "StreamAdapter",
    "StreamAdapterWrapper",
    "RecognitionUsage",
    "FallbackAdapter",
    "AvailabilityChangedEvent",
    "STTError",
    "SpeakerContext",
]


# A structural type, which has no core counterpart to be generated from.
@runtime_checkable
class SpeakerContext(Protocol):
    """A speaker model an STT context provides, which the expressive system turns into LLM
    instructions."""

    def to_instructions(self) -> str: ...


def _speech_event_from_core(ev: dict[str, Any]) -> SpeechEvent:
    """An event the core emits as a dict; one the core minted is stamped as it is read."""
    alternatives = [SpeechData(**alt) for alt in ev.pop("alternatives")]
    usage = ev.pop("recognition_usage")
    if ev.get("created_at") is None:
        ev.pop("created_at", None)
    return SpeechEvent(
        **ev,
        alternatives=alternatives,
        recognition_usage=RecognitionUsage(**usage) if usage else None,
    )


@dataclass
class AvailabilityChangedEvent:
    stt: STT
    available: bool


class FallbackAdapter(STT[Literal["stt_availability_changed"]]):
    """Several STT behind one: a failed one is skipped until it recovers."""

    def __init__(
        self,
        stt: list[STT],
        *,
        vad: VAD | None = None,
        attempt_timeout: float = 10.0,
        max_retry_per_stt: int = 1,
        retry_interval: float = 5,
    ) -> None:
        """
        Args:
            stt: The STT, in order of preference.
            vad: Wraps the non-streaming ones in a ``StreamAdapter``; required when any is.
            attempt_timeout: The timeout of one attempt.
            max_retry_per_stt: Retries of each before moving to the next.
            retry_interval: The wait between retries.
        """
        if not stt:
            raise ValueError("At least one STT instance must be provided.")
        if non_streaming := [t.label for t in stt if not t.capabilities.streaming]:
            if vad is None:
                raise ValueError(
                    f"STTs do not support streaming: {', '.join(non_streaming)}. "
                    "Provide a VAD to enable stt.StreamAdapter automatically "
                    "or wrap them with stt.StreamAdapter before using this adapter."
                )
            adapted: dict[int, STT] = {}
            for recognizer in stt:
                identity = id(recognizer)
                if identity not in adapted:
                    adapted[identity] = (
                        recognizer
                        if recognizer.capabilities.streaming
                        else StreamAdapter(stt=recognizer, vad=vad)
                    )
            stt = [adapted[id(recognizer)] for recognizer in stt]
        self._stt_instances = stt
        self._attempt_timeout = attempt_timeout
        self._max_retry_per_stt = max_retry_per_stt
        self._retry_interval = retry_interval
        # the chain's own merge, read before any chain is built
        merged = _core.stt_fallback_capabilities([t.capabilities for t in stt])
        super().__init__(capabilities=STTCapabilities(**merged))

    @property
    def _native(self) -> _core.FallbackStt:
        # one per running loop: its members are called on the loop the chain was built on
        return _bindings.on_loop(
            self,
            lambda: _core.FallbackStt(
                [_bindings.core_leg(t) for t in self._stt_instances],
                attempt_timeout=self._attempt_timeout,
                max_retry_per_stt=self._max_retry_per_stt,
                retry_interval=self._retry_interval,
                on_availability=self,
            ),
        )

    def _core_leg(self) -> Any:
        if _bindings.overrides(self, FallbackAdapter, "stream", "recognize", "_recognize_impl"):
            return self
        return self._native

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None:
        self.emit(
            "stt_availability_changed",
            AvailabilityChangedEvent(stt=self._stt_instances[ev.index], available=ev.available),
        )

    def _core_all_unavailable(self) -> None:
        logger.error("all STTs are unavailable, retrying..")

    @property
    def _active(self) -> STT:
        chains = _bindings.built_on_loop(self)
        return self._stt_instances[chains[0].active_index if chains else 0]

    @property
    def model(self) -> str:
        return self._active.model

    @property
    def provider(self) -> str:
        return self._active.provider

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        return self._active.metrics_metadata

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        for stt in self._stt_instances:
            stt._update_session_keyterms(keyterms)

    def _push_conversation_item(self, role: str, text: str) -> None:
        for stt in self._stt_instances:
            stt._push_conversation_item(role, text)

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        frames = buffer if isinstance(buffer, list) else [buffer]
        event = await self._native.recognize(
            frames, language if language else None, conn_options=conn_options
        )
        return _speech_event_from_core(event)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return StreamAdapterWrapper(
            stt=self, native=self._native, language=language, conn_options=conn_options
        )

    def prewarm(self) -> None:
        self._stt_instances[0].prewarm()

    async def aclose(self) -> None:
        for chain in _bindings.built_on_loop(self):
            await chain.aclose()


class StreamAdapterWrapper(RecognizeStream):
    """A recognition stream the core runs: a ``FallbackAdapter``'s chain or a
    ``StreamAdapter``'s VAD and recognitions."""

    def __init__(
        self,
        *,
        stt: STT,
        native: _core.FallbackStt | _core.SttStreamAdapter,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> None:
        # the core's chain reconnects: the stream does not retry on top of it
        super().__init__(stt=stt, conn_options=replace(conn_options, max_retry=0))
        self._request_conn_options = conn_options
        self._native = native
        self._language = language if language else None

    async def _run(self) -> None:
        native = await self._native.stream(
            self._language, self.start_time_offset, conn_options=self._request_conn_options
        )
        await _bindings.pump(
            native,
            self.input,
            lambda item: native.flush() if isinstance(item, self.Flush) else native.push(item),
            lambda ev: self.send(_speech_event_from_core(ev)),
        )


class StreamAdapter(STT):
    """A recognize-only STT as a streaming one: the VAD cuts the audio into utterances, each
    recognized on its own (``_core.SttStreamAdapter``)."""

    def __init__(self, *, stt: STT, vad: VAD) -> None:
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=False,
                diarization=False,  # diarization requires streaming STT
                keyterms=stt.capabilities.keyterms,
                chat_context=stt.capabilities.chat_context,
            )
        )
        self._vad = vad
        self._stt = stt

    @property
    def _native(self) -> _core.SttStreamAdapter:
        # one per running loop, as the fallback chain's
        return _bindings.on_loop(
            self,
            lambda: _core.SttStreamAdapter(
                _bindings.core_leg(self._stt), _bindings.core_leg(self._vad)
            ),
        )

    def _core_leg(self) -> Any:
        if _bindings.overrides(self, StreamAdapter, "stream", "recognize"):
            return self
        return self._native

    @property
    def wrapped_stt(self) -> STT:
        return self._stt

    @property
    def model(self) -> str:
        return self._stt.model

    @property
    def provider(self) -> str:
        return self._stt.provider

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self._stt._update_session_keyterms(keyterms)

    def _push_conversation_item(self, role: str, text: str) -> None:
        self._stt._push_conversation_item(role, text)

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        return await self._stt.recognize(
            buffer=buffer, language=language, conn_options=conn_options
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return StreamAdapterWrapper(
            stt=self, native=self._native, language=language, conn_options=conn_options
        )

    def prewarm(self) -> None:
        self._stt.prewarm()
