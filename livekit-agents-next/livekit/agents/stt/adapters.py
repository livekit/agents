"""The STT adapters: the fallback chain is the core's (``_core.FallbackStt``)."""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .. import _core, utils
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..vad import VAD
from .stt import (
    STT,
    RecognitionUsage,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    STTCapabilities,
)

if TYPE_CHECKING:
    from ..llm.chat_context import MetricsMetadata

__all__ = [
    "AvailabilityChangedEvent",
    "FallbackAdapter",
    "StreamAdapter",
    "StreamAdapterWrapper",
]


def _speech_event_from_core(ev: dict[str, Any], *, offset: float = 0.0) -> SpeechEvent:
    """An event the core emits as a dict, its times shifted by ``offset``."""
    alternatives = []
    for alt in ev.pop("alternatives"):
        alt["start_time"] += offset
        alt["end_time"] += offset
        for word in alt["words"] or []:
            word.start_time = (word.start_time or 0.0) + offset
            word.end_time = (word.end_time or 0.0) + offset
            word.start_time_offset = offset
        alternatives.append(SpeechData(**alt))
    usage = ev.pop("recognition_usage")
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
            stt = [t if t.capabilities.streaming else StreamAdapter(stt=t, vad=vad) for t in stt]
        self._stt_instances = stt
        self._opts = {
            "attempt_timeout": attempt_timeout,
            "max_retry_per_stt": max_retry_per_stt,
            "retry_interval": retry_interval,
        }
        aligned = all(t.capabilities.aligned_transcript for t in stt)
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=all(t.capabilities.interim_results for t in stt),
                diarization=all(t.capabilities.diarization for t in stt),
                aligned_transcript=stt[0].capabilities.aligned_transcript if aligned else False,
                keyterms=any(t.capabilities.keyterms for t in stt),
                chat_context=any(t.capabilities.chat_context for t in stt),
            )
        )

    @functools.cached_property
    def _native(self) -> _core.FallbackStt:
        # built on first use: the chain lives on the running loop
        stts = [t._native if isinstance(t, StreamAdapter) else t for t in self._stt_instances]
        return _core.FallbackStt(stts, on_availability=self, **self._opts)

    def _core_availability_changed(self, ev: dict[str, Any]) -> None:
        self.emit(
            "stt_availability_changed",
            AvailabilityChangedEvent(
                stt=self._stt_instances[ev["index"]], available=ev["available"]
            ),
        )

    def _core_all_unavailable(self) -> None:
        logger.error("all STTs are unavailable, retrying..")

    @property
    def _active(self) -> STT:
        return self._stt_instances[self._native.active_index if "_native" in vars(self) else 0]

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
        event = await self._native.recognize(frames, language if language else None)
        return _speech_event_from_core(event)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return StreamAdapterWrapper(stt=self, native=self._native, language=language)

    def prewarm(self) -> None:
        self._stt_instances[0].prewarm()

    async def aclose(self) -> None:
        if "_native" in vars(self):
            await self._native.aclose()


class StreamAdapterWrapper(RecognizeStream):
    """A recognition stream the core runs: a ``FallbackAdapter``'s chain or a
    ``StreamAdapter``'s VAD and recognitions."""

    def __init__(
        self, *, stt: STT, native: _core.FallbackStt | _core.SttStreamAdapter, language: Any
    ) -> None:
        super().__init__(stt=stt, conn_options=DEFAULT_API_CONNECT_OPTIONS)
        self._native = native
        self._language = language if language else None

    async def _run(self) -> None:
        native = await self._native.stream(self._language, self.start_time_offset)

        async def _push() -> None:
            async for item in self.input:
                if isinstance(item, self.Flush):
                    native.flush()
                else:
                    native.push_frame(item)
            native.end_input()

        push = asyncio.create_task(_push())
        try:
            while (ev := await native.recv()) is not None:
                self.send(_speech_event_from_core(ev))
        finally:
            await utils.aio.cancel_and_wait(push)
            await native.aclose()


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

    @functools.cached_property
    def _native(self) -> _core.SttStreamAdapter:
        # built on first use, on the running loop
        return _core.SttStreamAdapter(self._stt, self._vad)

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
        return StreamAdapterWrapper(stt=self, native=self._native, language=language)

    def prewarm(self) -> None:
        self._stt.prewarm()
