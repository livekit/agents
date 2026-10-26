"""The STT adapters: the fallback chain is the core's (``_core.FallbackStt``)."""

from __future__ import annotations

import asyncio
import functools
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .. import _core, utils
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..vad import VAD, VADEventType
from .stt import STT, RecognizeStream, SpeechEvent, SpeechEventType, STTCapabilities

if TYPE_CHECKING:
    from ..llm.chat_context import MetricsMetadata
    from ..voice.events import ConversationItemAddedEvent

__all__ = [
    "AvailabilityChangedEvent",
    "FallbackAdapter",
    "StreamAdapter",
    "StreamAdapterWrapper",
]


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
        return _core.FallbackStt(self._stt_instances, on_availability=self, **self._opts)

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

    def _push_conversation_item(self, ev: ConversationItemAddedEvent) -> None:
        for stt in self._stt_instances:
            stt._push_conversation_item(ev)

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        frames = buffer if isinstance(buffer, list) else [buffer]
        event = await self._native.recognize(frames, language if language else None)
        return SpeechEvent._from_core(event)

    async def recognize(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        # the chain retries each member itself
        return await self._recognize_impl(buffer, language=language, conn_options=conn_options)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return FallbackRecognizeStream(stt=self, language=language)

    def prewarm(self) -> None:
        self._stt_instances[0].prewarm()

    async def aclose(self) -> None:
        if "_native" in vars(self):
            await self._native.aclose()


class FallbackRecognizeStream(RecognizeStream):
    _stt: FallbackAdapter

    def __init__(self, *, stt: FallbackAdapter, language: NotGivenOr[str]) -> None:
        # the chain retries each member itself
        super().__init__(stt=stt, conn_options=APIConnectOptions(max_retry=0))
        self._language = language if language else None

    async def _run(self) -> None:
        native = await self._stt._native.stream(self._language, self.start_time_offset)

        async def _push() -> None:
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    native.flush()
                else:
                    native.push_frame(item)
            native.end_input()

        push = asyncio.create_task(_push())
        try:
            while (ev := await native.recv()) is not None:
                self._event_ch.send_nowait(SpeechEvent._from_core(ev))
        finally:
            await utils.aio.cancel_and_wait(push)
            await native.aclose()


class StreamAdapter(STT):
    """A recognize-only STT as a streaming one: the VAD cuts the audio into utterances, each
    recognized on its own."""

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

    def _push_conversation_item(self, item: ConversationItemAddedEvent) -> None:
        self._stt._push_conversation_item(item)

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
        return StreamAdapterWrapper(self, language=language, conn_options=conn_options)

    def prewarm(self) -> None:
        self._stt.prewarm()


class StreamAdapterWrapper(RecognizeStream):
    def __init__(
        self, stt: StreamAdapter, *, language: NotGivenOr[str], conn_options: APIConnectOptions
    ) -> None:
        # the wrapped STT's recognize() retries, so the stream does not
        super().__init__(stt=stt, conn_options=APIConnectOptions(max_retry=0))
        self._adapter = stt
        self._wrapped_conn_options = conn_options
        self._language = language

    async def _run(self) -> None:
        vad_stream = self._adapter._vad.stream()

        async def _forward_input() -> None:
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    vad_stream.flush()
                else:
                    vad_stream.push_frame(item)
            vad_stream.end_input()

        async def _recognize() -> None:
            async for event in vad_stream:
                if event.type == VADEventType.START_OF_SPEECH:
                    self._event_ch.send_nowait(SpeechEvent(SpeechEventType.START_OF_SPEECH))
                    continue
                if event.type != VADEventType.END_OF_SPEECH:
                    continue
                end = time.time() - event.silence_duration - event.inference_duration
                self._event_ch.send_nowait(
                    SpeechEvent(type=SpeechEventType.END_OF_SPEECH, speech_end_time=end)
                )
                recognized = await self._adapter._stt.recognize(
                    buffer=utils.merge_frames(event.frames),
                    language=self._language,
                    conn_options=self._wrapped_conn_options,
                )
                if recognized.alternatives and recognized.alternatives[0].text:
                    self._event_ch.send_nowait(
                        SpeechEvent(
                            type=SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[recognized.alternatives[0]],
                            speech_end_time=recognized.speech_end_time or end,
                        )
                    )

        tasks = [asyncio.create_task(_forward_input()), asyncio.create_task(_recognize())]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await vad_stream.aclose()
