from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from typing import Any

from .. import utils
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..vad import VAD, VADEventType
from .stt import STT, RecognizeStream, SpeechEvent, SpeechEventType, STTCapabilities

# already a retry mechanism in STT.recognize, don't retry in stream adapter
DEFAULT_STREAM_ADAPTER_API_CONNECT_OPTIONS = APIConnectOptions(
    max_retry=0, timeout=DEFAULT_API_CONNECT_OPTIONS.timeout
)


class StreamAdapter(STT):
    def __init__(self, *, stt: STT, vad: VAD) -> None:
        super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=False))
        self._vad = vad
        self._stt = stt

        @self._stt.on("metrics_collected")
        def _forward_metrics(*args: Any, **kwargs: Any) -> None:
            self.emit("metrics_collected", *args, **kwargs)

    @property
    def wrapped_stt(self) -> STT:
        return self._stt

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
            self,
            vad=self._vad,
            wrapped_stt=self._stt,
            language=language,
            conn_options=conn_options,
        )


class StreamAdapterWrapper(RecognizeStream):
    def __init__(
        self,
        stt: STT,
        *,
        vad: VAD,
        wrapped_stt: STT,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=DEFAULT_STREAM_ADAPTER_API_CONNECT_OPTIONS)
        self._vad = vad
        self._wrapped_stt = wrapped_stt
        self._wrapped_stt_conn_options = conn_options
        self._language = language

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SpeechEvent]) -> None:
        pass  # do nothing

    async def _run(self) -> None:
        vad_stream = self._vad.stream()

        async def _forward_input() -> None:
            """forward input to vad"""
            async for input in self._input_ch:
                if isinstance(input, self._FlushSentinel):
                    vad_stream.flush()
                    continue
                vad_stream.push_frame(input)

            vad_stream.end_input()

        async def _recognize() -> None:
            """recognize speech from vad"""
            async for event in vad_stream:
                if event.type == VADEventType.START_OF_SPEECH:
                    self._event_ch.send_nowait(SpeechEvent(SpeechEventType.START_OF_SPEECH))
                elif event.type == VADEventType.END_OF_SPEECH:
                    self._event_ch.send_nowait(
                        SpeechEvent(
                            type=SpeechEventType.END_OF_SPEECH,
                        )
                    )

                    merged_frames = utils.merge_frames(event.frames)
                    t_event = await self._wrapped_stt.recognize(
                        buffer=merged_frames,
                        language=self._language,
                        conn_options=self._wrapped_stt_conn_options,
                    )

                    if len(t_event.alternatives) == 0:
                        continue
                    elif not t_event.alternatives[0].text:
                        continue

                    self._event_ch.send_nowait(
                        SpeechEvent(
                            type=SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[t_event.alternatives[0]],
                        )
                    )

        tasks = [
            asyncio.create_task(_forward_input(), name="forward_input"),
            asyncio.create_task(_recognize(), name="recognize"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await vad_stream.aclose()
