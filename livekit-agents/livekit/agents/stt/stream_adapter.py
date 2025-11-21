from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any

from .. import utils
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..vad import VAD, VADEventType, VADStream
from .stt import STT, RecognizeStream, SpeechEvent, SpeechEventType, STTCapabilities

# already a retry mechanism in STT.recognize, don't retry in stream adapter
DEFAULT_STREAM_ADAPTER_API_CONNECT_OPTIONS = APIConnectOptions(
    max_retry=0, timeout=DEFAULT_API_CONNECT_OPTIONS.timeout
)


@dataclass
class StreamAdapterOptions:
    use_streaming: bool = False


class StreamAdapter(STT):
    def __init__(
        self,
        *,
        stt: STT,
        vad: VAD,
        use_streaming: bool = False,
    ) -> None:
        """
        Create a new instance of StreamAdapter.

        Args:
            stt: The STT to wrap.
            vad: The VAD to use.
            use_streaming: Whether to use streaming mode of the wrapped STT. Default is False.
        """
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=use_streaming,
                diarization=stt.capabilities.diarization and use_streaming,
            )
        )
        self._vad = vad
        self._stt = stt
        self._opts = StreamAdapterOptions(use_streaming=use_streaming)
        if use_streaming and not stt.capabilities.streaming:
            raise ValueError(
                f"STT {stt.label} does not support streaming while use_streaming is enabled"
            )
        if use_streaming and not stt.capabilities.flush:
            logger.warning(
                f"STT {stt.label} does not support flush while use_streaming is enabled, "
                "this may cause incomplete transcriptions."
            )

        # TODO(theomonnom): The segment_id needs to be populated!
        self._stt.on("metrics_collected", self._on_metrics_collected)

    @property
    def wrapped_stt(self) -> STT:
        return self._stt

    @property
    def model(self) -> str:
        return self._stt.model

    @property
    def provider(self) -> str:
        return self._stt.provider

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
            opts=self._opts,
        )

    def _on_metrics_collected(self, *args: Any, **kwargs: Any) -> None:
        self.emit("metrics_collected", *args, **kwargs)

    async def aclose(self) -> None:
        self._stt.off("metrics_collected", self._on_metrics_collected)


class StreamAdapterWrapper(RecognizeStream):
    def __init__(
        self,
        stt: STT,
        *,
        vad: VAD,
        wrapped_stt: STT,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
        opts: StreamAdapterOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=DEFAULT_STREAM_ADAPTER_API_CONNECT_OPTIONS)
        self._vad = vad
        self._wrapped_stt = wrapped_stt
        self._wrapped_stt_conn_options = conn_options
        self._language = language
        self._opts = opts

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

        async def _forward_stream_output(stream: RecognizeStream) -> None:
            async for event in stream:
                self._event_ch.send_nowait(event)

        stt_stream: RecognizeStream | None = None
        forward_input_task = asyncio.create_task(_forward_input(), name="forward_input")
        tasks = []
        if not self._opts.use_streaming:
            tasks.append(
                asyncio.create_task(
                    self._recognize_non_streaming(vad_stream), name="recognize_non_streaming"
                ),
            )
        else:
            stt_stream = self._wrapped_stt.stream(
                language=self._language, conn_options=self._wrapped_stt_conn_options
            )
            tasks += [
                asyncio.create_task(
                    _forward_stream_output(stt_stream), name="forward_stream_output"
                ),
                asyncio.create_task(
                    self._recognize_streaming(vad_stream, stt_stream),
                    name="recognize_streaming",
                ),
            ]

        try:
            await asyncio.gather(*tasks, forward_input_task)
        finally:
            await utils.aio.cancel_and_wait(forward_input_task)
            await vad_stream.aclose()
            if stt_stream is not None:
                stt_stream.end_input()
                await stt_stream.aclose()
            await utils.aio.cancel_and_wait(*tasks)

    async def _recognize_streaming(
        self, vad_stream: VADStream, stt_stream: RecognizeStream
    ) -> None:
        speaking = False
        async for event in vad_stream:
            frames = []
            if event.type == VADEventType.START_OF_SPEECH:
                speaking = True
                frames = event.frames
            elif event.type == VADEventType.INFERENCE_DONE and speaking:
                frames = event.frames
            elif event.type == VADEventType.END_OF_SPEECH:
                speaking = False
                stt_stream.flush()

            for f in frames:
                stt_stream.push_frame(f)

    async def _recognize_non_streaming(self, vad_stream: VADStream) -> None:
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
