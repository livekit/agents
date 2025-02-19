"""
Adapter for adding Voice Activity Detection (VAD) to non-streaming STT implementations.

Enables real-time speech recognition by:
1. Using VAD to detect speech segments
2. Aggregating audio during speech activity
3. Passing speech segments to base STT for recognition
4. Managing streaming lifecycle and event forwarding
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterable

from .. import utils
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from ..vad import VAD, VADEventType
from .stt import STT, RecognizeStream, SpeechEvent, SpeechEventType, STTCapabilities


class StreamAdapter(STT):
    """Adds streaming capabilities to non-streaming STT using VAD."""
    
    def __init__(self, *, stt: STT, vad: VAD) -> None:
        """
        Args:
            stt: Base STT implementation to wrap
            vad: Voice Activity Detector for speech segmentation
        """
        super().__init__(
            capabilities=STTCapabilities(streaming=True, interim_results=False)
        )
        self._vad = vad
        self._stt = stt

        # Forward metrics from underlying STT
        @self._stt.on("metrics_collected")
        def _forward_metrics(*args, **kwargs):
            self.emit("metrics_collected", *args, **kwargs)

    @property
    def wrapped_stt(self) -> STT:
        """Get the wrapped STT instance."""
        return self._stt

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        """Proxy recognition to underlying STT."""
        return await self._stt.recognize(
            buffer=buffer, language=language, conn_options=conn_options
        )

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        """Create a VAD-managed streaming session."""
        return StreamAdapterWrapper(
            self,
            vad=self._vad,
            wrapped_stt=self._stt,
            language=language,
            conn_options=conn_options,
        )


class StreamAdapterWrapper(RecognizeStream):
    """Streaming implementation that coordinates VAD and STT."""
    
    def __init__(
        self,
        stt: STT,
        *,
        vad: VAD,
        wrapped_stt: STT,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> None:
        """
        Args:
            stt: Parent adapter instance
            vad: Voice activity detector
            wrapped_stt: Base STT for recognition
            language: Transcription language hint
            conn_options: Connection/retry configuration
        """
        super().__init__(stt=stt, conn_options=conn_options)
        self._vad = vad
        self._wrapped_stt = wrapped_stt
        self._vad_stream = self._vad.stream()
        self._language = language

    async def _metrics_monitor_task(
        self, event_aiter: AsyncIterable[SpeechEvent]
    ) -> None:
        """Metrics handled by wrapped STT - no action needed."""
        pass

    async def _run(self) -> None:
        """Coordinate audio processing and recognition."""
        async def _forward_input():
            """Route audio input to VAD system."""
            async for input in self._input_ch:
                if isinstance(input, self._FlushSentinel):
                    self._vad_stream.flush()
                    continue
                self._vad_stream.push_frame(input)

            self._vad_stream.end_input()

        async def _recognize():
            """Handle VAD events and trigger STT recognition."""
            async for event in self._vad_stream:
                if event.type == VADEventType.START_OF_SPEECH:
                    # Notify speech start
                    self._event_ch.send_nowait(
                        SpeechEvent(SpeechEventType.START_OF_SPEECH)
                    )
                elif event.type == VADEventType.END_OF_SPEECH:
                    # Notify speech end
                    self._event_ch.send_nowait(
                        SpeechEvent(type=SpeechEventType.END_OF_SPEECH)
                    )

                    # Process captured audio
                    merged_frames = utils.merge_frames(event.frames)
                    t_event = await self._wrapped_stt.recognize(
                        buffer=merged_frames,
                        language=self._language,
                        conn_options=self._conn_options,
                    )

                    # Forward valid results                    
                    if len(t_event.alternatives) == 0:
                        continue
                    elif not t_event.alternatives[0].text:
                        continue

                    self._event_ch.send_nowait(
                        SpeechEvent(
                            type=SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[t_event.alternatives[0]],
                        )

        # Manage concurrent tasks
        tasks = [
            asyncio.create_task(_forward_input(), name="forward_input"),
            asyncio.create_task(_recognize(), name="recognize"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            # Cleanup on exit
            await utils.aio.cancel_and_wait(*tasks)
