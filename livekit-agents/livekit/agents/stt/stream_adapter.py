import asyncio
import logging
from ..vad import VADStream, VADEventType
from ..utils import AudioBuffer, merge_frames
from .stt import (
    STT,
    RecognizeOptions,
    SpeechStream,
    StreamOptions,
    SpeechEvent,
)
from livekit import rtc


class StreamAdapter(STT):
    def __init__(
        self,
        vad_stream: VADStream,
        stt: STT,
        recognize_options: RecognizeOptions = RecognizeOptions(),
    ) -> None:
        super().__init__(streaming_supported=True)
        self._vad_stream = vad_stream
        self._stt = stt
        self._recognize_options = recognize_options

    async def recognize(
        self, buffer: AudioBuffer, opts: RecognizeOptions = RecognizeOptions()
    ):
        return await self._stt.recognize(buffer, opts)

    def stream(self, opts: StreamOptions = StreamOptions()) -> SpeechStream:
        return AdapterStreamWrapper(
            self._vad_stream, self._stt, self._recognize_options, opts
        )


class AdapterStreamWrapper(SpeechStream):
    def __init__(
        self,
        vad_stream: VADStream,
        stt: STT,
        recognize_options: RecognizeOptions,
        stream_options: StreamOptions,
    ) -> None:
        super().__init__()
        self._vad_stream = vad_stream
        self._stt = stt
        self._recognize_options = recognize_options
        self._stream_options = stream_options
        self._event_queue = asyncio.Queue[SpeechEvent]()
        self._closed = False

        self._main_task = asyncio.create_task(self._run())

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(f"stream adapter task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)

    async def _run(self) -> None:
        # listen to vad events and send to stt on END_SPEAKING
        try:
            async for event in self._vad_stream:
                if event.type == VADEventType.END_SPEAKING:
                    merged_frames = merge_frames(event.speech)
                    event = await self._stt.recognize(
                        merged_frames, self._recognize_options
                    )
                    self._event_queue.put_nowait(event)
        except asyncio.CancelledError:
            pass

        self._closed = True

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self._vad_stream.push_frame(frame)

    async def close(self) -> None:
        await self._vad_stream.close()
        self._main_task.cancel()
        try:
            await self._main_task
        except asyncio.CancelledError:
            pass

    async def flush(self) -> None:
        await self._vad_stream.flush()

    async def __anext__(self) -> SpeechEvent:
        if self._closed and self._event_queue.empty():
            raise StopAsyncIteration

        return await self._event_queue.get()
