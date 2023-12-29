from ..vad import VAD
from ..utils import AudioBuffer
from .stt import STT, RecognizeOptions, SpeechStream, StreamOptions
from typing import Optional
from enum import Enum


class StreamAdapter(STT):
    def __init__(self, *, vad: VAD, stt: STT) -> None:
        super().__init__(streaming_supported=True)
        self._vad = vad
        self._stt = stt

    async def recognize(
        self, buffer: AudioBuffer, opts: RecognizeOptions = RecognizeOptions()
    ):
        return await self._stt.recognize(buffer, opts)

    def stream(self, opts: StreamOptions = StreamOptions()) -> SpeechStream:
        return self._stt.stream(opts)


class VADStream(SpeechStream):
    pass
