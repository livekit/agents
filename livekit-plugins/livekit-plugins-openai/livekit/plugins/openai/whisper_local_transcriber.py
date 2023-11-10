import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import AsyncIterable, Optional

import whisper
from livekit import rtc
from livekit.plugins import core
import numpy as np


WHISPER_SAMPLE_RATE = 16000
WHISPER_CHANNELS = 1


class WhisperLocalTranscriber(core.STTPlugin):

    def __init__(self):
        self._model = None
        super().__init__(process=self.process)

    def process(self, frame_groups: AsyncIterable[[rtc.AudioFrame]]) -> AsyncIterable[core.STTPluginResult]:
        async def iterator():
            async for frames in frame_groups:
                res = await self._push_frames(frames)
                yield core.AsyncIteratorList([res])

        return iterator()

    async def _push_frames(self, frames: [rtc.AudioFrame]) -> core.STTPluginResult:
        resampled = [
            frame.remix_and_resample(WHISPER_SAMPLE_RATE, WHISPER_CHANNELS) for frame in frames]

        total_len = 0
        for frame in resampled:
            total_len += len(frame.data)

        np_frames = np.zeros(total_len, dtype=np.int16)
        write_index = 0
        for i in range(len(resampled)):
            np_frames[write_index: write_index +
                      len(resampled[i].data)] = resampled[i].data
            write_index += len(resampled[i].data)

        result = await asyncio.get_event_loop().run_in_executor(None, self._transcribe, np_frames.astype(dtype=np.float32) / 32768.0)
        return core.STTPluginResult(type=core.STTPluginResultType.DELTA_RESULT, text=result)

    def _transcribe(self, buffer: np.array) -> str:
        # TODO: include this with the package
        if self._model is None:
            self._model = whisper.load_model('tiny.en')

        res = whisper.transcribe(self._model, buffer)

        segments = res.get('segments', [])
        result = ""
        for segment in segments:
            if segment['no_speech_prob'] < 0.5:
                result += segment["text"]

        return result
