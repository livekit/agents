import io
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import AsyncIterator, Optional
from openai import OpenAI
import torch
import torchaudio

import whisper
from livekit import rtc
from livekit.plugins import core
import numpy as np


class TTSPlugin(agents.TTSPlugin):

    def __init__(self):
        self._model = None
        super().__init__(process=self.process)

    def process(self, text_iterator: AsyncIterator[AsyncIterator[str]]) -> AsyncIterator[core.Plugin.Event[AsyncIterator[rtc.AudioFrame]]]:
        async def iterator():
            async for texts in text_iterator:
                complete_sentence = ""
                async for text in texts:
                    complete_sentence += text
                # run in executor
                result_queue = asyncio.Queue[rtc.AudioFrame]()
                result_iterator = core.AsyncQueueIterator(result_queue)
                event_loop = asyncio.get_event_loop()
                event_loop.run_in_executor(None, self._sync_process, complete_sentence, result_iterator, event_loop)
                event = core.Plugin.Event(type=core.PluginEventType.SUCCESS, data=result_iterator)
                yield event

        return iterator()

    def _sync_process(self, complete_sentence: str, result_iterator: core.AsyncQueueIterator[bytes], loop: asyncio.AbstractEventLoop):
        client = OpenAI()
        response = client.audio.speech.create(model="tts-1", voice="alloy", input=complete_sentence, )
        byte_stream = io.BytesIO(response.content)
        tensor, sample_rate = torchaudio.load(byte_stream)
        array = tensor.numpy()

        async def add_to_iterator(frame: rtc.AudioFrame, iterator: core.AsyncQueueIterator[bytes]):
            await iterator.push(frame)

        for b in array:
            audio_frame = rtc.AudioFrame(
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(b) // 2,
                data=b,
            )
            asyncio.run_coroutine_threadsafe(add_to_iterator(audio_frame, result_iterator), loop=loop)

        asyncio.run_coroutine_threadsafe(result_iterator.aclose(), loop=loop)

