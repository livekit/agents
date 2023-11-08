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
from livekit import agents
import numpy as np


WHISPER_SAMPLE_RATE = 16000
WHISPER_CHANNELS = 1


class TTSPlugin(agents.TTSPlugin):

    def __init__(self):
        self._model = None
        super().__init__(process=self.process)

    def process(self, text_iterator: AsyncIterator[AsyncIterator[str]]) -> AsyncIterator[agents.Plugin.Event[agents.STTPluginEvent]]:
        async def iterator():
            print("NEIL ti", text_iterator)
            async for texts in text_iterator:
                complete_sentence = ""
                async for text in texts:
                    complete_sentence += text
                # run in executor
                result_queue = asyncio.Queue[bytes]()
                result_iterator = agents.utils.AsyncQueueIterator(result_queue)
                event_loop = asyncio.get_event_loop()
                event_loop.run_in_executor(None, self._sync_process, complete_sentence, result_iterator, event_loop)
                yield result_iterator

        return iterator()

    def _sync_process(self, complete_sentence: str, result_iterator: agents.utils.AsyncQueueIterator[bytes], loop: asyncio.AbstractEventLoop):
        client = OpenAI()
        response = client.audio.speech.create(model="tts-1", voice="alloy", input=complete_sentence, )
        byte_stream = io.BytesIO(response.content)
        tensor, sample_rate = torchaudio.load(byte_stream)
        array = tensor.numpy()

        async def add_to_iterator(item, iterator: agents.utils.AsyncQueueIterator[bytes]):
            await iterator.push(item)

        for b in array:
            print("NEIL ti 2", b)
            asyncio.run_coroutine_threadsafe(add_to_iterator(b, result_iterator), loop=loop)

        asyncio.run_coroutine_threadsafe(result_iterator.aclose(), loop=loop)

        
