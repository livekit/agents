# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import asyncio
import functools
import os
from dataclasses import dataclass
from typing import Optional

from livekit import rtc
from livekit.agents import aio, tokenize, tts

import azure.cognitiveservices.speech as speechsdk

from .log import logger

@dataclass
class _TTSOptions:
    word_tokenizer: tokenize.WordTokenizer
    speech_key: str = None
    speech_region: str = None
    # see https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts
    voice: str = None

def _create_speech_synthesizer(*, config: _TTSOptions, stream : speechsdk.audio.AudioOutputStream) -> speechsdk.SpeechSynthesizer:
    # Creates an instance of a speech config with specified subscription key and service region.
    speech_config = speechsdk.SpeechConfig(subscription=config.speech_key, region=config.speech_region)
    # add stream output
    stream_config = speechsdk.audio.AudioOutputConfig(stream=stream)
    if config.voice is not None:
        # set custom voice, if specified
        speech_config.speech_synthesis_voice_name=config.voice

    return speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=stream_config)

class TTS(tts.TTS):
    SAMPLE_RATE: int = 16000
    BITS_PER_SAMPLE: int = 16
    NUM_CHANNELS: int = 1

    def __init__(
        self,
        *,
        speech_key: Optional[str] = None,
        speech_region: Optional[str] = None,
        voice: Optional[str] = None,
        word_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(),
    ) -> None:
        super().__init__(streaming_supported=True, sample_rate=TTS.SAMPLE_RATE, num_channels=TTS.NUM_CHANNELS)

        speech_key = speech_key or os.environ.get("AZURE_SPEECH_KEY")
        if not speech_key:
            raise ValueError("AZURE_SPEECH_KEY must be set")
        speech_region = speech_region or os.environ.get("AZURE_SPEECH_REGION")
        if not speech_region:
            raise ValueError("AZURE_SPEECH_REGION must be set")

        self._opts = _TTSOptions(
            speech_key=speech_key,
            speech_region=speech_region,
            voice=voice,
            word_tokenizer=word_tokenizer
        )

    def synthesize(
        self,
        text: str,
    ) -> "ChunkedStream":
        return ChunkedStream(text, self._opts)

    def stream(
        self,
    ) -> "SynthesizeStream":
        return SynthesizeStream(self._opts)

class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, text: str, opts: _TTSOptions
    ) -> None:
        self._opts = opts
        self._text = text
        self._main_task: asyncio.Task | None = None
        self._queue = asyncio.Queue[tts.SynthesizedAudio | None]()
 
    async def _run(self):
        loop = asyncio.get_running_loop()

        class PushAudioOutputStreamCallback(speechsdk.audio.PushAudioOutputStreamCallback):
            def __init__(self, push_queue: asyncio.Queue[tts.SynthesizedAudio | None]):
                super().__init__()
                self._event_queue = push_queue

            def write(self, audio_buffer: memoryview) -> int:
                # create a rtc frame
                audio_frame = rtc.AudioFrame(
                    data=audio_buffer,
                    sample_rate=TTS.SAMPLE_RATE,
                    num_channels=TTS.NUM_CHANNELS,
                    samples_per_channel=audio_buffer.nbytes // 2,
                )
                se = tts.SynthesizedAudio(
                        text="",
                        data=audio_frame
                    )
                # and write to output queue
                loop.call_soon_threadsafe(
                    functools.partial(self._event_queue.put_nowait, se)
                )
                return audio_buffer.nbytes

        try:
            stream_callback = PushAudioOutputStreamCallback(self._queue)
            push_stream = speechsdk.audio.PushAudioOutputStream(stream_callback)
            speech_synthesizer = _create_speech_synthesizer(config=self._opts, stream=push_stream)
            
            done = asyncio.Event()
            def speech_synthesizer_synthesis_completed_cb(evt):
                # fired when EOF reached on pull input stream or cancellation
                loop.call_soon_threadsafe(
                        functools.partial(done.set)
                    )
            speech_synthesizer.synthesis_completed.connect(speech_synthesizer_synthesis_completed_cb)

            # wait for completion
            result_future = speech_synthesizer.speak_text_async(self._text)
            await done.wait()
            result = result_future.get()
            # Destroys result which is necessary for destroying speech synthesizer
            del result
            
            speech_synthesizer.stop_speaking()
            # Destroys the synthesizer in order to close the output stream.
            del speech_synthesizer

        except Exception:
            logger.warning("failed to synthesize")
        finally:
            self._queue.put_nowait(None)
            
    async def __anext__(self) -> tts.SynthesizedAudio:
        if not self._main_task:
            self._main_task = asyncio.create_task(self._run())

        frame = await self._queue.get()
        if frame is None:
            raise StopAsyncIteration

        return frame

    async def aclose(self) -> None:
        if not self._main_task:
            return

        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        opts: _TTSOptions
    ):
        self._opts = opts
        self._main_task = asyncio.create_task(self._run())
        self._event_queue = asyncio.Queue[Optional[tts.SynthesisEvent]]()
        self._closed = False
        self._word_stream = opts.word_tokenizer.stream()

    def push_text(self, token: str | None) -> None:
        if self._closed:
            raise ValueError("cannot push to a closed stream")

        if token is None:
            self._word_stream.mark_segment_end()
            return

        self._word_stream.push_text(token)

    async def aclose(self, *, wait: bool = True) -> None:
        self._closed = True
        await self._word_stream.aclose()

        if not wait:
            self._main_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()

        class PushAudioOutputStreamCallback(speechsdk.audio.PushAudioOutputStreamCallback):
            def __init__(self, push_queue: asyncio.Queue[tts.SynthesisEvent]):
                super().__init__()
                self._event_queue = push_queue

            def write(self, audio_buffer: memoryview) -> int:
                # create a rtc frame
                audio_frame = rtc.AudioFrame(
                    data=audio_buffer,
                    sample_rate=TTS.SAMPLE_RATE,
                    num_channels=TTS.NUM_CHANNELS,
                    samples_per_channel=audio_buffer.nbytes // 2,
                )
                se = tts.SynthesisEvent(
                        type=tts.SynthesisEventType.AUDIO,
                        audio=tts.SynthesizedAudio(text="", data=audio_frame),
                    )
                # and push it to caller
                loop.call_soon_threadsafe(
                    functools.partial(self._event_queue.put_nowait, se)
                )
                return audio_buffer.nbytes
        
        stream_callback = PushAudioOutputStreamCallback(self._event_queue)
        push_stream = speechsdk.audio.PushAudioOutputStream(stream_callback)
        speech_synthesizer = _create_speech_synthesizer(config=self._opts, stream=push_stream)

        segment_done = asyncio.Event()
        def speech_synthesizer_synthesis_completed_cb(evt):
            # fired when EOF reached on pull input stream or cancellation
            loop.call_soon_threadsafe(
                    functools.partial(segment_done.set)
                )
        speech_synthesizer.synthesis_completed.connect(speech_synthesizer_synthesis_completed_cb)
    
        async for ev in self._word_stream:
            if ev.type == tokenize.TokenEventType.STARTED:
                self._event_queue.put_nowait(
                    tts.SynthesisEvent(type=tts.SynthesisEventType.STARTED)
                )
            elif ev.type == tokenize.TokenEventType.TOKEN:
                segment_done.clear()
                result_future = speech_synthesizer.speak_text_async(ev.token)
                await segment_done.wait()
                result = result_future.get()
                # Destroys result which is necessary for destroying speech synthesizer
                del result
            elif ev.type == tokenize.TokenEventType.FINISHED:
                self._event_queue.put_nowait(
                    tts.SynthesisEvent(type=tts.SynthesisEventType.FINISHED)
                )
                break
            
        speech_synthesizer.stop_speaking()
        # Destroys the synthesizer in order to close the output stream.
        del speech_synthesizer
        
        self._event_queue.put_nowait(None)

    async def __anext__(self) -> tts.SynthesisEvent:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration

        return evt
