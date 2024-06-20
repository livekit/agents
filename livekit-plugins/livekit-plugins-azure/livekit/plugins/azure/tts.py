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

from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass
from typing import Optional

from livekit import rtc
from livekit.agents import tts

import azure.cognitiveservices.speech as speechsdk

from .log import logger

AZURE_SAMPLE_RATE: int = 16000
AZURE_BITS_PER_SAMPLE: int = 16
AZURE_NUM_CHANNELS: int = 1


@dataclass
class _TTSOptions:
    speech_key: str | None = None
    speech_region: str | None = None
    # see https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts
    voice: str | None = None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        speech_key: str | None = None,
        speech_region: str | None = None,
        voice: str | None = None,
    ) -> None:
        super().__init__(
            streaming_supported=False,
            sample_rate=AZURE_SAMPLE_RATE,
            num_channels=AZURE_NUM_CHANNELS,
        )

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
        )

    def synthesize(
        self,
        text: str,
    ) -> "ChunkedStream":
        return ChunkedStream(text, self._opts)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, text: str, opts: _TTSOptions) -> None:
        self._opts = opts
        self._text = text
        self._main_task: asyncio.Task | None = None
        self._queue = asyncio.Queue[Optional[tts.SynthesizedAudio]]()

    async def _run(self):
        try:
            stream_callback = _PushAudioOutputStreamCallback(
                asyncio.get_running_loop(), self._queue
            )
            push_stream = speechsdk.audio.PushAudioOutputStream(stream_callback)
            synthesizer = _create_speech_synthesizer(
                config=self._opts, stream=push_stream
            )

            def _synthesize() -> speechsdk.SpeechSynthesisResult:
                return synthesizer.speak_text_async(self._text).get()  # type: ignore

            result = await asyncio.to_thread(_synthesize)
            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                raise ValueError(
                    f"failed to synthesize audio: {result.reason} {result.cancellation_details}"
                )

            def _cleanup() -> None:
                nonlocal synthesizer, result
                del synthesizer
                del result

            await asyncio.to_thread(_cleanup)

        except Exception:
            logger.exception("failed to synthesize")
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


def _create_speech_synthesizer(
    *, config: _TTSOptions, stream: speechsdk.audio.AudioOutputStream
) -> speechsdk.SpeechSynthesizer:
    speech_config = speechsdk.SpeechConfig(
        subscription=config.speech_key, region=config.speech_region
    )
    stream_config = speechsdk.audio.AudioOutputConfig(stream=stream)
    if config.voice is not None:
        speech_config.speech_synthesis_voice_name = config.voice

    return speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=stream_config
    )


class _PushAudioOutputStreamCallback(speechsdk.audio.PushAudioOutputStreamCallback):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        event_queue: asyncio.Queue[tts.SynthesizedAudio | None],
    ):
        super().__init__()
        self._event_queue = event_queue
        self._loop = loop

    def write(self, audio_buffer: memoryview) -> int:
        audio_frame = rtc.AudioFrame(
            data=audio_buffer,
            sample_rate=AZURE_SAMPLE_RATE,
            num_channels=AZURE_NUM_CHANNELS,
            samples_per_channel=audio_buffer.nbytes // 2,
        )

        audio = tts.SynthesizedAudio(text="", data=audio_frame)
        self._loop.call_soon_threadsafe(self._event_queue.put_nowait, audio)
        return audio_buffer.nbytes
