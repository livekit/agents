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
import os
from dataclasses import dataclass
from typing import Optional

from livekit import rtc
from livekit.agents import stt
from livekit.agents.utils import AudioBuffer

import azure.cognitiveservices.speech as speechsdk

from .log import logger


@dataclass
class STTOptions:
    speech_key: str
    speech_region: str
    sample_rate: int
    num_channels: int
    languages: list[
        str
    ]  # see https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=stt


class STT(stt.STT):
    def __init__(
        self,
        *,
        speech_key: str | None = None,
        speech_region: str | None = None,
        sample_rate: int = 48000,
        num_channels: int = 1,
        languages: list[str] = [],  # when empty, auto-detect the language
    ):
        super().__init__(streaming_supported=True)

        speech_key = speech_key or os.environ.get("AZURE_SPEECH_KEY")
        if not speech_key:
            raise ValueError("AZURE_SPEECH_KEY must be set")

        speech_region = speech_region or os.environ.get("AZURE_SPEECH_REGION")
        if not speech_region:
            raise ValueError("AZURE_SPEECH_REGION must be set")

        self._config = STTOptions(
            speech_key=speech_key,
            speech_region=speech_region,
            languages=languages,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
        language: str | None = None,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Azure STT does not support single frame recognition")

    def stream(
        self,
        *,
        language: str | None = None,
    ) -> "SpeechStream":
        return SpeechStream(self._config)


class SpeechStream(stt.SpeechStream):
    def __init__(self, opts: STTOptions) -> None:
        super().__init__()
        self._opts = opts
        self._event_queue = asyncio.Queue[Optional[stt.SpeechEvent]]()
        self._closed = False
        self._speaking = False

        self._stream = speechsdk.audio.PushAudioInputStream(
            stream_format=speechsdk.audio.AudioStreamFormat(
                samples_per_second=self._opts.sample_rate,
                bits_per_sample=16,
                channels=self._opts.num_channels,
            )
        )
        self._recognizer = _create_speech_recognizer(
            config=self._opts, stream=self._stream
        )
        self._recognizer.recognizing.connect(self._on_recognizing)
        self._recognizer.recognized.connect(self._on_recognized)
        self._recognizer.speech_start_detected.connect(self._on_speech_start)
        self._recognizer.speech_end_detected.connect(self._on_speech_end)
        self._recognizer.session_stopped.connect(self._on_session_stopped)
        self._recognizer.start_continuous_recognition()
        self._done_event = asyncio.Event()
        self._loop = asyncio.get_running_loop()

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")

        self._stream.write(frame.data.tobytes())

    async def aclose(self, *, wait: bool = True) -> None:
        if self._closed:
            return

        self._closed = True
        self._stream.close()

        await self._done_event.wait()

        def _cleanup():
            self._recognizer.stop_continuous_recognition()
            del self._recognizer

        await asyncio.to_thread(_cleanup)

    def _on_recognized(self, evt: speechsdk.SpeechRecognitionEventArgs):
        detected_lg = speechsdk.AutoDetectSourceLanguageResult(evt.result).language
        text = evt.result.text.strip()
        if not text:
            return

        final_data = stt.SpeechData(
            language=detected_lg,
            confidence=1.0,
            text=evt.result.text,
        )

        self._threadsafe_put(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[final_data],
            )
        )

    def _on_recognizing(self, evt: speechsdk.SpeechRecognitionEventArgs):
        detected_lg = speechsdk.AutoDetectSourceLanguageResult(evt.result).language
        text = evt.result.text.strip()
        if not text:
            return

        interim_data = stt.SpeechData(
            language=detected_lg,
            confidence=0.0,
            text=evt.result.text,
        )

        self._threadsafe_put(
            stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[interim_data],
            )
        )

    def _on_speech_start(self, evt: speechsdk.SpeechRecognitionEventArgs):
        if self._speaking:
            return

        self._speaking = True
        self._threadsafe_put(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

    def _on_speech_end(self, evt: speechsdk.SpeechRecognitionEventArgs):
        if not self._speaking:
            return

        self._speaking = False
        self._threadsafe_put(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

    def _on_session_stopped(self, evt: speechsdk.SpeechRecognitionEventArgs):
        if not self._closed:
            logger.error("session stopped unexpectedly")

        self._loop.call_soon_threadsafe(self._done_event.set)
        self._threadsafe_put(None)

    def _threadsafe_put(self, evt: stt.SpeechEvent | None):
        self._loop.call_soon_threadsafe(self._event_queue.put_nowait, evt)

    async def __anext__(self) -> stt.SpeechEvent:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration

        return evt


def _create_speech_recognizer(
    *, config: STTOptions, stream: speechsdk.audio.AudioInputStream
) -> speechsdk.SpeechRecognizer:
    speech_config = speechsdk.SpeechConfig(
        subscription=config.speech_key, region=config.speech_region
    )

    auto_detect_source_language_config = None
    if config.languages:
        auto_detect_source_language_config = (
            speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                languages=config.languages
            )
        )

    audio_config = speechsdk.audio.AudioConfig(stream=stream)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
        auto_detect_source_language_config=auto_detect_source_language_config,  # type: ignore
    )

    return speech_recognizer
