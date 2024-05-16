# Copyright 2023 LiveKit, Inc.
#
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
from dataclasses import dataclass
from typing import Union

from livekit import rtc
from livekit.agents import codecs, tts

from google.cloud import texttospeech
from google.cloud.texttospeech_v1.types import (
    SsmlVoiceGender,
    SynthesizeSpeechResponse,
)

from .log import logger
from .models import AudioEncoding, Gender, SpeechLanguages

LgType = Union[SpeechLanguages, str]
GenderType = Union[Gender, str]
AudioEncodingType = Union[AudioEncoding, str]


@dataclass
class _TTSOptions:
    voice: texttospeech.VoiceSelectionParams
    audio_config: texttospeech.AudioConfig


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        language: LgType = "en-US",
        gender: GenderType = "neutral",
        voice_name: str = "",  # Not required
        audio_encoding: AudioEncodingType = "linear16",
        sample_rate: int = 24000,
        speaking_rate: float = 1.0,
        credentials_info: dict | None = None,
        credentials_file: str | None = None,
    ) -> None:
        super().__init__(
            streaming_supported=False, sample_rate=sample_rate, num_channels=1
        )

        if credentials_info:
            self._client = (
                texttospeech.TextToSpeechAsyncClient.from_service_account_info(
                    credentials_info
                )
            )

        elif credentials_file:
            self._client = (
                texttospeech.TextToSpeechAsyncClient.from_service_account_file(
                    credentials_file
                )
            )
        else:
            self._client = texttospeech.TextToSpeechAsyncClient()

        ssml_gender = SsmlVoiceGender.NEUTRAL
        if gender == "male":
            ssml_gender = SsmlVoiceGender.MALE
        elif gender == "female":
            ssml_gender = SsmlVoiceGender.FEMALE

        voice = texttospeech.VoiceSelectionParams(
            name=voice_name,
            language_code=language,
            ssml_gender=ssml_gender,
        )

        if audio_encoding == "linear16" or audio_encoding == "wav":
            _audio_encoding = texttospeech.AudioEncoding.LINEAR16
        elif audio_encoding == "mp3":
            _audio_encoding = texttospeech.AudioEncoding.MP3
        else:
            raise NotImplementedError(
                f"audio encoding {audio_encoding} is not supported"
            )

        self._opts = _TTSOptions(
            voice=voice,
            audio_config=texttospeech.AudioConfig(
                audio_encoding=_audio_encoding,
                sample_rate_hertz=sample_rate,
                speaking_rate=speaking_rate,
            ),
        )

    def synthesize(
        self,
        text: str,
    ) -> "ChunkedStream":
        return ChunkedStream(text, self._opts, self._client)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, text: str, opts: _TTSOptions, client: texttospeech.TextToSpeechAsyncClient
    ) -> None:
        self._text = text
        self._opts = opts
        self._client = client
        self._main_task: asyncio.Task | None = None
        self._queue = asyncio.Queue[tts.SynthesizedAudio | None]()

    async def _run(self) -> None:
        try:
            response: SynthesizeSpeechResponse = await self._client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=self._text),
                voice=self._opts.voice,
                audio_config=self._opts.audio_config,
            )

            data = response.audio_content
            if self._opts.audio_config.audio_encoding == "mp3":
                decoder = codecs.Mp3StreamDecoder()
                frames = decoder.decode_chunk(data)
                for frame in frames:
                    self._queue.put_nowait(tts.SynthesizedAudio(text="", data=frame))
            else:
                self._queue.put_nowait(
                    tts.SynthesizedAudio(
                        text="",
                        data=rtc.AudioFrame(
                            data=data,
                            sample_rate=self._opts.audio_config.sample_rate_hertz,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,  # 16-bit
                        ),
                    )
                )

        except Exception as e:
            logger.error(f"failed to synthesize: {e}")
        finally:
            self._queue.put_nowait(None)

    async def __anext__(self) -> tts.SynthesizedAudio:
        if not self._main_task:
            self.main_task = asyncio.create_task(self._run())

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
