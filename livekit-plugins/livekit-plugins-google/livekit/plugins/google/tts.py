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

from typing import Optional, Union
from google.cloud import texttospeech
from google.cloud.texttospeech_v1.types import (
    SynthesizeSpeechResponse,
    SsmlVoiceGender,
)
from livekit import rtc
from livekit.agents import tts
from .models import AudioEncoding, Gender, SpeechLanguages
from dataclasses import dataclass


LgType = Union[SpeechLanguages, str]
GenderType = Union[Gender, str]
AudioEncodingType = Union[AudioEncoding, str]


@dataclass
class TTSOptions:
    voice: texttospeech.VoiceSelectionParams
    audio_config: texttospeech.AudioConfig


class TTS(tts.TTS):
    def __init__(
        self,
        config: Optional[TTSOptions] = None,
        *,
        language: LgType = "en-US",
        gender: GenderType = "neutral",
        voice_name: str = "",  # Not required
        audio_encoding: AudioEncodingType = "wav",
        sample_rate: int = 24000,
        speaking_rate: float = 1.0,
        credentials_info: Optional[dict] = None,
        credentials_file: Optional[str] = None,
    ) -> None:
        super().__init__(streaming_supported=True)

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

        if not config:
            _gender = SsmlVoiceGender.NEUTRAL
            if gender == "male":
                _gender = SsmlVoiceGender.MALE
            elif gender == "female":
                _gender = SsmlVoiceGender.FEMALE
            voice = texttospeech.VoiceSelectionParams(
                name=voice_name,
                language_code=language,
                ssml_gender=_gender,
            )
            # Default audio to LINEAR16 (the encoding used in WAV files).
            _audio_encoding = texttospeech.AudioEncoding.LINEAR16
            if audio_encoding == "mp3":
                _audio_encoding = texttospeech.AudioEncoding.MP3
            elif audio_encoding == "opus":
                _audio_encoding = texttospeech.AudioEncoding.OGG_OPUS
            elif audio_encoding == "mulaw":
                _audio_encoding = texttospeech.AudioEncoding.MULAW
            elif audio_encoding == "alaw":
                _audio_encoding = texttospeech.AudioEncoding.ALAW
            config = TTSOptions(
                voice=voice,
                audio_config=texttospeech.AudioConfig(
                    audio_encoding=_audio_encoding,
                    sample_rate_hertz=sample_rate,
                    speaking_rate=speaking_rate,
                ),
            )
        self._config = config
        self._creds = self._client.transport._credentials  # TODO: is this needed?

    async def synthesize(
        self,
        *,
        text: str,
    ) -> tts.SynthesizedAudio:
        # Perform the text-to-speech request on the text input with the selected
        # voice parameters and audio file type
        response: SynthesizeSpeechResponse = await self._client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=self._config.voice,
            audio_config=self._config.audio_config,
        )

        data = response.audio_content
        return tts.SynthesizedAudio(
            text=text,
            data=rtc.AudioFrame(
                data=data,
                sample_rate=self._config.audio_config.sample_rate_hertz,
                num_channels=1,
                samples_per_channel=len(data) // 2,  # 16-bit
            ),
        )
