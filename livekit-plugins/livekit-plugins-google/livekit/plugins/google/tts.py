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

from dataclasses import dataclass

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)

from google.api_core.exceptions import DeadlineExceeded, GoogleAPICallError
from google.cloud import texttospeech
from google.cloud.texttospeech_v1.types import SsmlVoiceGender, SynthesizeSpeechResponse

from .models import AudioEncoding, Gender, SpeechLanguages


@dataclass
class _TTSOptions:
    voice: texttospeech.VoiceSelectionParams
    audio_config: texttospeech.AudioConfig


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        language: SpeechLanguages | str = "en-US",
        gender: Gender | str = "neutral",
        voice_name: str = "",  # Not required
        encoding: AudioEncoding | str = "linear16",
        sample_rate: int = 24000,
        pitch: int = 0,
        effects_profile_id: str = "",
        speaking_rate: float = 1.0,
        credentials_info: dict | None = None,
        credentials_file: str | None = None,
    ) -> None:
        """
        Create a new instance of Google TTS.

        Credentials must be provided, either by using the ``credentials_info`` dict, or reading
        from the file specified in ``credentials_file`` or the ``GOOGLE_APPLICATION_CREDENTIALS``
        environmental variable.

        Args:
            language (SpeechLanguages | str, optional): Language code (e.g., "en-US"). Default is "en-US".
            gender (Gender | str, optional): Voice gender ("male", "female", "neutral"). Default is "neutral".
            voice_name (str, optional): Specific voice name. Default is an empty string.
            encoding (AudioEncoding | str, optional): Audio encoding format (e.g., "linear16"). Default is "linear16".
            sample_rate (int, optional): Audio sample rate in Hz. Default is 24000.
            pitch (float, optional): Speaking pitch, ranging from -20.0 to 20.0 semitones relative to the original pitch. Default is 0.
            effects_profile_id (str): Optional identifier for selecting audio effects profiles to apply to the synthesized speech.
            speaking_rate (float, optional): Speed of speech. Default is 1.0.
            credentials_info (dict, optional): Dictionary containing Google Cloud credentials. Default is None.
            credentials_file (str, optional): Path to the Google Cloud credentials JSON file. Default is None.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._client: texttospeech.TextToSpeechAsyncClient | None = None
        self._credentials_info = credentials_info
        self._credentials_file = credentials_file

        voice = texttospeech.VoiceSelectionParams(
            name=voice_name,
            language_code=language,
            ssml_gender=_gender_from_str(gender),
        )

        if encoding == "linear16" or encoding == "wav":
            _audio_encoding = texttospeech.AudioEncoding.LINEAR16
        elif encoding == "mp3":
            _audio_encoding = texttospeech.AudioEncoding.MP3
        else:
            raise NotImplementedError(f"audio encoding {encoding} is not supported")

        self._opts = _TTSOptions(
            voice=voice,
            audio_config=texttospeech.AudioConfig(
                audio_encoding=_audio_encoding,
                sample_rate_hertz=sample_rate,
                pitch=pitch,
                effects_profile_id=effects_profile_id,
                speaking_rate=speaking_rate,
            ),
        )

    def update_options(
        self,
        *,
        language: SpeechLanguages | str = "en-US",
        gender: Gender | str = "neutral",
        voice_name: str = "",  # Not required
        speaking_rate: float = 1.0,
    ) -> None:
        """
        Update the TTS options.

        Args:
            language (SpeechLanguages | str, optional): Language code (e.g., "en-US"). Default is "en-US".
            gender (Gender | str, optional): Voice gender ("male", "female", "neutral"). Default is "neutral".
            voice_name (str, optional): Specific voice name. Default is an empty string.
            speaking_rate (float, optional): Speed of speech. Default is 1.0.
        """
        self._opts.voice = texttospeech.VoiceSelectionParams(
            name=voice_name,
            language_code=language,
            ssml_gender=_gender_from_str(gender),
        )
        self._opts.audio_config.speaking_rate = speaking_rate

    def _ensure_client(self) -> texttospeech.TextToSpeechAsyncClient:
        if self._client is None:
            if self._credentials_info:
                self._client = (
                    texttospeech.TextToSpeechAsyncClient.from_service_account_info(
                        self._credentials_info
                    )
                )

            elif self._credentials_file:
                self._client = (
                    texttospeech.TextToSpeechAsyncClient.from_service_account_file(
                        self._credentials_file
                    )
                )
            else:
                self._client = texttospeech.TextToSpeechAsyncClient()

        assert self._client is not None
        return self._client

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        segment_id: str | None = None,
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            client=self._ensure_client(),
            segment_id=segment_id,
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
        client: texttospeech.TextToSpeechAsyncClient,
        segment_id: str | None = None,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts, self._client = opts, client
        self._segment_id = segment_id or utils.shortuuid()

    async def _run(self) -> None:
        request_id = utils.shortuuid()

        try:
            response: SynthesizeSpeechResponse = await self._client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=self._input_text),
                voice=self._opts.voice,
                audio_config=self._opts.audio_config,
                timeout=self._conn_options.timeout,
            )

            if self._opts.audio_config.audio_encoding == "mp3":
                decoder = utils.codecs.Mp3StreamDecoder()
                bstream = utils.audio.AudioByteStream(
                    sample_rate=self._opts.audio_config.sample_rate_hertz,
                    num_channels=1,
                )
                for frame in decoder.decode_chunk(response.audio_content):
                    for frame in bstream.write(frame.data.tobytes()):
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                segment_id=self._segment_id,
                                frame=frame,
                            )
                        )

                for frame in bstream.flush():
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id,
                            segment_id=self._segment_id,
                            frame=frame,
                        )
                    )
            else:
                data = response.audio_content[44:]  # skip WAV header
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=request_id,
                        segment_id=self._segment_id,
                        frame=rtc.AudioFrame(
                            data=data,
                            sample_rate=self._opts.audio_config.sample_rate_hertz,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,  # 16-bit
                        ),
                    )
                )

        except DeadlineExceeded:
            raise APITimeoutError()
        except GoogleAPICallError as e:
            raise APIStatusError(
                e.message,
                status_code=e.code or -1,
                request_id=None,
                body=None,
            )
        except Exception as e:
            raise APIConnectionError() from e


def _gender_from_str(gender: str) -> SsmlVoiceGender:
    ssml_gender = SsmlVoiceGender.NEUTRAL
    if gender == "male":
        ssml_gender = SsmlVoiceGender.MALE
    elif gender == "female":
        ssml_gender = SsmlVoiceGender.FEMALE

    return ssml_gender  # type: ignore
