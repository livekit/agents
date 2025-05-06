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

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import DeadlineExceeded, GoogleAPICallError
from google.cloud import texttospeech
from google.cloud.texttospeech_v1.types import SsmlVoiceGender, SynthesizeSpeechResponse
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .models import Gender, SpeechLanguages


@dataclass
class _TTSOptions:
    voice: texttospeech.VoiceSelectionParams
    audio_config: texttospeech.AudioConfig


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        language: NotGivenOr[SpeechLanguages | str] = NOT_GIVEN,
        gender: NotGivenOr[Gender | str] = NOT_GIVEN,
        voice_name: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = 24000,
        pitch: int = 0,
        effects_profile_id: str = "",
        speaking_rate: float = 1.0,
        location: str = "global",
        credentials_info: NotGivenOr[dict] = NOT_GIVEN,
        credentials_file: NotGivenOr[str] = NOT_GIVEN,
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
            sample_rate (int, optional): Audio sample rate in Hz. Default is 24000.
            location (str, optional): Location for the TTS client. Default is "global".
            pitch (float, optional): Speaking pitch, ranging from -20.0 to 20.0 semitones relative to the original pitch. Default is 0.
            effects_profile_id (str): Optional identifier for selecting audio effects profiles to apply to the synthesized speech.
            speaking_rate (float, optional): Speed of speech. Default is 1.0.
            credentials_info (dict, optional): Dictionary containing Google Cloud credentials. Default is None.
            credentials_file (str, optional): Path to the Google Cloud credentials JSON file. Default is None.
        """  # noqa: E501

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._client: texttospeech.TextToSpeechAsyncClient | None = None
        self._credentials_info = credentials_info
        self._credentials_file = credentials_file
        self._location = location

        lang = language if is_given(language) else "en-US"
        ssml_gender = _gender_from_str("neutral" if not is_given(gender) else gender)
        name = "" if not is_given(voice_name) else voice_name

        voice_params = texttospeech.VoiceSelectionParams(
            name=name,
            language_code=lang,
            ssml_gender=ssml_gender,
        )

        self._opts = _TTSOptions(
            voice=voice_params,
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.PCM,
                sample_rate_hertz=sample_rate,
                pitch=pitch,
                effects_profile_id=effects_profile_id,
                speaking_rate=speaking_rate,
            ),
        )

    def update_options(
        self,
        *,
        language: NotGivenOr[SpeechLanguages | str] = NOT_GIVEN,
        gender: NotGivenOr[Gender | str] = NOT_GIVEN,
        voice_name: NotGivenOr[str] = NOT_GIVEN,
        speaking_rate: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options.

        Args:
            language (SpeechLanguages | str, optional): Language code (e.g., "en-US").
            gender (Gender | str, optional): Voice gender ("male", "female", "neutral").
            voice_name (str, optional): Specific voice name.
            speaking_rate (float, optional): Speed of speech.
        """  # noqa: E501
        params = {}
        if is_given(language):
            params["language_code"] = str(language)
        if is_given(gender):
            params["ssml_gender"] = _gender_from_str(str(gender))
        if is_given(voice_name):
            params["name"] = voice_name

        if params:
            self._opts.voice = texttospeech.VoiceSelectionParams(**params)

        if is_given(speaking_rate):
            self._opts.audio_config.speaking_rate = speaking_rate

    def _ensure_client(self) -> texttospeech.TextToSpeechAsyncClient:
        api_endpoint = "texttospeech.googleapis.com"
        if self._location != "global":
            api_endpoint = f"{self._location}-texttospeech.googleapis.com"

        if self._client is None:
            if self._credentials_info:
                self._client = texttospeech.TextToSpeechAsyncClient.from_service_account_info(
                    self._credentials_info, client_options=ClientOptions(api_endpoint=api_endpoint)
                )

            elif self._credentials_file:
                self._client = texttospeech.TextToSpeechAsyncClient.from_service_account_file(
                    self._credentials_file, client_options=ClientOptions(api_endpoint=api_endpoint)
                )
            else:
                self._client = texttospeech.TextToSpeechAsyncClient(
                    client_options=ClientOptions(api_endpoint=api_endpoint)
                )

        assert self._client is not None
        return self._client

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            client=self._ensure_client(),
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        client: texttospeech.TextToSpeechAsyncClient,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts, self._client = opts, client

    async def _run(self, output_emitter: tts.SynthesizedAudioEmitter):
        try:
            response: SynthesizeSpeechResponse = await self._client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=self._input_text),
                voice=self._opts.voice,
                audio_config=self._opts.audio_config,
                timeout=self._conn_options.timeout,
            )

            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=self._opts.audio_config.sample_rate_hertz,
                num_channels=1,
                mime_type="audio/pcm",
            )

            output_emitter.push(response.audio_content)
            output_emitter.flush()
        except DeadlineExceeded:
            raise APITimeoutError() from None
        except GoogleAPICallError as e:
            raise APIStatusError(
                e.message, status_code=e.code or -1, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


def _gender_from_str(gender: str) -> SsmlVoiceGender:
    ssml_gender = SsmlVoiceGender.NEUTRAL
    if gender == "male":
        ssml_gender = SsmlVoiceGender.MALE
    elif gender == "female":
        ssml_gender = SsmlVoiceGender.FEMALE

    return ssml_gender  # type: ignore
