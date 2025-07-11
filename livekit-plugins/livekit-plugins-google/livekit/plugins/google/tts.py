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
import weakref
from collections.abc import AsyncGenerator
from dataclasses import dataclass, replace

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import DeadlineExceeded, GoogleAPICallError
from google.cloud import texttospeech
from google.cloud.texttospeech_v1.types import (
    CustomPronunciations,
    SsmlVoiceGender,
    SynthesizeSpeechResponse,
)
from livekit.agents import APIConnectOptions, APIStatusError, APITimeoutError, tokenize, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import Gender, SpeechLanguages

NUM_CHANNELS = 1
DEFAULT_VOICE_NAME = "en-US-Chirp3-HD-Charon"
DEFAULT_LANGUAGE = "en-US"
DEFAULT_GENDER = "neutral"


@dataclass
class _TTSOptions:
    voice: texttospeech.VoiceSelectionParams
    encoding: texttospeech.AudioEncoding
    sample_rate: int
    pitch: float
    effects_profile_id: str
    speaking_rate: float
    tokenizer: tokenize.SentenceTokenizer
    volume_gain_db: float
    custom_pronunciations: CustomPronunciations | None
    enable_ssml: bool


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
        volume_gain_db: float = 0.0,
        location: str = "global",
        audio_encoding: texttospeech.AudioEncoding = texttospeech.AudioEncoding.OGG_OPUS,  # type: ignore
        credentials_info: NotGivenOr[dict] = NOT_GIVEN,
        credentials_file: NotGivenOr[str] = NOT_GIVEN,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        custom_pronunciations: NotGivenOr[CustomPronunciations] = NOT_GIVEN,
        use_streaming: bool = True,
        enable_ssml: bool = False,
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
            volume_gain_db (float, optional): Volume gain in decibels. Default is 0.0. In the range [-96.0, 16.0]. Strongly recommended not to exceed +10 (dB).
            credentials_info (dict, optional): Dictionary containing Google Cloud credentials. Default is None.
            credentials_file (str, optional): Path to the Google Cloud credentials JSON file. Default is None.
            tokenizer (tokenize.SentenceTokenizer, optional): Tokenizer for the TTS. Default is a basic sentence tokenizer.
            custom_pronunciations (CustomPronunciations, optional): Custom pronunciations for the TTS. Default is None.
            use_streaming (bool, optional): Whether to use streaming synthesis. Default is True.
            enable_ssml (bool, optional): Whether to enable SSML support. Default is False.
        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=use_streaming),
            sample_rate=sample_rate,
            num_channels=1,
        )

        if enable_ssml and use_streaming:
            raise ValueError("SSML support is not available for streaming synthesis")

        self._client: texttospeech.TextToSpeechAsyncClient | None = None
        self._credentials_info = credentials_info
        self._credentials_file = credentials_file
        self._location = location

        lang = language if is_given(language) else DEFAULT_LANGUAGE
        ssml_gender = _gender_from_str(DEFAULT_GENDER if not is_given(gender) else gender)
        name = DEFAULT_VOICE_NAME if not is_given(voice_name) else voice_name

        voice_params = texttospeech.VoiceSelectionParams(
            name=name,
            language_code=lang,
            ssml_gender=ssml_gender,
        )
        if not is_given(tokenizer):
            tokenizer = tokenize.blingfire.SentenceTokenizer()

        pronunciations = None if not is_given(custom_pronunciations) else custom_pronunciations

        self._opts = _TTSOptions(
            voice=voice_params,
            encoding=audio_encoding,
            sample_rate=sample_rate,
            pitch=pitch,
            effects_profile_id=effects_profile_id,
            speaking_rate=speaking_rate,
            tokenizer=tokenizer,
            volume_gain_db=volume_gain_db,
            custom_pronunciations=pronunciations,
            enable_ssml=enable_ssml,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    def update_options(
        self,
        *,
        language: NotGivenOr[SpeechLanguages | str] = NOT_GIVEN,
        gender: NotGivenOr[Gender | str] = NOT_GIVEN,
        voice_name: NotGivenOr[str] = NOT_GIVEN,
        speaking_rate: NotGivenOr[float] = NOT_GIVEN,
        volume_gain_db: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options.

        Args:
            language (SpeechLanguages | str, optional): Language code (e.g., "en-US").
            gender (Gender | str, optional): Voice gender ("male", "female", "neutral").
            voice_name (str, optional): Specific voice name.
            speaking_rate (float, optional): Speed of speech.
            volume_gain_db (float, optional): Volume gain in decibels.
        """
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
            self._opts.speaking_rate = speaking_rate
        if is_given(volume_gain_db):
            self._opts.volume_gain_db = volume_gain_db

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

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    def _build_ssml(self) -> str:
        ssml = "<speak>"
        ssml += self._input_text
        ssml += "</speak>"
        return ssml

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            input = (
                texttospeech.SynthesisInput(
                    ssml=self._build_ssml(),
                    custom_pronunciations=self._opts.custom_pronunciations,
                )
                if self._opts.enable_ssml
                else texttospeech.SynthesisInput(
                    text=self._input_text,
                    custom_pronunciations=self._opts.custom_pronunciations,
                )
            )
            response: SynthesizeSpeechResponse = await self._tts._ensure_client().synthesize_speech(
                input=input,
                voice=self._opts.voice,
                audio_config=texttospeech.AudioConfig(
                    audio_encoding=self._opts.encoding,
                    sample_rate_hertz=self._opts.sample_rate,
                    pitch=self._opts.pitch,
                    effects_profile_id=self._opts.effects_profile_id,
                    speaking_rate=self._opts.speaking_rate,
                    volume_gain_db=self._opts.volume_gain_db,
                ),
                timeout=self._conn_options.timeout,
            )

            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                mime_type=_encoding_to_mimetype(self._opts.encoding),
            )

            output_emitter.push(response.audio_content)
        except DeadlineExceeded:
            raise APITimeoutError() from None
        except GoogleAPICallError as e:
            raise APIStatusError(e.message, status_code=e.code or -1) from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._segments_ch = utils.aio.Chan[tokenize.SentenceStream]()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        encoding = self._opts.encoding
        if encoding not in (texttospeech.AudioEncoding.OGG_OPUS, texttospeech.AudioEncoding.PCM):
            enc_name = texttospeech.AudioEncoding._member_names_[encoding]
            logger.warning(
                f"encoding {enc_name} isn't supported by the streaming_synthesize, "
                "fallbacking to PCM"
            )
            encoding = texttospeech.AudioEncoding.PCM  # type: ignore

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type=_encoding_to_mimetype(encoding),
            stream=True,
        )

        streaming_config = texttospeech.StreamingSynthesizeConfig(
            voice=self._opts.voice,
            streaming_audio_config=texttospeech.StreamingAudioConfig(
                audio_encoding=encoding,
                sample_rate_hertz=self._opts.sample_rate,
                speaking_rate=self._opts.speaking_rate,
            ),
            custom_pronunciations=self._opts.custom_pronunciations,
        )

        async def _tokenize_input() -> None:
            input_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if input_stream is None:
                        input_stream = self._opts.tokenizer.stream()
                        self._segments_ch.send_nowait(input_stream)
                    input_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if input_stream:
                        input_stream.end_input()
                    input_stream = None

            self._segments_ch.close()

        async def _run_segments() -> None:
            async for input_stream in self._segments_ch:
                await self._run_stream(input_stream, output_emitter, streaming_config)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)

    async def _run_stream(
        self,
        input_stream: tokenize.SentenceStream,
        output_emitter: tts.AudioEmitter,
        streaming_config: texttospeech.StreamingSynthesizeConfig,
    ) -> None:
        @utils.log_exceptions(logger=logger)
        async def input_generator() -> AsyncGenerator[
            texttospeech.StreamingSynthesizeRequest, None
        ]:
            try:
                yield texttospeech.StreamingSynthesizeRequest(streaming_config=streaming_config)

                async for input in input_stream:
                    self._mark_started()
                    yield texttospeech.StreamingSynthesizeRequest(
                        input=texttospeech.StreamingSynthesisInput(text=input.token)
                    )

            except Exception:
                logger.exception("an error occurred while streaming input to google TTS")

        input_gen = input_generator()
        try:
            stream = await self._tts._ensure_client().streaming_synthesize(
                input_gen, timeout=self._conn_options.timeout
            )
            output_emitter.start_segment(segment_id=utils.shortuuid())

            async for resp in stream:
                output_emitter.push(resp.audio_content)

            output_emitter.end_segment()

        except DeadlineExceeded:
            raise APITimeoutError() from None
        except GoogleAPICallError as e:
            raise APIStatusError(e.message, status_code=e.code or -1) from e
        finally:
            await input_gen.aclose()


def _gender_from_str(gender: str) -> SsmlVoiceGender:
    ssml_gender = SsmlVoiceGender.NEUTRAL
    if gender == "male":
        ssml_gender = SsmlVoiceGender.MALE
    elif gender == "female":
        ssml_gender = SsmlVoiceGender.FEMALE

    return ssml_gender  # type: ignore


def _encoding_to_mimetype(encoding: texttospeech.AudioEncoding) -> str:
    if encoding == texttospeech.AudioEncoding.PCM:
        return "audio/pcm"
    elif encoding == texttospeech.AudioEncoding.LINEAR16:
        return "audio/wav"
    elif encoding == texttospeech.AudioEncoding.MP3:
        return "audio/mp3"
    elif encoding == texttospeech.AudioEncoding.OGG_OPUS:
        return "audio/opus"
    else:
        raise RuntimeError(f"encoding {encoding} isn't supported")
