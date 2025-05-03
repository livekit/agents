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
import base64
import os
from dataclasses import dataclass

import aiohttp

from hume import AsyncHumeClient
from hume.tts import Format, FormatWav, PostedContext, PostedUtterance, PostedUtteranceVoice
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

# Default audio settings
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    """TTS options for Hume API"""

    api_key: str
    voice: PostedUtteranceVoice | None
    description: str | None
    speed: float | None
    context: PostedContext | None
    format: Format
    strip_headers: bool
    instant_mode: bool
    word_tokenizer: tokenize.WordTokenizer


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: NotGivenOr[PostedUtteranceVoice] = NOT_GIVEN,
        description: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        context: NotGivenOr[PostedContext] = NOT_GIVEN,
        format: NotGivenOr[Format] = NOT_GIVEN,
        instant_mode: bool = False,
        strip_headers: bool = True,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        word_tokenizer: tokenize.WordTokenizer | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Initialize the Hume TTS client.

        See https://dev.hume.ai/reference/text-to-speech-tts/synthesize-json-streaming for API doc

        Args:
            voice (NotGivenOr[PostedUtteranceVoice]): The voice, specified by name or id, to be
                used. When no voice is specified, a novel voice will be generated based on the
                text and optionally provided description.
            description (NotGivenOr[str]): Natural language instructions describing how the
                synthesized speech should sound, including but not limited to tone, intonation,
                pacing, and accent. If a Voice is specified in the request, this description
                serves as acting instructions. If no Voice is specified, a new voice is generated
                based on this description.
            speed: (NotGivenOr[float]): Adjusts the relative speaking rate on a non-linear scale
                from 0.25 (much slower) to 3.0 (much faster), where 1.0 represents normal speaking
                pace.
            context (NotGivenOr[PostedContext]): Utterances to use as context for generating
                consistent speech style and prosody across multiple requests.
            format (NotGivenOr[Format]): Specifies the output audio file format (WAV, MP3 or PCM).
                Defaults to WAV format.
            instant_mode (bool): Enables ultra-low latency streaming, reducing time to first chunk.
                Recommended for real-time applications. Only for streaming endpoints.
                With this enabled, requests incur 10% higher cost. Defaults to False.
            strip_headers (bool): If enabled, the audio for all the chunks of a generation.
                Once concatenated together, will constitute a single audio file.
                If disabled, each chunk’s audio will be its own audio file, each with its headers.
            api_key (NotGivenOr[str]): Hume API key for authentication. If not provided,
                will attempt to read from HUME_API_KEY environment variable.
            word_tokenizer (tokenize.WordTokenizer | None): Custom word tokenizer to use for text.
                If None, a basic word tokenizer will be used.
            http_session (aiohttp.ClientSession | None): Optional HTTP session for API requests.
                If None, a new session will be created.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=DEFAULT_NUM_CHANNELS,
        )

        self._api_key = api_key if is_given(api_key) else os.environ.get("HUME_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Hume API key is required, either as argument or set HUME_API_KEY env variable"
            )

        if not word_tokenizer:
            word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)

        self._opts = _TTSOptions(
            voice=voice if is_given(voice) else None,
            description=description if is_given(description) else None,
            speed=speed if is_given(speed) else None,
            context=context if is_given(context) else None,
            format=format if is_given(format) else FormatWav(),
            api_key=self._api_key,
            strip_headers=strip_headers,
            instant_mode=instant_mode,
            word_tokenizer=word_tokenizer,
        )

        self._client = AsyncHumeClient(api_key=self._api_key)
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[PostedUtteranceVoice] = NOT_GIVEN,
        description: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        context: NotGivenOr[PostedContext] = NOT_GIVEN,
        format: NotGivenOr[Format] = NOT_GIVEN,
        instant_mode: NotGivenOr[bool] = NOT_GIVEN,
        strip_headers: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """Update TTS options for synthesizing speech.

        Args:
            voice (NotGivenOr[PostedUtteranceVoice]): The voice, specified by name or id, to be
                used. When no voice is specified, a novel voice will be generated based on the
                text and optionally provided description.
            description (NotGivenOr[str]): Natural language instructions describing how the
                synthesized speech should sound, including but not limited to tone, intonation,
                pacing, and accent. If a Voice is specified in the request, this description
                serves as acting instructions. If no Voice is specified, a new voice is generated
                based on this description.
            speed: (NotGivenOr[float]): Adjusts the relative speaking rate on a non-linear scale
                from 0.25 (much slower) to 3.0 (much faster), where 1.0 represents normal speaking
                pace.
            context (Optional[PostedContext]): Utterances to use as context for generating
                consistent speech style and prosody across multiple requests.
            format (NotGivenOr[Format]): Specifies the output audio file format (WAV, MP3 or PCM).
            instant_mode (NotGivenOr[bool]): Enables ultra-low latency streaming.
                Reduces time to first audio chunk, recommended for real-time applications.
                Note: Incurs 10% higher cost when enabled.
            strip_headers (NotGivenOr[bool]): If enabled, the audio for the chunks of a generation.
                Once concatenated together, will constitute a single audio file.
                If disabled, each chunk’s audio will be its own audio file, each with its headers.
        """

        if is_given(voice):
            self._opts.voice = voice
        if is_given(description):
            self._opts.description = description
        if is_given(speed):
            self._opts.speed = speed
        if is_given(format):
            self._opts.format = format
        if is_given(context):
            self._opts.context = context
        if is_given(instant_mode):
            self._opts.instant_mode = instant_mode
        if is_given(strip_headers):
            self._opts.strip_headers = strip_headers

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
        )


class ChunkedStream(tts.ChunkedStream):
    """Stream for Hume TTS JSON streaming API."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._client = tts._client

    async def _run(self) -> None:
        request_id = utils.shortuuid()

        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=DEFAULT_NUM_CHANNELS,
        )

        decode_task: asyncio.Task | None = None

        try:

            async def _decode_loop():
                utterance_options = {
                    "voice": self._opts.voice,
                    "description": self._opts.description,
                    "speed": self._opts.speed,
                }

                utterance_kwargs = {
                    "text": self._input_text,
                    **{k: v for k, v in utterance_options.items() if v is not None},
                }

                try:
                    utterance = PostedUtterance(**utterance_kwargs)

                    async for chunk in self._client.tts.synthesize_json_streaming(
                        utterances=[utterance],
                        context=self._opts.context,
                        format=self._opts.format,
                        instant_mode=self._opts.instant_mode,
                        strip_headers=self._opts.strip_headers,
                    ):
                        decoder.push(base64.b64decode(chunk.audio))

                finally:
                    decoder.end_input()

            decode_task = asyncio.create_task(_decode_loop())
            emitter = tts.SynthesizedAudioEmitter(
                event_ch=self._event_ch,
                request_id=request_id,
            )
            async for frame in decoder:
                emitter.push(frame)

            emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if decode_task:
                await utils.aio.gracefully_cancel(decode_task)
            await decoder.aclose()
