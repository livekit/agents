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
import weakref
from dataclasses import dataclass

import aiohttp

from hume import AsyncHumeClient
from hume.tts import Format, FormatWav, PostedContext, PostedUtterance, PostedUtteranceVoiceWithName
from livekit.agents import APIConnectionError, APIConnectOptions, tokenize, tts, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger

# Default audio settings
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_NUM_CHANNELS = 1

# Default TTS settings
DEFAULT_VOICE = PostedUtteranceVoiceWithName(name="Colton Rivers", provider="HUME_AI")

# text is required in PostedUtterance but it is declared as an empty string
# it will be overwritten when input tokens are received
DEFAULT_UTTERANCE = PostedUtterance(
    voice=DEFAULT_VOICE, speed=1, trailing_silence=0.35, description="", text=""
)


@dataclass
class _TTSOptions:
    """TTS options for Hume API"""

    api_key: str
    utterance_options: PostedUtterance
    context: PostedContext | None
    format: Format
    sample_rate: int
    split_utterances: bool
    strip_headers: bool
    num_generations: int
    instant_mode: bool
    word_tokenizer: tokenize.WordTokenizer


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        utterance_options: NotGivenOr[PostedUtterance] = NOT_GIVEN,
        context: NotGivenOr[PostedContext] = NOT_GIVEN,
        format: NotGivenOr[Format] = NOT_GIVEN,
        split_utterances: bool = False,
        num_generations: int = 1,
        instant_mode: bool = False,
        strip_headers: bool = True,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        word_tokenizer: tokenize.WordTokenizer | None = None,
        http_session: aiohttp.ClientSession | None = None,
        sample_rate: int = 24000,
    ) -> None:
        """Initialize the Hume TTS client.

        See https://dev.hume.ai/reference/text-to-speech-tts/synthesize-json-streaming for API doc

        Args:
            utterance_options (NotGivenOr[PostedUtterance]): Default options for utterances,
                including description, voice, and delivery controls.
            context (NotGivenOr[PostedContext]): Utterances to use as context for generating
                consistent speech style and prosody across multiple requests.
            format (NotGivenOr[Format]): Specifies the output audio file format (WAV, MP3 or PCM).
                Defaults to WAV format.
            split_utterances (bool): Controls how audio output is segmented in the response.
                When enabled (True), input utterances are split into natural-sounding segments.
                When disabled (False), maintains one-to-one mapping between input and output.
                Defaults to False.
            num_generations (int): Number of generations of the audio to produce.
                Must be between 1 and 5. Defaults to 1.
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
            sample_rate (int): Audio sample rate in Hz. Defaults to 24000.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            ),
            sample_rate=sample_rate,
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
            utterance_options=utterance_options
            if is_given(utterance_options)
            else DEFAULT_UTTERANCE,
            context=context if is_given(context) else None,
            format=format if is_given(format) else FormatWav(),
            api_key=self._api_key,
            sample_rate=self.sample_rate,
            split_utterances=split_utterances,
            num_generations=num_generations,
            strip_headers=strip_headers,
            instant_mode=instant_mode,
            word_tokenizer=word_tokenizer,
        )

        self._client = AsyncHumeClient(api_key=self._api_key)
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        utterance_options: NotGivenOr[PostedUtterance] = NOT_GIVEN,
        context: NotGivenOr[PostedContext] = NOT_GIVEN,
        format: NotGivenOr[Format] = NOT_GIVEN,
        split_utterances: NotGivenOr[bool] = NOT_GIVEN,
        num_generations: NotGivenOr[int] = NOT_GIVEN,
        instant_mode: NotGivenOr[bool] = NOT_GIVEN,
        strip_headers: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """Update TTS options for synthesizing speech.

        Args:
            utterance_options (NotGivenOr[PostedUtterance]): Options for utterances,
                including text, description, voice, and additional controls.
            context (Optional[PostedContext]): Utterances to use as context for generating
                consistent speech style and prosody across multiple requests.
            format (NotGivenOr[Format]): Specifies the output audio file format (WAV, MP3 or PCM).
            split_utterances (NotGivenOr[bool]): Controls how audio output is segmented.
                When True, utterances are split into natural-sounding segments.
                When False, maintains one-to-one mapping between input and output.
            num_generations (NotGivenOr[int]): Number of speech generations to produce (1-5).
            instant_mode (NotGivenOr[bool]): Enables ultra-low latency streaming.
                Reduces time to first audio chunk, recommended for real-time applications.
                Note: Incurs 10% higher cost when enabled.
            strip_headers (NotGivenOr[bool]): If enabled, the audio for the chunks of a generation.
                Once concatenated together, will constitute a single audio file.
                If disabled, each chunk’s audio will be its own audio file, each with its headers.
        """

        if is_given(utterance_options):
            # text is required in PostedUtterance but it is declared as an empty string
            # it will be overwritten when input tokens are received
            self._opts.utterance_options = PostedUtterance(
                description=utterance_options.description if utterance_options.description else "",
                voice=utterance_options.voice if utterance_options.voice else DEFAULT_VOICE,
                speed=utterance_options.speed if utterance_options.speed else 1,
                trailing_silence=utterance_options.trailing_silence
                if utterance_options.trailing_silence
                else 0.35,
                text="",
            )
        if is_given(format):
            self._opts.format = format
        if is_given(context):
            self._opts.context = context
        if is_given(split_utterances):
            self._opts.split_utterances = split_utterances
        if is_given(num_generations):
            self._opts.num_generations = num_generations
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

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        stream = SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            opts=self._opts,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close all streams and cleanup resources."""
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await super().aclose()


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
            sample_rate=self._opts.sample_rate,
            num_channels=DEFAULT_NUM_CHANNELS,
        )

        decode_task: asyncio.Task | None = None

        try:

            async def _decode_loop():
                try:
                    async for chunk in self._client.tts.synthesize_json_streaming(
                        utterances=[
                            PostedUtterance(
                                text=self._input_text,
                                description=self._opts.utterance_options.description,
                                voice=self._opts.utterance_options.voice,
                                speed=self._opts.utterance_options.speed,
                                trailing_silence=self._opts.utterance_options.trailing_silence,
                            )
                        ],
                        context=self._opts.context,
                        format=self._opts.format,
                        num_generations=self._opts.num_generations,
                        split_utterances=self._opts.split_utterances,
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

        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if decode_task:
                await utils.aio.gracefully_cancel(decode_task)
            await decoder.aclose()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        opts: _TTSOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._client = tts._client
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()

    @utils.log_exceptions(logger=logger)
    async def _create_text_stream(self):
        async def text_stream():
            async for word_stream in self._segments_ch:
                async for word in word_stream:
                    self._mark_started()
                    yield word.token

        return text_stream()

    @utils.log_exceptions(logger=logger)
    async def _tokenize_input(self):
        # Converts incoming text into WordStreams and sends them into _segments_ch
        word_stream = None
        async for input in self._input_ch:
            if isinstance(input, str):
                if word_stream is None:
                    word_stream = self._opts.word_tokenizer.stream()
                    self._segments_ch.send_nowait(word_stream)
                word_stream.push_text(input)
            elif isinstance(input, self._FlushSentinel):
                if word_stream:
                    word_stream.end_input()
                word_stream = None
        self._segments_ch.close()

    @utils.log_exceptions(logger=logger)
    async def _get_text_as_string(self, text_stream):
        full_text = ""

        async for chunk in text_stream:
            full_text += f"{chunk} "

        return full_text

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        input_task = asyncio.create_task(self._tokenize_input())

        try:
            text_stream = await self._create_text_stream()
            decoder = utils.codecs.AudioStreamDecoder(
                sample_rate=self._opts.sample_rate,
                num_channels=DEFAULT_NUM_CHANNELS,
            )

            async def decode_loop():
                try:
                    full_text = await self._get_text_as_string(text_stream)
                    logger.info("description %s", self._opts.utterance_options.description)
                    logger.info("speed %s", self._opts.utterance_options.speed)
                    logger.info(
                        "trailing silence %s", self._opts.utterance_options.trailing_silence
                    )

                    async for chunk in self._client.tts.synthesize_json_streaming(
                        utterances=[
                            PostedUtterance(
                                text=full_text,
                                description=self._opts.utterance_options.description,
                                voice=self._opts.utterance_options.voice,
                                speed=self._opts.utterance_options.speed,
                                trailing_silence=self._opts.utterance_options.trailing_silence,
                            )
                        ],
                        context=self._opts.context,
                        format=self._opts.format,
                        num_generations=self._opts.num_generations,
                        split_utterances=self._opts.split_utterances,
                        instant_mode=self._opts.instant_mode,
                        strip_headers=self._opts.strip_headers,
                    ):
                        decoder.push(base64.b64decode(chunk.audio))

                finally:
                    decoder.end_input()

            decode_task = asyncio.create_task(decode_loop())

            try:
                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                )

                async for frame in decoder:
                    emitter.push(frame)
                emitter.flush()
            finally:
                await utils.aio.gracefully_cancel(decode_task)
                await decoder.aclose()

        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(input_task)
