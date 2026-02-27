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
import dataclasses
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Any, Literal

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    Language,
    tokenize,
    tts,
    utils,
)
from livekit.agents.tokenize.basic import split_words
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from .log import logger
from .models import TTSEncoding, TTSModels

# by default, use 22.05kHz sample rate at 32kbps
# in our testing,  reduce TTFB by about ~110ms
_DefaultEncoding: TTSEncoding = "mp3_22050_32"


def _sample_rate_from_format(output_format: TTSEncoding) -> int:
    split = output_format.split("_")  # e.g: mp3_44100
    return int(split[1])


def _encoding_to_mimetype(encoding: TTSEncoding) -> str:
    if encoding.startswith("mp3"):
        return "audio/mp3"
    elif encoding.startswith("opus"):
        return "audio/opus"
    elif encoding.startswith("pcm"):
        return "audio/pcm"
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")


@dataclass
class VoiceSettings:
    stability: float  # [0.0 - 1.0]
    similarity_boost: float  # [0.0 - 1.0]
    style: NotGivenOr[float] = NOT_GIVEN  # [0.0 - 1.0]
    speed: NotGivenOr[float] = NOT_GIVEN  # [0.7 - 1.2]
    use_speaker_boost: NotGivenOr[bool] = NOT_GIVEN


@dataclass
class Voice:
    id: str
    name: str
    category: str


@dataclass
class PronunciationDictionaryLocator:
    pronunciation_dictionary_id: str
    version_id: str


DEFAULT_VOICE_ID = "bIHbv24MWmeRgasZH58o"
API_BASE_URL_V1 = "https://api.elevenlabs.io/v1"
AUTHORIZATION_HEADER = "xi-api-key"
WS_INACTIVITY_TIMEOUT = 180


def _validate_voice_settings(settings: VoiceSettings) -> None:
    """Log warnings for VoiceSettings parameters outside documented ElevenLabs API ranges."""
    if not (0.0 <= settings.stability <= 1.0):
        logger.warning(
            "voice_settings.stability must be between 0.0 and 1.0",
            extra={"stability": settings.stability},
        )
    if not (0.0 <= settings.similarity_boost <= 1.0):
        logger.warning(
            "voice_settings.similarity_boost must be between 0.0 and 1.0",
            extra={"similarity_boost": settings.similarity_boost},
        )
    if is_given(settings.style) and not (0.0 <= settings.style <= 1.0):
        logger.warning(
            "voice_settings.style must be between 0.0 and 1.0",
            extra={"style": settings.style},
        )
    if is_given(settings.speed) and not (0.7 <= settings.speed <= 1.2):
        logger.warning(
            "voice_settings.speed must be between 0.7 and 1.2",
            extra={"speed": settings.speed},
        )


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        voice_settings: NotGivenOr[VoiceSettings] = NOT_GIVEN,
        model: TTSModels | str = "eleven_turbo_v2_5",
        encoding: NotGivenOr[TTSEncoding] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        streaming_latency: NotGivenOr[int] = NOT_GIVEN,
        inactivity_timeout: int = WS_INACTIVITY_TIMEOUT,
        auto_mode: NotGivenOr[bool] = NOT_GIVEN,
        apply_text_normalization: Literal["auto", "off", "on"] = "auto",
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer | tokenize.SentenceTokenizer] = NOT_GIVEN,
        enable_ssml_parsing: bool = False,
        enable_logging: bool = True,
        chunk_length_schedule: NotGivenOr[list[int]] = NOT_GIVEN,  # range is [50, 500]
        http_session: aiohttp.ClientSession | None = None,
        language: NotGivenOr[str] = NOT_GIVEN,
        sync_alignment: bool = True,
        preferred_alignment: Literal["normalized", "original"] = "normalized",
        pronunciation_dictionary_locators: NotGivenOr[
            list[PronunciationDictionaryLocator]
        ] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of ElevenLabs TTS.

        Args:
            voice_id (str): Voice ID. Defaults to `DEFAULT_VOICE_ID`.
            voice_settings (NotGivenOr[VoiceSettings]): Voice settings.
            model (TTSModels | str): TTS model to use. Defaults to "eleven_turbo_v2_5".
            api_key (NotGivenOr[str]): ElevenLabs API key. Can be set via argument or `ELEVEN_API_KEY` environment variable.
            base_url (NotGivenOr[str]): Custom base URL for the API. Optional.
            streaming_latency (NotGivenOr[int]): Optimize for streaming latency, defaults to 0 - disabled. 4 for max latency optimizations. deprecated
            inactivity_timeout (int): Inactivity timeout in seconds for the websocket connection. Defaults to 300.
            auto_mode (bool): Reduces latency by disabling chunk schedule and buffers. Sentence tokenizer will be used to synthesize one sentence at a time. Defaults to True.
            word_tokenizer (NotGivenOr[tokenize.WordTokenizer | tokenize.SentenceTokenizer]): Tokenizer for processing text. Defaults to basic WordTokenizer when auto_mode=False, `livekit.agents.tokenize.blingfire.SentenceTokenizer` otherwise.
            enable_ssml_parsing (bool): Enable SSML parsing for input text. Defaults to False.
            enable_logging (bool): Enable logging of the request. When set to false, zero retention mode will be used. Defaults to True.
            chunk_length_schedule (NotGivenOr[list[int]]): Schedule for chunk lengths, ranging from 50 to 500. Defaults are [120, 160, 250, 290].
            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests. Optional.
            language (NotGivenOr[str]): Language code for the TTS model, as of 10/24/24 only valid for "eleven_turbo_v2_5".
            sync_alignment (bool): Enable sync alignment for the TTS model. Defaults to True.
            preferred_alignment (Literal["normalized", "original"]): Use normalized or original alignment. Defaults to "normalized".
            pronunciation_dictionary_locators (NotGivenOr[list[PronunciationDictionaryLocator]]): List of pronunciation dictionary locators to use for pronunciation control.
        """  # noqa: E501

        if not is_given(encoding):
            encoding = _DefaultEncoding

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=sync_alignment,
            ),
            sample_rate=_sample_rate_from_format(encoding),
            num_channels=1,
        )

        elevenlabs_api_key = api_key if is_given(api_key) else os.environ.get("ELEVEN_API_KEY")
        if not elevenlabs_api_key:
            raise ValueError(
                "ElevenLabs API key is required, either as argument or set ELEVEN_API_KEY environmental variable"  # noqa: E501
            )

        if not is_given(auto_mode):
            auto_mode = True

        if not is_given(word_tokenizer):
            word_tokenizer = (
                tokenize.basic.WordTokenizer(ignore_punctuation=False)
                if not auto_mode
                else tokenize.blingfire.SentenceTokenizer()
            )
        elif auto_mode and not isinstance(word_tokenizer, tokenize.SentenceTokenizer):
            logger.warning(
                "auto_mode is enabled, it expects full sentences or phrases, "
                "please provide a SentenceTokenizer instead of a WordTokenizer."
            )

        if is_given(voice_settings):
            _validate_voice_settings(voice_settings)

        self._opts = _TTSOptions(
            voice_id=voice_id,
            voice_settings=voice_settings,
            model=model,
            api_key=elevenlabs_api_key,
            base_url=base_url if is_given(base_url) else API_BASE_URL_V1,
            encoding=encoding,
            sample_rate=self.sample_rate,
            streaming_latency=streaming_latency,
            word_tokenizer=word_tokenizer,
            chunk_length_schedule=chunk_length_schedule,
            enable_ssml_parsing=enable_ssml_parsing,
            enable_logging=enable_logging,
            language=Language(language) if is_given(language) else NOT_GIVEN,
            inactivity_timeout=inactivity_timeout,
            sync_alignment=sync_alignment,
            auto_mode=auto_mode,
            apply_text_normalization=apply_text_normalization,
            preferred_alignment=preferred_alignment,
            pronunciation_dictionary_locators=pronunciation_dictionary_locators,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

        self._current_connection: _Connection | None = None
        self._connection_lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "ElevenLabs"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def list_voices(self) -> list[Voice]:
        async with self._ensure_session().get(
            f"{self._opts.base_url}/voices",
            headers={AUTHORIZATION_HEADER: self._opts.api_key},
        ) as resp:
            return _dict_to_voices_list(await resp.json())

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        voice_settings: NotGivenOr[VoiceSettings] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        pronunciation_dictionary_locators: NotGivenOr[
            list[PronunciationDictionaryLocator]
        ] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            voice_id (NotGivenOr[str]): Voice ID.
            voice_settings (NotGivenOr[VoiceSettings]): Voice settings.
            model (NotGivenOr[TTSModels | str]): TTS model to use.
            language (NotGivenOr[str]): Language code for the TTS model.
            pronunciation_dictionary_locators (NotGivenOr[list[PronunciationDictionaryLocator]]): List of pronunciation dictionary locators.
        """
        changed = False

        if is_given(model) and model != self._opts.model:
            self._opts.model = model
            changed = True

        if is_given(voice_id) and voice_id != self._opts.voice_id:
            self._opts.voice_id = voice_id
            changed = True

        if is_given(voice_settings):
            _validate_voice_settings(voice_settings)
            self._opts.voice_settings = voice_settings
            changed = True

        if is_given(language):
            language = Language(language)
            if language != self._opts.language:
                self._opts.language = language
                changed = True

        if is_given(pronunciation_dictionary_locators):
            self._opts.pronunciation_dictionary_locators = pronunciation_dictionary_locators
            changed = True

        if changed and self._current_connection:
            self._current_connection.mark_non_current()
            self._current_connection = None

    async def current_connection(self) -> _Connection:
        """Get the current connection, creating one if needed"""
        async with self._connection_lock:
            if (
                self._current_connection
                and self._current_connection.is_current
                and not self._current_connection._closed
            ):
                return self._current_connection

            session = self._ensure_session()
            conn = _Connection(self._opts, session)
            await conn.connect()
            self._current_connection = conn
            return conn

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()

        if self._current_connection:
            await self._current_connection.aclose()
            self._current_connection = None


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        voice_settings = (
            _strip_nones(dataclasses.asdict(self._opts.voice_settings))
            if is_given(self._opts.voice_settings)
            else None
        )
        try:
            async with self._tts._ensure_session().post(
                _synthesize_url(self._opts),
                headers={AUTHORIZATION_HEADER: self._opts.api_key},
                json={
                    "text": self._input_text,
                    "model_id": self._opts.model,
                    "voice_settings": voice_settings,
                },
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                resp.raise_for_status()

                if not resp.content_type.startswith("audio/"):
                    content = await resp.text()
                    raise APIError(message="11labs returned non-audio data", body=content)

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type=_encoding_to_mimetype(self._opts.encoding),
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streamed API using websockets

    Uses multi-stream API:
    https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-multi-stream-input
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._context_id = utils.shortuuid()
        self._sent_tokenizer_stream = self._opts.word_tokenizer.stream()
        self._text_buffer = ""
        self._start_times_ms: list[int] = []
        self._durations_ms: list[int] = []
        self._connection: _Connection | None = None

    async def aclose(self) -> None:
        await self._sent_tokenizer_stream.aclose()
        await super().aclose()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=self._context_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            stream=True,
            mime_type=_encoding_to_mimetype(self._opts.encoding),
        )
        output_emitter.start_segment(segment_id=self._context_id)

        connection: _Connection
        try:
            connection = await asyncio.wait_for(
                self._tts.current_connection(), self._conn_options.timeout
            )
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            raise APIConnectionError("could not connect to ElevenLabs") from e

        waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        connection.register_stream(self, output_emitter, waiter)

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._sent_tokenizer_stream.flush()
                    continue
                self._sent_tokenizer_stream.push_text(data)
            self._sent_tokenizer_stream.end_input()

        async def _sentence_stream_task() -> None:
            flush_on_chunk = (
                isinstance(self._opts.word_tokenizer, tokenize.SentenceTokenizer)
                and is_given(self._opts.auto_mode)
                and self._opts.auto_mode
            )
            xml_content: list[str] = []
            async for data in self._sent_tokenizer_stream:
                text = data.token
                # send xml tags fully formed
                xml_start_tokens = ["<phoneme", "<break"]
                xml_end_tokens = ["</phoneme>", "/>"]

                if (
                    self._opts.enable_ssml_parsing
                    and any(text.startswith(start) for start in xml_start_tokens)
                    or xml_content
                ):
                    xml_content.append(text)

                    if any(text.find(end) > -1 for end in xml_end_tokens):
                        text = (
                            self._opts.word_tokenizer.format_words(xml_content)
                            if isinstance(self._opts.word_tokenizer, tokenize.WordTokenizer)
                            else " ".join(xml_content)
                        )
                        xml_content = []
                    else:
                        continue

                formatted_text = f"{text} "  # must always end with a space
                # when using auto_mode, we are flushing for each sentence
                connection.send_content(
                    _SynthesizeContent(self._context_id, formatted_text, flush=flush_on_chunk)
                )
                self._mark_started()

            if xml_content:
                logger.warning("ElevenLabs stream ended with incomplete xml content")

            connection.send_content(_SynthesizeContent(self._context_id, "", flush=True))
            connection.close_context(self._context_id)

        input_t = asyncio.create_task(_input_task())
        stream_t = asyncio.create_task(_sentence_stream_task())

        try:
            await waiter
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            if isinstance(e, APIStatusError):
                raise e
            raise APIStatusError("Could not synthesize") from e
        finally:
            output_emitter.end_segment()
            await utils.aio.gracefully_cancel(input_t, stream_t)


@dataclass
class _TTSOptions:
    api_key: str
    voice_id: str
    voice_settings: NotGivenOr[VoiceSettings]
    model: TTSModels | str
    language: NotGivenOr[Language]
    base_url: str
    encoding: TTSEncoding
    sample_rate: int
    streaming_latency: NotGivenOr[int]
    word_tokenizer: tokenize.WordTokenizer | tokenize.SentenceTokenizer
    chunk_length_schedule: NotGivenOr[list[int]]
    enable_ssml_parsing: bool
    enable_logging: bool
    inactivity_timeout: int
    sync_alignment: bool
    apply_text_normalization: Literal["auto", "on", "off"]
    preferred_alignment: Literal["normalized", "original"]
    auto_mode: NotGivenOr[bool]
    pronunciation_dictionary_locators: NotGivenOr[list[PronunciationDictionaryLocator]]


@dataclass
class _SynthesizeContent:
    context_id: str
    text: str
    flush: bool = False


@dataclass
class _CloseContext:
    context_id: str


@dataclass
class _StreamData:
    emitter: tts.AudioEmitter
    stream: SynthesizeStream
    waiter: asyncio.Future[None]
    timeout_timer: asyncio.TimerHandle | None = None


class _Connection:
    """Manages a single WebSocket connection with send/recv loops for multi-context TTS"""

    def __init__(self, opts: _TTSOptions, session: aiohttp.ClientSession):
        self._opts = opts
        self._session = session
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._is_current = True
        self._active_contexts: set[str] = set()
        self._input_queue = utils.aio.Chan[_SynthesizeContent | _CloseContext]()

        self._context_data: dict[str, _StreamData] = {}

        self._send_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None
        self._closed = False

    @property
    def voice_id(self) -> str:
        return self._opts.voice_id

    @property
    def is_current(self) -> bool:
        return self._is_current

    def mark_non_current(self) -> None:
        """Mark this connection as no longer current - it will shut down when drained"""
        self._is_current = False

    async def connect(self) -> None:
        """Establish WebSocket connection and start send/recv loops"""
        if self._ws or self._closed:
            return

        url = _multi_stream_url(self._opts)
        headers = {AUTHORIZATION_HEADER: self._opts.api_key}
        self._ws = await self._session.ws_connect(url, headers=headers)

        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())

    def register_stream(
        self, stream: SynthesizeStream, emitter: tts.AudioEmitter, done_fut: asyncio.Future[None]
    ) -> None:
        """Register a new synthesis stream with this connection"""
        context_id = stream._context_id
        self._context_data[context_id] = _StreamData(
            emitter=emitter, stream=stream, waiter=done_fut
        )

    def send_content(self, content: _SynthesizeContent) -> None:
        """Send synthesis content to the connection"""
        if self._closed or not self._ws or self._ws.closed:
            raise APIConnectionError("WebSocket connection is closed")
        self._input_queue.send_nowait(content)

    def close_context(self, context_id: str) -> None:
        """Close a specific context"""
        if self._closed or not self._ws or self._ws.closed:
            raise APIConnectionError("WebSocket connection is closed")
        self._input_queue.send_nowait(_CloseContext(context_id))

    async def _send_loop(self) -> None:
        """Send loop - processes messages from input queue"""
        try:
            while not self._closed:
                try:
                    msg = await self._input_queue.recv()
                except utils.aio.ChanClosed:
                    break

                if not self._ws or self._ws.closed:
                    break

                if isinstance(msg, _SynthesizeContent):
                    is_new_context = msg.context_id not in self._active_contexts

                    if is_new_context:
                        voice_settings = (
                            _strip_nones(dataclasses.asdict(self._opts.voice_settings))
                            if is_given(self._opts.voice_settings)
                            else {}
                        )
                        init_pkt: dict[str, Any] = {
                            "text": " ",
                            "voice_settings": voice_settings,
                            "context_id": msg.context_id,
                        }
                        if is_given(self._opts.pronunciation_dictionary_locators):
                            init_pkt["pronunciation_dictionary_locators"] = [
                                {
                                    "pronunciation_dictionary_id": locator.pronunciation_dictionary_id,
                                    "version_id": locator.version_id,
                                }
                                for locator in self._opts.pronunciation_dictionary_locators
                            ]
                        await self._ws.send_json(init_pkt)
                        self._active_contexts.add(msg.context_id)

                    pkt: dict[str, Any] = {
                        "text": msg.text,
                        "context_id": msg.context_id,
                    }
                    if msg.flush:
                        pkt["flush"] = True

                    # start timeout timer for this context
                    self._start_timeout_timer(msg.context_id)

                    await self._ws.send_json(pkt)

                elif isinstance(msg, _CloseContext):
                    if msg.context_id in self._active_contexts:
                        close_pkt = {
                            "context_id": msg.context_id,
                            "close_context": True,
                        }
                        await self._ws.send_json(close_pkt)

        except Exception as e:
            logger.warning("send loop error", exc_info=e)
        finally:
            if not self._closed:
                await self.aclose()

    async def _recv_loop(self) -> None:
        """Receive loop - processes messages from WebSocket"""
        try:
            while not self._closed and self._ws and not self._ws.closed:
                msg = await self._ws.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if not self._closed and len(self._context_data) > 0:
                        # websocket will be closed after all contexts are closed
                        logger.warning("websocket closed unexpectedly")
                    break

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                context_id = data.get("contextId")
                ctx = self._context_data.get(context_id) if context_id is not None else None

                if error := data.get("error"):
                    logger.error(
                        "elevenlabs tts returned error",
                        extra={"context_id": context_id, "error": error, "data": data},
                    )
                    if context_id is not None:
                        if ctx and not ctx.waiter.done():
                            ctx.waiter.set_exception(APIError(message=error))
                        self._cleanup_context(context_id)
                    continue

                if ctx is None:
                    logger.warning(
                        "unexpected message received from elevenlabs tts", extra={"data": data}
                    )
                    continue

                emitter = ctx.emitter
                stream = ctx.stream

                # ensure alignment
                alignment = (
                    data.get("normalizedAlignment")
                    if self._opts.preferred_alignment == "normalized"
                    else data.get("alignment")
                )
                if alignment and stream is not None:
                    chars = alignment["chars"]
                    starts = alignment.get("charStartTimesMs") or alignment.get("charsStartTimesMs")
                    durs = alignment.get("charDurationsMs") or alignment.get("charsDurationsMs")
                    if starts and durs and len(chars) == len(durs) and len(starts) == len(durs):
                        stream._text_buffer += "".join(chars)
                        # in case item in chars has multiple characters
                        for char, start, dur in zip(chars, starts, durs, strict=False):
                            if len(char) > 1:
                                stream._start_times_ms += [start] * (len(char) - 1)
                                stream._durations_ms += [0] * (len(char) - 1)
                            stream._start_times_ms.append(start)
                            stream._durations_ms.append(dur)

                        timed_words, stream._text_buffer = _to_timed_words(
                            stream._text_buffer, stream._start_times_ms, stream._durations_ms
                        )
                        emitter.push_timed_transcript(timed_words)
                        stream._start_times_ms = stream._start_times_ms[-len(stream._text_buffer) :]
                        stream._durations_ms = stream._durations_ms[-len(stream._text_buffer) :]

                if data.get("audio"):
                    b64data = base64.b64decode(data["audio"])
                    emitter.push(b64data)
                    if ctx.timeout_timer:
                        ctx.timeout_timer.cancel()

                if data.get("isFinal"):
                    if stream is not None:
                        timed_words, _ = _to_timed_words(
                            stream._text_buffer,
                            stream._start_times_ms,
                            stream._durations_ms,
                            flush=True,
                        )
                        emitter.push_timed_transcript(timed_words)

                    if not ctx.waiter.done():
                        ctx.waiter.set_result(None)
                    self._cleanup_context(context_id)

                    if not self._is_current and not self._active_contexts:
                        logger.debug("no active contexts, shutting down connection")
                        break
        except Exception as e:
            logger.warning("recv loop error", exc_info=e)
            for ctx in self._context_data.values():
                if not ctx.waiter.done():
                    ctx.waiter.set_exception(e)
                if ctx.timeout_timer:
                    ctx.timeout_timer.cancel()
            self._context_data.clear()
        finally:
            if not self._closed:
                await self.aclose()

    def _cleanup_context(self, context_id: str) -> None:
        """Clean up context state"""
        ctx = self._context_data.pop(context_id, None)
        if ctx and ctx.timeout_timer:
            ctx.timeout_timer.cancel()

        self._active_contexts.discard(context_id)

    def _start_timeout_timer(self, context_id: str) -> None:
        """Start a timeout timer for a context"""
        if not (ctx := self._context_data.get(context_id)) or ctx.timeout_timer:
            return

        timeout = ctx.stream._conn_options.timeout

        def _on_timeout() -> None:
            if not ctx.waiter.done():
                ctx.waiter.set_exception(
                    APITimeoutError(f"11labs tts timed out after {timeout} seconds")
                )
            self._cleanup_context(context_id)

        ctx.timeout_timer = asyncio.get_event_loop().call_later(timeout, _on_timeout)

    async def aclose(self) -> None:
        """Close the connection and clean up"""
        if self._closed:
            return

        self._closed = True
        self._input_queue.close()

        for ctx in self._context_data.values():
            if not ctx.waiter.done():
                # do not cancel the future as it becomes difficult to catch
                # all pending tasks will be aborted with an exception
                ctx.waiter.set_exception(APIStatusError("connection closed"))
            if ctx.timeout_timer:
                ctx.timeout_timer.cancel()
        self._context_data.clear()

        if self._ws:
            await self._ws.close()

        if self._send_task:
            await utils.aio.gracefully_cancel(self._send_task)
        if self._recv_task:
            await utils.aio.gracefully_cancel(self._recv_task)

        self._ws = None


def _dict_to_voices_list(data: dict[str, Any]) -> list[Voice]:
    voices: list[Voice] = []
    for voice in data["voices"]:
        voices.append(Voice(id=voice["voice_id"], name=voice["name"], category=voice["category"]))

    return voices


def _strip_nones(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if is_given(v) and v is not None}


def _synthesize_url(opts: _TTSOptions) -> str:
    base_url = opts.base_url
    voice_id = opts.voice_id
    model_id = opts.model
    output_format = opts.encoding
    url = (
        f"{base_url}/text-to-speech/{voice_id}/stream?"
        f"model_id={model_id}&output_format={output_format}"
    )
    if is_given(opts.streaming_latency):
        url += f"&optimize_streaming_latency={opts.streaming_latency}"
    return url


def _multi_stream_url(opts: _TTSOptions) -> str:
    base_url = opts.base_url.replace("https://", "wss://").replace("http://", "ws://")
    voice_id = opts.voice_id
    url = f"{base_url}/text-to-speech/{voice_id}/multi-stream-input?"
    params = []
    params.append(f"model_id={opts.model}")
    params.append(f"output_format={opts.encoding}")
    if is_given(opts.language):
        params.append(f"language_code={opts.language.language}")
    params.append(f"enable_ssml_parsing={str(opts.enable_ssml_parsing).lower()}")
    params.append(f"enable_logging={str(opts.enable_logging).lower()}")
    params.append(f"inactivity_timeout={opts.inactivity_timeout}")
    params.append(f"apply_text_normalization={opts.apply_text_normalization}")
    if opts.sync_alignment:
        params.append("sync_alignment=true")
    if is_given(opts.auto_mode):
        params.append(f"auto_mode={str(opts.auto_mode).lower()}")
    url += "&".join(params)
    return url


def _to_timed_words(
    text: str, start_times_ms: list[int], durations_ms: list[int], flush: bool = False
) -> tuple[list[TimedString], str]:
    """Return timed words and the remaining text"""
    if not text:
        return [], ""

    timestamps = start_times_ms + [start_times_ms[-1] + durations_ms[-1]]  # N+1

    words = split_words(text, ignore_punctuation=False, split_character=True)
    timed_words = []
    _, start_indices, _ = zip(*words, strict=False)
    end = 0
    # we don't know if the last word is complete, always leave it as remaining
    for start, end in zip(start_indices[:-1], start_indices[1:], strict=False):
        start_t = timestamps[start] / 1000
        end_t = timestamps[end] / 1000
        timed_words.append(
            TimedString(text=text[start:end], start_time=start_t, end_time=end_t),
        )

    if flush:
        start_t = timestamps[end] / 1000
        end_t = timestamps[-1] / 1000
        timed_words.append(TimedString(text=text[end:], start_time=start_t, end_time=end_t))
        end = len(text)

    return timed_words, text[end:]
