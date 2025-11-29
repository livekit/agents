# Copyright 2025 LiveKit, Inc.
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
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Any, Literal, Union, cast
from urllib.parse import urljoin

import aiohttp

from livekit.agents import tokenize, tts, utils
from livekit.agents._exceptions import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from .log import logger

DEFAULT_BIT_RATE = 64000
DEFAULT_ENCODING = "OGG_OPUS"
DEFAULT_MODEL = "inworld-tts-1"
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_URL = "https://api.inworld.ai/"
DEFAULT_WS_URL = "wss://api.inworld.ai/"
DEFAULT_VOICE = "Ashley"
DEFAULT_TEMPERATURE = 1.1
DEFAULT_SPEAKING_RATE = 1.0
DEFAULT_BUFFER_CHAR_THRESHOLD = 100
DEFAULT_MAX_BUFFER_DELAY_MS = 3000
NUM_CHANNELS = 1

Encoding = Union[Literal["LINEAR16", "MP3", "OGG_OPUS", "ALAW", "MULAW", "FLAC"], str]
TimestampType = Literal["TIMESTAMP_TYPE_UNSPECIFIED", "WORD", "CHARACTER"]
TextNormalization = Literal["APPLY_TEXT_NORMALIZATION_UNSPECIFIED", "ON", "OFF"]


@dataclass
class _TTSOptions:
    model: str
    encoding: Encoding
    voice: str
    sample_rate: int
    bit_rate: int
    speaking_rate: float
    temperature: float
    timestamp_type: NotGivenOr[TimestampType] = NOT_GIVEN
    text_normalization: NotGivenOr[TextNormalization] = NOT_GIVEN
    buffer_char_threshold: int = DEFAULT_BUFFER_CHAR_THRESHOLD
    max_buffer_delay_ms: int = DEFAULT_MAX_BUFFER_DELAY_MS

    @property
    def mime_type(self) -> str:
        if self.encoding == "MP3":
            return "audio/mpeg"
        elif self.encoding == "OGG_OPUS":
            return "audio/ogg"
        elif self.encoding == "FLAC":
            return "audio/flac"
        elif self.encoding in ("ALAW", "MULAW"):
            return "audio/basic"
        else:
            return "audio/wav"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[Encoding] = NOT_GIVEN,
        bit_rate: NotGivenOr[int] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        speaking_rate: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        timestamp_type: NotGivenOr[TimestampType] = NOT_GIVEN,
        text_normalization: NotGivenOr[TextNormalization] = NOT_GIVEN,
        buffer_char_threshold: NotGivenOr[int] = NOT_GIVEN,
        max_buffer_delay_ms: NotGivenOr[int] = NOT_GIVEN,
        base_url: str = DEFAULT_URL,
        ws_url: str = DEFAULT_WS_URL,
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Inworld TTS.

        Args:
            api_key (str, optional): The Inworld API key.
                If not provided, it will be read from the INWORLD_API_KEY environment variable.
            voice (str, optional): The voice to use. Defaults to "Ashley".
            model (str, optional): The Inworld model to use. Defaults to "inworld-tts-1".
            encoding (str, optional): The encoding to use. Defaults to "OGG_OPUS".
            bit_rate (int, optional): Bits per second of the audio. Defaults to 64000.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 48000.
            speaking_rate (float, optional): The speed of the voice, in the range [0.5, 1.5].
                Defaults to 1.0.
            temperature (float, optional): Determines the degree of randomness when sampling audio
                tokens to generate the response. Range [0, 2]. Defaults to 1.1.
            timestamp_type (str, optional): Controls timestamp metadata returned with the audio.
                Use "WORD" for word-level timestamps or "CHARACTER" for character-level.
                Useful for karaoke-style captions, word highlighting, and lipsync.
            text_normalization (str, optional): Controls text normalization. When "ON", numbers,
                dates, and abbreviations are expanded (e.g., "Dr." -> "Doctor"). When "OFF",
                text is read exactly as written. Defaults to automatic.
                _threshold (int, optional): For streaming, the minimum number of characters
                in the buffer that automatically triggers audio generation. Defaults to 100.
            max_buffer_delay_ms (int, optional): For streaming, the maximum time in ms to buffer
                before starting generation. Defaults to 3000.
            base_url (str, optional): The base URL for the Inworld TTS API.
                Defaults to "https://api.inworld.ai/".
            ws_url (str, optional): The WebSocket URL for streaming TTS.
                Defaults to "wss://api.inworld.ai/".
            http_session (aiohttp.ClientSession, optional): The HTTP session to use.
            tokenizer (tokenize.SentenceTokenizer, optional): The tokenizer to use for streaming.
                Defaults to `livekit.agents.tokenize.blingfire.SentenceTokenizer`.
        """
        if not is_given(sample_rate):
            sample_rate = DEFAULT_SAMPLE_RATE
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=is_given(timestamp_type)
                and timestamp_type != "TIMESTAMP_TYPE_UNSPECIFIED",
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        key = api_key if is_given(api_key) else os.getenv("INWORLD_API_KEY")
        if not key:
            raise ValueError("Inworld API key required. Set INWORLD_API_KEY or provide api_key.")

        self._authorization = f"Basic {key}"
        self._base_url = base_url
        self._ws_url = ws_url
        self._session = http_session

        self._opts = _TTSOptions(
            voice=voice if is_given(voice) else DEFAULT_VOICE,
            model=model if is_given(model) else DEFAULT_MODEL,
            encoding=encoding if is_given(encoding) else DEFAULT_ENCODING,
            bit_rate=bit_rate if is_given(bit_rate) else DEFAULT_BIT_RATE,
            sample_rate=sample_rate if is_given(sample_rate) else DEFAULT_SAMPLE_RATE,
            speaking_rate=speaking_rate if is_given(speaking_rate) else DEFAULT_SPEAKING_RATE,
            temperature=temperature if is_given(temperature) else DEFAULT_TEMPERATURE,
            timestamp_type=timestamp_type,
            text_normalization=text_normalization,
            buffer_char_threshold=buffer_char_threshold
            if is_given(buffer_char_threshold)
            else DEFAULT_BUFFER_CHAR_THRESHOLD,
            max_buffer_delay_ms=max_buffer_delay_ms
            if is_given(max_buffer_delay_ms)
            else DEFAULT_MAX_BUFFER_DELAY_MS,
        )

        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Inworld"

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        url = urljoin(self._ws_url, "/tts/v1/voice:streamBidirectional")
        ws = await asyncio.wait_for(
            session.ws_connect(url, headers={"Authorization": self._authorization}),
            timeout,
        )
        logger.debug("Established new Inworld TTS WebSocket connection")
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[Encoding] = NOT_GIVEN,
        bit_rate: NotGivenOr[int] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        speaking_rate: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        timestamp_type: NotGivenOr[TimestampType] = NOT_GIVEN,
        text_normalization: NotGivenOr[TextNormalization] = NOT_GIVEN,
        buffer_char_threshold: NotGivenOr[int] = NOT_GIVEN,
        max_buffer_delay_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS configuration options.

        Args:
            voice (str, optional): The voice to use.
            model (str, optional): The Inworld model to use.
            encoding (str, optional): The encoding to use.
            bit_rate (int, optional): Bits per second of the audio.
            sample_rate (int, optional): The audio sample rate in Hz.
            speaking_rate (float, optional): The speed of the voice.
            temperature (float, optional): Determines the degree of randomness when sampling audio
                tokens to generate the response.
            timestamp_type (str, optional): Controls timestamp metadata ("WORD" or "CHARACTER").
            text_normalization (str, optional): Controls text normalization ("ON" or "OFF").
            buffer_char_threshold (int, optional): For streaming, min characters before triggering.
            max_buffer_delay_ms (int, optional): For streaming, max time to buffer.
        """
        if is_given(voice):
            self._opts.voice = voice
        if is_given(model):
            self._opts.model = model
        if is_given(encoding):
            self._opts.encoding = encoding
        if is_given(bit_rate):
            self._opts.bit_rate = bit_rate
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(speaking_rate):
            self._opts.speaking_rate = speaking_rate
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(timestamp_type):
            self._opts.timestamp_type = cast(TimestampType, timestamp_type)
        if is_given(text_normalization):
            self._opts.text_normalization = cast(TextNormalization, text_normalization)
        if is_given(buffer_char_threshold):
            self._opts.buffer_char_threshold = buffer_char_threshold
        if is_given(max_buffer_delay_ms):
            self._opts.max_buffer_delay_ms = max_buffer_delay_ms

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def prewarm(self) -> None:
        self._pool.prewarm()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
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
        await self._pool.aclose()

    async def list_voices(self, language: str | None = None) -> list[dict[str, Any]]:
        """
        List all available voices in the workspace associated with the API key.

        Args:
            language (str, optional): ISO 639-1 language code to filter voices (e.g., 'en', 'es', 'fr').
        """
        url = urljoin(self._base_url, "tts/v1/voices")
        params = {}
        if language:
            params["filter"] = f"language={language}"

        async with self._ensure_session().get(
            url,
            headers={"Authorization": self._authorization},
            params=params,
        ) as resp:
            if not resp.ok:
                error_body = await resp.json()
                raise APIStatusError(
                    message=error_body.get("message"),
                    status_code=resp.status,
                    request_id=None,
                    body=None,
                )

            data = await resp.json()
            return cast(list[dict[str, Any]], data.get("voices", []))


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            audio_config: dict[str, Any] = {
                "audioEncoding": self._opts.encoding,
                "bitrate": self._opts.bit_rate,
                "sampleRateHertz": self._opts.sample_rate,
                "temperature": self._opts.temperature,
                "speakingRate": self._opts.speaking_rate,
            }

            body_params: dict[str, Any] = {
                "text": self._input_text,
                "voiceId": self._opts.voice,
                "modelId": self._opts.model,
                "audioConfig": audio_config,
                "temperature": self._opts.temperature,
            }
            if utils.is_given(self._opts.timestamp_type):
                body_params["timestampType"] = self._opts.timestamp_type
            if utils.is_given(self._opts.text_normalization):
                body_params["applyTextNormalization"] = self._opts.text_normalization

            async with self._tts._ensure_session().post(
                urljoin(self._tts._base_url, "/tts/v1/voice:stream"),
                headers={
                    "Authorization": self._tts._authorization,
                },
                json=body_params,
                timeout=aiohttp.ClientTimeout(sock_connect=self._conn_options.timeout),
                # large read_bufsize to avoid `ValueError: Chunk too big`
                read_bufsize=10 * 1024 * 1024,
            ) as resp:
                resp.raise_for_status()

                request_id = utils.shortuuid()
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=self._opts.mime_type,
                )

                async for raw_line in resp.content:
                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("failed to parse Inworld response line: %s", line)
                        continue

                    if result := data.get("result"):
                        # Handle timestamp info if present
                        if timestamp_info := result.get("timestampInfo"):
                            timed_strings = _parse_timestamp_info(timestamp_info)
                            if timed_strings:
                                output_emitter.push_timed_transcript(timed_strings)

                        if audio_content := result.get("audioContent"):
                            output_emitter.push(base64.b64decode(audio_content))
                            output_emitter.flush()
                    elif error := data.get("error"):
                        raise APIStatusError(
                            message=error.get("message"),
                            status_code=error.get("code"),
                            request_id=request_id,
                            body=None,
                        )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._context_id = utils.shortuuid()
        self._sent_tokenizer_stream = self._tts._sentence_tokenizer.stream()
        self._input_flushed = asyncio.Event()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type=self._opts.mime_type,
            stream=True,
        )

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                await self._create_context(ws)

                tasks = [
                    asyncio.create_task(self._input_task()),
                    asyncio.create_task(self._send_task(ws)),
                    asyncio.create_task(self._recv_task(ws, output_emitter, request_id)),
                ]

                try:
                    await asyncio.gather(*tasks)
                finally:
                    await utils.aio.gracefully_cancel(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except APIError:
            raise
        except Exception as e:
            logger.error(
                "Inworld WebSocket connection error",
                extra={"context_id": self._context_id, "error": e},
            )
            raise APIConnectionError() from e
        finally:
            await self._sent_tokenizer_stream.aclose()

    async def _create_context(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Create a new context on the WebSocket connection."""
        create_msg: dict[str, Any] = {
            "create": {
                "voiceId": self._opts.voice,
                "modelId": self._opts.model,
                "audioConfig": {
                    "audioEncoding": self._opts.encoding,
                    "sampleRateHertz": self._opts.sample_rate,
                    "bitrate": self._opts.bit_rate,
                    "speakingRate": self._opts.speaking_rate,
                },
                "temperature": self._opts.temperature,
                "bufferCharThreshold": self._opts.buffer_char_threshold,
                "maxBufferDelayMs": self._opts.max_buffer_delay_ms,
            },
            "contextId": self._context_id,
        }
        if is_given(self._opts.timestamp_type):
            create_msg["create"]["timestampType"] = self._opts.timestamp_type
        if is_given(self._opts.text_normalization):
            create_msg["create"]["applyTextNormalization"] = self._opts.text_normalization
        await ws.send_str(json.dumps(create_msg))

    async def _input_task(self) -> None:
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                self._sent_tokenizer_stream.flush()
                continue
            self._sent_tokenizer_stream.push_text(data)
        self._sent_tokenizer_stream.end_input()

    async def _send_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for ev in self._sent_tokenizer_stream:
            send_msg = {
                "send_text": {
                    "text": ev.token,
                },
                "contextId": self._context_id,
            }
            self._mark_started()
            await ws.send_str(json.dumps(send_msg))

        # Flush remaining text and close the context
        flush_msg = {"flush_context": {}, "contextId": self._context_id}
        await ws.send_str(json.dumps(flush_msg))

        close_msg = {"close_context": {}, "contextId": self._context_id}
        await ws.send_str(json.dumps(close_msg))
        self._input_flushed.set()

    async def _recv_task(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        output_emitter: tts.AudioEmitter,
        request_id: str,
    ) -> None:
        current_segment_id: str | None = None

        while True:
            try:
                timeout = 0.5 if self._input_flushed.is_set() else self._conn_options.timeout
                msg = await ws.receive(timeout=timeout)
            except asyncio.TimeoutError:
                if self._input_flushed.is_set():
                    logger.debug(
                        "Inworld stream completed",
                        extra={"context_id": self._context_id},
                    )
                    output_emitter.end_input()
                    return
                raise

            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                if self._input_flushed.is_set():
                    logger.debug(
                        "Inworld WebSocket closed after flush",
                        extra={"context_id": self._context_id},
                    )
                    output_emitter.end_input()
                    return
                logger.error(
                    "Inworld WebSocket connection closed unexpectedly",
                    extra={"context_id": self._context_id},
                )
                raise APIStatusError(
                    "Inworld connection closed unexpectedly", request_id=request_id
                )

            if msg.type != aiohttp.WSMsgType.TEXT:
                logger.warning("unexpected Inworld message type %s", msg.type)
                continue

            data = json.loads(msg.data)
            result = data.get("result", {})
            result_context_id = result.get("contextId")

            # Check for errors in status
            status = result.get("status", {})
            if status.get("code", 0) != 0:
                raise APIError(f"Inworld error: {status.get('message', 'Unknown error')}")

            # Handle context created response
            if result.get("contextCreated"):
                logger.debug(
                    "Inworld context created",
                    extra={"context_id": result_context_id},
                )
                continue

            # Handle context closed response - this is the completion signal
            if result.get("contextClosed"):
                logger.debug(
                    "Inworld context closed",
                    extra={"context_id": result_context_id},
                )
                output_emitter.end_input()
                return

            # Handle audio chunks
            if audio_chunk := result.get("audioChunk"):
                if current_segment_id is None:
                    current_segment_id = result_context_id or self._context_id
                    output_emitter.start_segment(segment_id=current_segment_id)

                # Handle timestamp info if present
                if timestamp_info := audio_chunk.get("timestampInfo"):
                    timed_strings = _parse_timestamp_info(timestamp_info)
                    for ts in timed_strings:
                        output_emitter.push_timed_transcript(ts)

                # Handle audio content
                if audio_content := audio_chunk.get("audioContent"):
                    output_emitter.push(base64.b64decode(audio_content))


def _parse_timestamp_info(timestamp_info: dict[str, Any]) -> list[TimedString]:
    """Parse timestamp info from API response into TimedString objects."""
    timed_strings: list[TimedString] = []

    # Handle word-level alignment
    if word_align := timestamp_info.get("wordAlignment"):
        words = word_align.get("words", [])
        starts = word_align.get("wordStartTimeSeconds", [])
        ends = word_align.get("wordEndTimeSeconds", [])

        for word, start, end in zip(words, starts, ends):
            timed_strings.append(TimedString(word, start_time=start, end_time=end))

    # Handle character-level alignment
    if char_align := timestamp_info.get("characterAlignment"):
        chars = char_align.get("characters", [])
        starts = char_align.get("characterStartTimeSeconds", [])
        ends = char_align.get("characterEndTimeSeconds", [])

        for char, start, end in zip(chars, starts, ends):
            timed_strings.append(TimedString(char, start_time=start, end_time=end))

    return timed_strings
