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
import os
import ssl
import weakref
from dataclasses import dataclass, replace

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# End-of-text sentinel expected by Baseten TTS WebSocket API
_END_SENTINEL = "__END__"


@dataclass
class _TTSOptions:
    language: str
    voice: str
    temperature: float
    max_tokens: int
    buffer_size: int
    api_key: str
    model_endpoint: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_endpoint: str | None = None,
        voice: str = "tara",
        language: str = "en",
        temperature: float = 0.6,
        max_tokens: int = 2000,
        buffer_size: int = 10,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Initialize the Baseten TTS with streaming support.

        Args:
            api_key (str): Baseten API key, or `BASETEN_API_KEY` env var.
            model_endpoint (str): Baseten model endpoint, or `BASETEN_MODEL_ENDPOINT` env var.
                For streaming, this should be the WebSocket endpoint or the HTTP endpoint
                (which will be converted to WebSocket automatically).
            voice (str): Speaker voice. Defaults to "tara".
            language (str): Language code. Defaults to "en".
            temperature (float): Sampling temperature. Defaults to 0.6.
            max_tokens (int): Maximum tokens for generation. Defaults to 2000.
            buffer_size (int): Words per chunk for streaming. Defaults to 10.
            http_session: Optional aiohttp session for connection pooling.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=24000,
            num_channels=1,
        )

        api_key = api_key or os.environ.get("BASETEN_API_KEY")

        if not api_key:
            raise ValueError(
                "Baseten API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `BASETEN_API_KEY` environment variable"
            )

        model_endpoint = model_endpoint or os.environ.get("BASETEN_MODEL_ENDPOINT")

        if not model_endpoint:
            raise ValueError(
                "The model endpoint is required, you can find it in the Baseten dashboard"
            )

        self._opts = _TTSOptions(
            voice=voice,
            language=language,
            temperature=temperature,
            max_tokens=max_tokens,
            buffer_size=buffer_size,
            api_key=api_key,
            model_endpoint=model_endpoint,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.basic.SentenceTokenizer()
        )

    @property
    def model(self) -> str:
        return "orpheus-tts"

    @property
    def provider(self) -> str:
        return "Baseten"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        buffer_size: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(max_tokens):
            self._opts.max_tokens = max_tokens
        if is_given(buffer_size):
            self._opts.buffer_size = buffer_size

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

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


class ChunkedStream(tts.ChunkedStream):
    """Synthesize text using the HTTP endpoint (non-streaming, for single requests)."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            tts=tts,
            input_text=input_text,
            conn_options=conn_options,
        )

        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            # Use the original HTTP endpoint for non-streaming
            http_endpoint = self._opts.model_endpoint
            if http_endpoint.endswith("/websocket"):
                http_endpoint = http_endpoint[:-10] + "/predict"

            async with self._tts._ensure_session().post(
                http_endpoint,
                headers={
                    "Authorization": f"Api-Key {self._opts.api_key}",
                },
                json={
                    "prompt": self._input_text,
                    "voice": self._opts.voice,
                    "temperature": self._opts.temperature,
                    "language": self._opts.language,
                },
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
                ssl=ssl_context,
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=24000,
                    num_channels=1,
                    mime_type="audio/pcm",
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Stream text to Baseten TTS via WebSocket and receive audio chunks."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        ws_url = self._opts.get_ws_url()
        headers = {"Authorization": f"Api-Key {self._opts.api_key}"}

        sent_tokenizer_stream = self._tts._sentence_tokenizer.stream()

        async def _input_task() -> None:
            """Read input text and feed to tokenizer."""
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_tokenizer_stream.flush()
                    continue
                sent_tokenizer_stream.push_text(data)
            sent_tokenizer_stream.end_input()

        async def _send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Send tokenized sentences as words to WebSocket."""
            async for ev in sent_tokenizer_stream:
                # Split sentence into words and send each word
                words = ev.token.strip().split()
                for word in words:
                    self._mark_started()
                    await ws.send_str(word)

            # Signal end of text
            await ws.send_str(_END_SENTINEL)
            logger.debug("Baseten TTS: sent END sentinel")

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Receive audio chunks from WebSocket."""
            segment_id = utils.shortuuid()
            output_emitter.start_segment(segment_id=segment_id)

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    # Raw PCM audio data
                    output_emitter.push(msg.data)
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    logger.debug("Baseten TTS: WebSocket closed")
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError(f"WebSocket error: {ws.exception()}")
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    # Baseten might send JSON messages for errors or status
                    logger.debug(f"Baseten TTS: received text message: {msg.data}")

            output_emitter.end_input()

        try:
            async with self._tts._ensure_session().ws_connect(
                ws_url,
                headers=headers,
                ssl=ssl_context,
                timeout=aiohttp.ClientTimeout(
                    total=None,  # No total timeout for streaming
                    sock_connect=self._conn_options.timeout,
                ),
            ) as ws:
                logger.debug(f"Baseten TTS: WebSocket connected to {ws_url}")

                # Send metadata first
                metadata = {
                    "voice": self._opts.voice,
                    "max_tokens": self._opts.max_tokens,
                    "buffer_size": self._opts.buffer_size,
                }
                await ws.send_json(metadata)
                logger.debug(f"Baseten TTS: sent metadata {metadata}")

                tasks = [
                    asyncio.create_task(_input_task()),
                    asyncio.create_task(_send_task(ws)),
                    asyncio.create_task(_recv_task(ws)),
                ]

                try:
                    await asyncio.gather(*tasks)
                finally:
                    await sent_tokenizer_stream.aclose()
                    await utils.aio.gracefully_cancel(*tasks)

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except aiohttp.ClientError as e:
            logger.exception("Baseten TTS WebSocket connection error")
            raise APIConnectionError() from e
        except Exception as e:
            logger.exception("Baseten TTS error")
            raise APIConnectionError() from e
