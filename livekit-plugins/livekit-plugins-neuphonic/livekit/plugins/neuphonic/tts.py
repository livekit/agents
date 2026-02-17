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

from __future__ import annotations  # noqa: I001

import asyncio
import base64
import json
import os
import weakref
from dataclasses import dataclass, replace

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    create_api_error_from_http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger  # noqa: I001
from .models import TTSLangCodes  # noqa: I001

API_AUTH_HEADER = "x-api-key"


@dataclass
class _TTSOptions:
    lang_code: TTSLangCodes | str
    encoding: str
    sample_rate: int
    voice_id: str
    speed: float | None
    api_key: str
    jwt_token: str | None
    base_url: str
    word_tokenizer: tokenize.WordTokenizer

    def get_http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_ws_url(self, path: str) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}{path}"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        jwt_token: str | None = None,
        lang_code: TTSLangCodes | str = "en",
        encoding: str = "pcm_linear",
        voice_id: str = "8e9c4bc8-3979-48ab-8626-df53befc2090",
        speed: float | None = 1.0,
        sample_rate: int = 22050,
        http_session: aiohttp.ClientSession | None = None,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer] = NOT_GIVEN,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        base_url: str = "https://api.neuphonic.com",
    ) -> None:
        """
        Create a new instance of NeuPhonic TTS.

        See https://docs.neuphonic.com for more details on the NeuPhonic API.

        Args:
            lang_code (TTSLangCodes | str, optional): The language code for synthesis. Defaults to "en".
            encoding (str, optional): The audio encoding format. Defaults to "pcm_linear".
            voice_id (str, optional): The voice ID for the desired voice.
            speed (float, optional): The audio playback speed. Defaults to 1.0.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 22050.
            api_key (str, optional): The NeuPhonic API key. If not provided, it will be read from the NEUPHONIC_API_KEY environment variable.
            jwt_token (str, optional): The NeuPhonic JWT token.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
            word_tokenizer (tokenize.WordTokenizer, optional): The word tokenizer to use. Defaults to tokenize.basic.WordTokenizer().
            tokenizer (tokenize.SentenceTokenizer, optional): The sentence tokenizer to use. Defaults to tokenize.blingfire.SentenceTokenizer().
            base_url (str, optional): The base URL for the NeuPhonic API. Defaults to "https://api.neuphonic.com".
        """  # noqa: E501

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        neuphonic_api_key = api_key or os.environ.get("NEUPHONIC_API_KEY")
        if not neuphonic_api_key and not jwt_token:
            raise ValueError(
                "Neuphonic API key or JWT token is required, either as argument or set"
                " NEUPHONIC_API_KEY environment variable"
            )

        if not is_given(word_tokenizer):
            word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)

        self._opts = _TTSOptions(
            lang_code=lang_code,
            encoding=encoding,
            sample_rate=sample_rate,
            voice_id=voice_id,
            speed=speed,
            api_key=neuphonic_api_key,
            jwt_token=jwt_token,
            base_url=base_url,
            word_tokenizer=word_tokenizer,
        )
        self._session = http_session
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

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        url = self._opts.get_ws_url(
            f"/speak/en?api_key={self._opts.api_key}&speed={self._opts.speed}&lang_code={self._opts.lang_code}&sampling_rate={self._opts.sample_rate}&voice_id={self._opts.voice_id}"
        )
        if self._opts.jwt_token:
            url += f"&jwt_token={self._opts.jwt_token}"

        if self._opts.api_key:
            headers = {API_AUTH_HEADER: self._opts.api_key}
        else:
            headers = {}

        return await asyncio.wait_for(session.ws_connect(url, headers=headers), timeout)

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    @property
    def model(self) -> str:
        return "Octave"

    @property
    def provider(self) -> str:
        return "Neuphonic"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def prewarm(self) -> None:
        self._pool.prewarm()

    def update_options(
        self,
        *,
        lang_code: NotGivenOr[TTSLangCodes | str] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        This allows updating the TTS settings, including lang_code, voice_id, and speed.
        If any parameter is not provided, the existing value will be retained.

        Args:
            lang_code (TTSLangCodes | str, optional): The language code for synthesis.
            voice_id (str, optional): The voice ID for the desired voice.
            speed (float, optional): The audio playback speed.
        """
        if is_given(lang_code):
            self._opts.lang_code = lang_code
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(speed):
            self._opts.speed = speed

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
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
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the SSE endpoint"""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}/sse/speak/{self._opts.lang_code}",
                headers={API_AUTH_HEADER: self._opts.api_key},
                json={
                    "text": self._input_text,
                    "voice_id": self._opts.voice_id,
                    "lang_code": self._opts.lang_code,
                    "encoding": "pcm_linear",
                    "sampling_rate": self._opts.sample_rate,
                    "speed": self._opts.speed,
                },
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
                # large read_bufsize to avoid `ValueError: Chunk too big`
                read_bufsize=10 * 1024 * 1024,
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
                )

                async for line in resp.content:
                    message = line.decode("utf-8")
                    if not message:
                        continue

                    parsed_message = _parse_sse_message(message)

                    if (
                        parsed_message is not None
                        and parsed_message.get("data", {}).get("audio") is not None
                    ):
                        audio_bytes = base64.b64decode(parsed_message["data"]["audio"])
                        output_emitter.push(audio_bytes)

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise create_api_error_from_http(e.message, status=e.status) from None
        except Exception as e:
            raise APIConnectionError() from e


def _parse_sse_message(message: str) -> dict | None:
    """
    Parse each response from the SSE endpoint.

    The message will either be a string reading:
    - `event: error`
    - `event: message`
    - `data: { "status_code": 200, "data": {"audio": ... } }`
    """
    message = message.strip()

    if not message or "data" not in message:
        return None

    _, value = message.split(": ", 1)
    message_dict: dict = json.loads(value)

    if message_dict.get("errors") is not None:
        raise Exception(
            f"received error status {message_dict['status_code']}:{message_dict['errors']}"
        )

    return message_dict


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._segments_ch = utils.aio.Chan[tokenize.SentenceStream]()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _tokenize_input() -> None:
            chunks_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if chunks_stream is None:
                        chunks_stream = self._tts._sentence_tokenizer.stream()
                        self._segments_ch.send_nowait(chunks_stream)
                    chunks_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if chunks_stream:
                        chunks_stream.end_input()
                    chunks_stream = None

            self._segments_ch.close()

        async def _run_segments() -> None:
            async for chunk_stream in self._segments_ch:
                await self._run_ws(chunk_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise create_api_error_from_http(
                e.message, status=e.status, request_id=request_id
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self, chunks_stream: tokenize.SentenceStream, output_emitter: tts.AudioEmitter
    ) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)
        chunks = 0

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for sentence in chunks_stream:
                self._mark_started()

                nonlocal chunks
                chunks += 1

                msg = {"text": f"{sentence.token}<STOP>", "context_id": segment_id}
                await ws.send_str(json.dumps(msg))

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "NeuPhonic websocket connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        resp = json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON from NeuPhonic")
                        continue

                    if resp.get("type") == "error":
                        raise APIError(f"NeuPhonic returned error: {resp}")

                    data = resp.get("data", {})
                    audio_data = data.get("audio")
                    if audio_data and audio_data != "" and data.get("context_id") == segment_id:
                        try:
                            b64data = base64.b64decode(audio_data)
                            if b64data:
                                output_emitter.push(b64data)
                        except Exception as e:
                            logger.warning("Failed to decode NeuPhonic audio data: %s", e)

                    nonlocal chunks
                    if data.get("stop"):
                        chunks -= 1

                    if data.get("context_id") != segment_id or chunks == 0:
                        output_emitter.end_segment()
                        break
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    pass
                else:
                    logger.warning("Unexpected NeuPhonic message type: %s", msg.type)

        async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
            tasks = [
                asyncio.create_task(send_task(ws)),
                asyncio.create_task(recv_task(ws)),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
