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
from dataclasses import dataclass, replace

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


@dataclass
class _TTSOptions:
    language: str
    voice: str
    temperature: float
    ws_url: str | None
    max_tokens: int
    buffer_size: int


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_endpoint: str | None = None,
        ws_url: str | None = None,
        voice: str = "tara",
        language: str = "en",
        temperature: float = 0.6,
        max_tokens: int = 2000,
        buffer_size: int = 10,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Initialize the Baseten TTS.

        Args:
            api_key: Baseten API key, or ``BASETEN_API_KEY`` env var.
            model_endpoint: Baseten model HTTP endpoint for non-streaming synthesis,
                or ``BASETEN_MODEL_ENDPOINT`` env var.
            ws_url: Baseten WebSocket URL for streaming synthesis,
                or ``BASETEN_WS_URL`` env var. Example:
                ``wss://model-{MODEL_ID}.api.baseten.co/environments/production/websocket``
            voice: Speaker voice. Defaults to "tara".
            language: Language code. Defaults to "en".
            temperature: Sampling temperature. Defaults to 0.6.
            max_tokens: Maximum tokens for WebSocket streaming synthesis. Defaults to 2000.
            buffer_size: Number of words per chunk for WebSocket streaming. Defaults to 10.
            http_session: Optional aiohttp session to reuse.
        """
        api_key = api_key or os.environ.get("BASETEN_API_KEY")

        if not api_key:
            raise ValueError(
                "Baseten API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `BASETEN_API_KEY` environment variable"
            )

        model_endpoint = model_endpoint or os.environ.get("BASETEN_MODEL_ENDPOINT")
        ws_url = ws_url or os.environ.get("BASETEN_WS_URL")

        if not model_endpoint and not ws_url:
            raise ValueError(
                "At least one of model_endpoint or ws_url is required. "
                "Set model_endpoint (BASETEN_MODEL_ENDPOINT) for HTTP synthesis, "
                "or ws_url (BASETEN_WS_URL) for WebSocket streaming."
            )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=ws_url is not None),
            sample_rate=24000,
            num_channels=1,
        )

        self._api_key = api_key
        self._model_endpoint = model_endpoint

        self._opts = _TTSOptions(
            voice=voice,
            language=language,
            temperature=temperature,
            ws_url=ws_url,
            max_tokens=max_tokens,
            buffer_size=buffer_size,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return "unknown"

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
        if not self._model_endpoint:
            raise ValueError(
                "model_endpoint is required for non-streaming synthesis. "
                "Provide it via the constructor or BASETEN_MODEL_ENDPOINT env var."
            )
        return ChunkedStream(
            tts=self,
            api_key=self._api_key,
            input_text=text,
            model_endpoint=self._model_endpoint,
            conn_options=conn_options,
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        if not self._opts.ws_url:
            raise ValueError(
                "ws_url is required for streaming synthesis. "
                "Provide it via the constructor or BASETEN_WS_URL env var."
            )
        return SynthesizeStream(tts=self, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        api_key: str,
        model_endpoint: str,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            tts=tts,
            input_text=input_text,
            conn_options=conn_options,
        )

        self._tts: TTS = tts
        self._api_key = api_key
        self._model_endpoint = model_endpoint
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            async with self._tts._ensure_session().post(
                self._model_endpoint,
                headers={
                    "Authorization": f"Api-Key {self._api_key}",
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
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
    ) -> None:
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

        assert self._opts.ws_url is not None

        async def _send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    continue
                words = data.split()
                for word in words:
                    self._mark_started()
                    await ws.send_str(word)
            await ws.send_str("__END__")

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            segment_started = False
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    if not segment_started:
                        output_emitter.start_segment(segment_id=request_id)
                        segment_started = True
                    output_emitter.push(msg.data)
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError()

        try:
            session = self._tts._ensure_session()
            async with session.ws_connect(
                self._opts.ws_url,
                headers={"Authorization": f"Api-Key {self._tts._api_key}"},
                ssl=ssl_context,
            ) as ws:
                await ws.send_json(
                    {
                        "voice": self._opts.voice,
                        "max_tokens": self._opts.max_tokens,
                        "buffer_size": self._opts.buffer_size,
                    }
                )

                tasks = [
                    asyncio.create_task(_send_task(ws)),
                    asyncio.create_task(_recv_task(ws)),
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
        except (APIConnectionError, APIStatusError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError() from e
