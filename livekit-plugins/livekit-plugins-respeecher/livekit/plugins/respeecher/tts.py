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
from dataclasses import dataclass

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import TTSEncoding, TTSModels, Voice, VoiceSettings
from .version import __version__

API_VERSION = __version__
API_AUTH_HEADER = "X-API-Key"
API_VERSION_HEADER = "LiveKit-Plugin-Respeecher-Version"
API_BASE_URL = "https://api.respeecher.com/v1"


@dataclass
class _TTSOptions:
    model: TTSModels | str
    encoding: TTSEncoding
    sample_rate: int
    voice_id: str
    voice_settings: NotGivenOr[VoiceSettings]
    api_key: str
    base_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: TTSModels | str = "/public/tts/en-rt",
        encoding: TTSEncoding = "pcm_s16le",
        voice_id: str = "samantha",
        voice_settings: NotGivenOr[VoiceSettings] = NOT_GIVEN,
        sample_rate: int = 24000,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = API_BASE_URL,
    ) -> None:
        """
        Create a new instance of Respeecher TTS.

        Args:
            api_key: Respeecher API key. If not provided, uses RESPEECHER_API_KEY env variable.
            model: The Respeecher TTS model to use.
            encoding: Audio encoding format.
            voice_id: ID of the voice to use. Different set of voices is available for different models. Thus, update the value after getting list_voices() API.
            voice_settings: Optional voice settings including sampling parameters.
            sample_rate: Audio sample rate in Hz.
            http_session: Optional aiohttp session to use for requests.
            base_url: The base URL for the Respeecher API.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=False,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        respeecher_api_key = api_key if is_given(api_key) else os.environ.get("RESPEECHER_API_KEY")
        if not respeecher_api_key:
            raise ValueError("RESPEECHER_API_KEY must be set")

        self._opts = _TTSOptions(
            model=model,
            encoding=encoding,
            sample_rate=sample_rate,
            voice_id=voice_id,
            voice_settings=voice_settings,
            api_key=respeecher_api_key,
            base_url=base_url,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
        )

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        # WebSocket protocol does not support custom headers, using query parameter
        ws_url = self._opts.base_url.replace("https://", "wss://").replace("http://", "ws://")
        if not ws_url.startswith("wss://"):
            logger.error("Insecure WebSocket connection detected, wss:// required")
            raise APIConnectionError("Secure WebSocket connection (wss://) required")

        full_ws_url = f"{ws_url}{self._opts.model}/tts/websocket?api_key={self._opts.api_key}&source={API_VERSION_HEADER}&version={API_VERSION}"
        return await asyncio.wait_for(session.ws_connect(full_ws_url), timeout)

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def list_voices(self) -> list[Voice]:
        """List available voices from Respeecher API"""
        async with self._ensure_session().get(
            f"{self._opts.base_url}{self._opts.model}/voices",
            headers={
                API_AUTH_HEADER: self._opts.api_key,
                API_VERSION_HEADER: API_VERSION,
            },
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            voices = []
            for voice_data in data:
                voices.append(Voice(voice_data))

            if len(voices) == 0:
                raise APIError("No voices are available")

            return voices

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        voice_settings: NotGivenOr[VoiceSettings] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
    ) -> None:
        """Update TTS options"""
        if is_given(model) and model != self._opts.model:
            self._opts.model = model
            # Clear the connection pool when model changes to force reconnection
            asyncio.create_task(self._pool.aclose())
            self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
                connect_cb=self._connect_ws,
                close_cb=self._close_ws,
            )

        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(voice_settings):
            self._opts.voice_settings = voice_settings

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def prewarm(self) -> None:
        self._pool.prewarm()

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

        if self._session:
            await self._session.close()
            self._session = None


class ChunkedStream(tts.ChunkedStream):
    """Synthesize text using Respeecher HTTPS endpoint"""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the TTS synthesis"""
        json_data = {
            "transcript": self._input_text,
            "voice": {
                "id": self._tts._opts.voice_id,
            },
            "output_format": {
                "sample_rate": self._tts._opts.sample_rate,
                "encoding": self._tts._opts.encoding,
            },
        }

        if (
            is_given(self._tts._opts.voice_settings)
            and self._tts._opts.voice_settings.sampling_params
        ):
            json_data["voice"]["sampling_params"] = self._tts._opts.voice_settings.sampling_params  # type: ignore[index]

        http_url = f"{self._tts._opts.base_url}{self._tts._opts.model}/tts/bytes"

        try:
            async with self._tts._ensure_session().post(
                http_url,
                headers={
                    API_AUTH_HEADER: self._tts._opts.api_key,
                    API_VERSION_HEADER: API_VERSION,
                    "Content-Type": "application/json",
                },
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._tts._opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/wav",
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
    """Streamed API using WebSocket for real-time synthesis"""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)

    async def aclose(self) -> None:
        await super().aclose()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        context_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=context_id,
            sample_rate=self._tts._opts.sample_rate,  # type: ignore[attr-defined]
            num_channels=1,
            stream=True,
            mime_type="audio/pcm",
        )
        output_emitter.start_segment(segment_id=context_id)

        sent_tokenizer_stream = self._tts._sentence_tokenizer.stream()  # type: ignore[attr-defined]

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_tokenizer_stream.flush()
                    continue
                sent_tokenizer_stream.push_text(data)
            sent_tokenizer_stream.end_input()

        async def _sentence_stream_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for sent in sent_tokenizer_stream:
                generate_request = {
                    "context_id": context_id,
                    "transcript": sent.token,
                    "voice": {
                        "id": self._tts._opts.voice_id,  # type: ignore[attr-defined]
                    },
                    "continue": True,
                    "output_format": {
                        "encoding": self._tts._opts.encoding,  # type: ignore[attr-defined]
                        "sample_rate": self._tts._opts.sample_rate,  # type: ignore[attr-defined]
                    },
                }
                if (
                    is_given(self._tts._opts.voice_settings)  # type: ignore[attr-defined]
                    and self._tts._opts.voice_settings.sampling_params  # type: ignore[attr-defined]
                ):
                    generate_request["voice"]["sampling_params"] = (
                        self._tts._opts.voice_settings.sampling_params  # type: ignore[attr-defined]
                    )

                self._mark_started()
                await ws.send_str(json.dumps(generate_request))

            # Send final message with continue=False
            end_request = {
                "context_id": context_id,
                "transcript": "",
                "voice": {
                    "id": self._tts._opts.voice_id,  # type: ignore[attr-defined]
                },
                "continue": False,
                "output_format": {
                    "encoding": self._tts._opts.encoding,  # type: ignore[attr-defined]
                    "sample_rate": self._tts._opts.sample_rate,  # type: ignore[attr-defined]
                },
            }
            if (
                is_given(self._tts._opts.voice_settings)  # type: ignore[attr-defined]
                and self._tts._opts.voice_settings.sampling_params  # type: ignore[attr-defined]
            ):
                end_request["voice"]["sampling_params"] = (  # type: ignore[index]
                    self._tts._opts.voice_settings.sampling_params  # type: ignore[attr-defined]
                )
            await ws.send_str(json.dumps(end_request))

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Respeecher connection closed unexpectedly", request_id=context_id
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("Unexpected Respeecher message type %s", msg.type)
                    continue

                data = json.loads(msg.data)

                if data.get("context_id") != context_id:
                    logger.warning(
                        "Received a message with context_id=%s instead of expected %s",
                        data.get("context_id"),
                        context_id,
                    )
                    continue

                if data.get("type") == "error":
                    raise APIError(f"Respeecher returned error: {data.get('error')}")

                if data.get("type") == "chunk":
                    audio_data = base64.b64decode(data["data"])
                    output_emitter.push(audio_data)

                elif data.get("type") == "done":
                    if sent_tokenizer_stream.closed:
                        output_emitter.end_input()
                        break

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:  # type: ignore[attr-defined]
                tasks = [
                    asyncio.create_task(_input_task()),
                    asyncio.create_task(_sentence_stream_task(ws)),
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
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            output_emitter.end_segment()
