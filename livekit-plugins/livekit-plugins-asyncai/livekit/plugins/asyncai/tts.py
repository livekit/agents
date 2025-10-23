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
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Union, cast

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
from .models import (
    TTSDefaultVoiceId,
    TTSEncoding,
    TTSModels,
)

API_AUTH_HEADER = "api_key"
API_VERSION_HEADER = "version"
API_VERSION = "v1"

BUFFERED_WORDS_COUNT = 10


@dataclass
class _TTSOptions:
    model: TTSModels | str
    encoding: TTSEncoding
    sample_rate: int
    voice: str | list[float]
    api_key: str
    language: str
    base_url: str

    def get_http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_ws_url(self, path: str) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}{path}"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = "asyncflow_multilingual_v1.0",
        language: str = "en",
        encoding: TTSEncoding = "pcm_s16le",
        voice: str = TTSDefaultVoiceId,
        sample_rate: int = 32000,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = "https://api.async.ai",
    ) -> None:
        """
        Create a new instance of Async TTS.

        See https://docs.async.ai/text-to-speech-websocket-3477526w0 for more details
            on the the Async API.

        Args:
            model (TTSModels, optional): The Async TTS model to use. Defaults to "asyncflow_multilingual_v1.0".
            language (str, optional): The language code for synthesis. Defaults to "en".
            encoding (TTSEncoding, optional): The audio encoding format. Defaults to "pcm_s16le".
            voice (str, optional): The voice ID.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 32000.
            api_key (str, optional): The Async API key. If not provided, it will be
                read from the ASYNCAI_API_KEY environment variable.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp
                ClientSession to use. If not provided, a new session will be created.
            base_url (str, optional): The base URL for the Async API. Defaults to "https://api.async.ai".
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        async_api_key = api_key or os.environ.get("ASYNCAI_API_KEY")
        if not async_api_key:
            raise ValueError("ASYNCAI_API_KEY must be set")

        self._opts = _TTSOptions(
            model=model,
            language=language,
            encoding=encoding,
            sample_rate=sample_rate,
            voice=voice,
            api_key=async_api_key,
            base_url=base_url,
        )
        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        url = self._opts.get_ws_url(
            f"/text_to_speech/websocket/ws?api_key={self._opts.api_key}&version={API_VERSION}"
        )

        init_payload = {
            "model_id": self._opts.model,
            "voice": {"mode": "id", "id": self._opts.voice},
            "output_format": {
                "container": "raw",
                "encoding": self._opts.encoding,
                "sample_rate": self._opts.sample_rate,
            },
        }
        ws = await asyncio.wait_for(session.ws_connect(url), timeout)
        await ws.send_str(json.dumps(init_payload))
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def prewarm(self) -> None:
        self._pool.prewarm()

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        This method allows updating the TTS settings, including model type, language and voice.
        If any parameter is not provided, the existing value will be retained.

        Args:
            model (TTSModels, optional): The Async TTS model to use. Defaults to "asyncflow_multilingual_v1.0".
            language (str, optional): The language code for synthesis. Defaults to "en".
            voice (str, optional): The voice ID.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(voice):
            self._opts.voice = cast(Union[str, list[float]], voice)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ):
        pass

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._pool.aclose()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._sent_tokenizer_stream = tokenize.basic.SentenceTokenizer(
            min_sentence_len=BUFFERED_WORDS_COUNT
        ).stream()
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _sentence_stream_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for ev in self._sent_tokenizer_stream:
                token_pkt = {}
                token_pkt["transcript"] = ev.token + " "
                token_pkt["force"] = True
                self._mark_started()
                await ws.send_str(json.dumps(token_pkt))

            # end_pkt = {}
            # end_pkt["transcript"] = ""
            # await ws.send_str(json.dumps(end_pkt))

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._sent_tokenizer_stream.flush()
                    continue

                self._sent_tokenizer_stream.push_text(data)

            self._sent_tokenizer_stream.end_input()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            current_segment_id: str | None = None
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Async connection closed unexpectedly", request_id=request_id
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Async message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                if current_segment_id is None:
                    current_segment_id = "new_segment"
                    output_emitter.start_segment(segment_id="new_segment")
                if data.get("audio"):
                    b64data = base64.b64decode(data["audio"])
                    output_emitter.push(b64data)
                    if data.get("final") and data["final"] is True:
                        output_emitter.end_input()
                        break
                else:
                    logger.warning("unexpected message %s", data)

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                tasks = [
                    asyncio.create_task(_input_task()),
                    asyncio.create_task(_sentence_stream_task(ws)),
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
        except Exception as e:
            raise APIConnectionError() from e
