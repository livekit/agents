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
import uuid
import weakref
from collections import deque
from dataclasses import dataclass, replace
from typing import cast
from urllib.parse import urlencode

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

from .constants import (
    API_AUTH_HEADER,
    API_VERSION,
    API_VERSION_HEADER,
)
from .log import logger
from .models import (
    TTSDefaultVoiceId,
    TTSEncoding,
    TTSModels,
)


@dataclass
class _TTSOptions:
    model: TTSModels | str
    encoding: TTSEncoding
    sample_rate: int
    voice: str | list[float]
    api_key: str
    language: str | None
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
        language: str | None = None,
        encoding: TTSEncoding = "pcm_s16le",
        voice: str = TTSDefaultVoiceId,
        sample_rate: int = 32000,
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        base_url: str = "https://api.async.ai",
    ) -> None:
        """
        Create a new instance of Async TTS.

        See https://docs.async.ai/text-to-speech-websocket-3477526w0 for more details
            on the the Async API.

        Args:
            model (TTSModels, optional): The Async TTS model to use. Defaults to "asyncflow_multilingual_v1.0".
            language (str, optional): The language code for synthesis.
            encoding (TTSEncoding, optional): The audio encoding format. Defaults to "pcm_s16le".
            voice (str, optional): The voice ID.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 32000.
            api_key (str, optional): The Async API key. If not provided, it will be
                read from the ASYNCAI_API_KEY environment variable.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp
                ClientSession to use. If not provided, a new session will be created.
            tokenizer (tokenize.SentenceTokenizer, optional): The tokenizer to use. Defaults to `livekit.agents.tokenize.blingfire.SentenceTokenizer`.
            base_url (str, optional): The base URL for the Async API. Defaults to "https://api.async.ai".
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        async_api_key = api_key or os.environ.get("ASYNCAI_API_KEY")
        if not async_api_key:
            raise ValueError(
                "AsyncAI API key is required, either as argument or set"
                " ASYNCAI_API_KEY environment variable"
            )

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

        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "AsyncAI"

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        query = urlencode({API_AUTH_HEADER: self._opts.api_key, API_VERSION_HEADER: API_VERSION})
        url = self._opts.get_ws_url(f"/text_to_speech/websocket/ws?{query}")

        init_payload = {
            "model_id": self._opts.model,
            "voice": {"mode": "id", "id": self._opts.voice},
            "output_format": {
                "container": "raw",
                "encoding": self._opts.encoding,
                "sample_rate": self._opts.sample_rate,
            },
        }

        if self._opts.language is not None:
            init_payload["language"] = self._opts.language
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
            self._opts.voice = cast(str | list[float], voice)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        raise NotImplementedError("AsyncAI TTS supports streaming only; use tts.stream().")

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._pool.aclose()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
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

        input_sent_event = asyncio.Event()
        sent_tokens = deque[str]()

        sent_tokenizer_stream = self._tts._sentence_tokenizer.stream()

        async def _sentence_stream_task(
            ws: aiohttp.ClientWebSocketResponse, asyncai_context_id: str
        ) -> None:
            async for ev in sent_tokenizer_stream:
                token_pkt: dict[str, object] = {}
                token_pkt["transcript"] = ev.token + " "
                token_pkt["context_id"] = asyncai_context_id
                token_pkt["force"] = True
                sent_tokens.append(ev.token + " ")
                self._mark_started()
                await ws.send_str(json.dumps(token_pkt))
                input_sent_event.set()

            end_pkt = {}
            end_pkt["transcript"] = ""
            end_pkt["context_id"] = asyncai_context_id
            await ws.send_str(json.dumps(end_pkt))
            input_sent_event.set()

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_tokenizer_stream.flush()
                    continue

                sent_tokenizer_stream.push_text(data)

            sent_tokenizer_stream.end_input()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse, asyncai_context_id: str) -> None:
            current_segment_id: str | None = None
            await input_sent_event.wait()

            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Async connection closed unexpectedly",
                        request_id=request_id,
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Async message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                segment_id = data.get("context_id")
                if current_segment_id is None:
                    current_segment_id = segment_id
                    output_emitter.start_segment(segment_id=segment_id)
                final = data.get("final")
                if isinstance(final, bool) and final:
                    if sent_tokenizer_stream.closed:
                        output_emitter.end_input()
                        break
                elif (audio := data.get("audio")) is not None:
                    if not audio:
                        continue
                    try:
                        b64data = base64.b64decode(audio)
                        output_emitter.push(b64data)
                    except Exception:
                        logger.warning("invalid audio payload %s", data)
                        continue
                else:
                    logger.warning("unexpected message %s", data)

        async_context_id = str(uuid.uuid4())
        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                tasks = [
                    asyncio.create_task(_input_task()),
                    asyncio.create_task(_sentence_stream_task(ws, async_context_id)),
                    asyncio.create_task(_recv_task(ws, async_context_id)),
                ]

                try:
                    await asyncio.gather(*tasks)
                finally:
                    input_sent_event.set()
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
