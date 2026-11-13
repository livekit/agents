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
from dataclasses import dataclass, replace
from typing import cast
from urllib.parse import urlencode

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
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
        model: TTSModels | str = "async_flash_v1.0",
        language: str | None = None,
        encoding: TTSEncoding = "pcm_s16le",
        voice: str = TTSDefaultVoiceId,
        sample_rate: int = 32000,
        http_session: http.HttpSession | None = None,
        base_url: str = "https://api.async.com",
    ) -> None:
        """
        Create a new instance of Async TTS.

        See https://docs.async.com/text-to-speech-websocket-3477526w0 for more details
            on the the Async API.

        Args:
            model (TTSModels, optional): The Async TTS model to use. Defaults to "async_flash_v1.0".
            language (str, optional): The language code for synthesis.
            encoding (TTSEncoding, optional): The audio encoding format. Defaults to "pcm_s16le".
            voice (str, optional): The voice ID.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 32000.
            api_key (str, optional): The Async API key. If not provided, it will be
                read from the ASYNCAI_API_KEY environment variable.
            http_session (http.HttpSession | None, optional): An existing HttpSession to
                use. If not provided, the job's shared session is used.
            base_url (str, optional): The base URL for the Async API. Defaults to "https://api.async.com".
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
        self._pool = utils.ConnectionPool[http.WebSocket](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "AsyncAI"

    async def _connect_ws(self, conn_options: APIConnectOptions) -> http.WebSocket:
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
        ws = await session.ws_connect(url, conn_options=conn_options)
        await ws.send_str(json.dumps(init_payload))
        return ws

    async def _close_ws(self, ws: http.WebSocket) -> None:
        await ws.close()

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()

        return self._session

    def prewarm(self) -> None:
        self._pool.prewarm()

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        This method allows updating the TTS settings, including model type, language and voice.
        If any parameter is not provided, the existing value will be retained.

        Args:
            model (TTSModels, optional): The Async TTS model to use. Defaults to "async_flash_v1.0".
            language (str, optional): The language code for synthesis. Defaults to "en".
            voice (str, optional): The voice ID.
            sentence_tokenizer (tokenize.SentenceTokenizer, optional): What splits the text into the
                chunks the TTS is sent.
            pacing (tts.SentenceStreamPacer | None, optional): A ``SentenceStreamPacer``, or
                ``None`` to send each chunk as soon as it is split.
        """
        super().update_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing)
        connection_params_changed = False
        if is_given(model):
            self._opts.model = model
            connection_params_changed = True
        if is_given(language):
            self._opts.language = language
            connection_params_changed = True
        if is_given(voice):
            self._opts.voice = cast(str | list[float], voice)
            connection_params_changed = True

        if connection_params_changed:
            # model, voice and language are sent once, in the init payload _connect_ws
            # writes immediately after the handshake, so a pooled connection keeps
            # synthesising with the previous settings. mark_refreshed_on_get=True resets
            # max_session_duration on every acquire, so under steady use that connection
            # can outlive the change indefinitely.
            self._pool.invalidate()

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(
            pool=self._pool,
            opts=replace(self._opts),
            tts=self,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(chunk, conn_options=conn_options)

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._pool.aclose()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        pool: utils.ConnectionPool[http.WebSocket],
        opts: _TTSOptions,
        tts: TTS,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = opts
        self._pool = pool

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
        input_done = False

        async def _send_task(ws: http.WebSocket, asyncai_context_id: str) -> None:
            nonlocal input_done
            async for data in self.input:
                if isinstance(data, self.Flush):
                    continue

                token_pkt = {
                    "transcript": data.text + " ",
                    "context_id": asyncai_context_id,
                    "force": True,
                }
                self.mark_started()
                await ws.send_str(json.dumps(token_pkt))
                input_sent_event.set()

            input_done = True
            end_pkt = {"transcript": "", "context_id": asyncai_context_id}
            await ws.send_str(json.dumps(end_pkt))
            input_sent_event.set()

        async def _recv_task(ws: http.WebSocket, asyncai_context_id: str) -> None:
            current_segment_id: str | None = None
            await input_sent_event.wait()

            while True:
                msg = await ws.receive(timeout=self._conn_options.read_timeout)
                if msg.type is http.WebSocketMessageType.CLOSE:
                    raise APIStatusError(
                        "Async connection closed unexpectedly",
                        request_id=request_id,
                        status_code=ws.close_code or -1,
                        body=msg.data,
                    )

                if msg.type is not http.WebSocketMessageType.TEXT:
                    logger.warning("unexpected Async message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                segment_id = data.get("context_id")
                if current_segment_id is None:
                    current_segment_id = segment_id
                    output_emitter.start_segment(segment_id=segment_id)
                final = data.get("final")
                if isinstance(final, bool) and final:
                    if input_done:
                        output_emitter.end_input()
                        break
                elif (audio := data.get("audio")) is not None:
                    if not audio:
                        continue
                    try:
                        b64data = base64.b64decode(audio)
                        output_emitter.push(b64data)
                    except Exception:
                        logger.warning("invalid audio payload", extra={"lk.pii.data": data})
                        continue
                else:
                    logger.warning("unexpected message", extra={"lk.pii.data": data})

        async_context_id = str(uuid.uuid4())
        try:
            async with self._pool.connection(conn_options=self._conn_options) as ws:
                tasks = [
                    asyncio.create_task(_send_task(ws, async_context_id)),
                    asyncio.create_task(_recv_task(ws, async_context_id)),
                ]

                try:
                    await asyncio.gather(*tasks)
                finally:
                    input_sent_event.set()
                    await utils.aio.cancel_and_wait(*tasks)
        except TimeoutError:
            raise APITimeoutError() from None
        except (APIStatusError, APIConnectionError):
            raise
        except Exception as e:
            raise APIConnectionError() from e
