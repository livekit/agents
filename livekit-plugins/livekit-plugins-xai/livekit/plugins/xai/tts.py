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
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .types import GrokVoices, TTSLanguages

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

XAI_WEBSOCKET_URL = "wss://api.x.ai/v1/tts"
DEFAULT_VOICE = "ara"


@dataclass
class _TTSOptions:
    voice: GrokVoices | str
    language: TTSLanguages | str
    tokenizer: tokenize.WordTokenizer
    optimize_streaming_latency: NotGivenOr[int]
    speed: NotGivenOr[float]
    text_normalization: NotGivenOr[bool]


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice: GrokVoices | str = DEFAULT_VOICE,
        language: TTSLanguages | str = "auto",
        optimize_streaming_latency: NotGivenOr[int] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        text_normalization: NotGivenOr[bool] = NOT_GIVEN,
        tokenizer: tokenize.WordTokenizer | None = None,
        http_session: http.HttpSession | None = None,
    ) -> None:
        """
        Create a new instance of the xAI TTS.

        See [xAI TTS Documentation Link] for more documentation on all of these options.

        Args:
            voice (str, optional): The voice ID for the desired voice. Defaults to "ara".
            language (TTSLanguages | str, optional): Language code for synthesis (e.g., "en", "fr", "ja"). Defaults to "auto".
            optimize_streaming_latency (int, optional): Latency optimization level for the xAI TTS websocket.
            speed (float, optional): Speaking-rate multiplier for the generated audio.
            text_normalization (bool, optional): Whether to normalize text before synthesis.
            api_key (str | None, optional): The xAI API key. If not provided, it will be read from the xAI environment variable.
            http_session (http.HttpSession | None, optional): An existing HttpSession to use. If not provided, the job's shared session is used.
        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        resolved_key: str | None = api_key if is_given(api_key) else os.environ.get("XAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "xAI API key is required, either as argument or set XAI_API_KEY"
                " environment variable"
            )
        self._api_key = resolved_key
        if tokenizer is None:
            tokenizer = tokenize.WordTokenizer(ignore_punctuation=False)
        self._opts = _TTSOptions(
            voice=voice,
            language=language,
            tokenizer=tokenizer,
            optimize_streaming_latency=optimize_streaming_latency,
            speed=speed,
            text_normalization=text_normalization,
        )

        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._pool = utils.ConnectionPool[http.WebSocket](
            connect_cb=self._connect_pooled_ws,
            close_cb=self._close_pooled_ws,
            # xAI's TTS server enforces an undocumented ~2100s deadline per websocket
            # connection; stay below it so connections rotate before the server kills them
            max_session_duration=1800,
            mark_refreshed_on_get=False,
        )

    class Markup(tts.TTS.Markup):
        # markup delegation lives in the base class, keyed on _provider_key()
        def _provider_key(self) -> str:
            return "xai"

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "xAI"

    async def _connect_ws(
        self, conn_options: APIConnectOptions, opts: _TTSOptions
    ) -> http.WebSocket:
        params: dict[str, str | int | float] = {
            "voice": opts.voice,
            "language": opts.language,
            "codec": "pcm",
            "sample_rate": SAMPLE_RATE,
        }
        if is_given(opts.optimize_streaming_latency):
            params["optimize_streaming_latency"] = opts.optimize_streaming_latency
        if is_given(opts.speed):
            params["speed"] = opts.speed
        if is_given(opts.text_normalization):
            params["text_normalization"] = str(opts.text_normalization).lower()

        url = f"{XAI_WEBSOCKET_URL}?{urlencode(params)}"
        return await self._ensure_session().ws_connect(
            url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            conn_options=conn_options,
        )

    async def _close_ws(self, ws: http.WebSocket) -> None:
        await ws.close()

    async def _connect_pooled_ws(self, conn_options: APIConnectOptions) -> http.WebSocket:
        return await self._connect_ws(conn_options, self._opts)

    async def _close_pooled_ws(self, ws: http.WebSocket) -> None:
        await self._close_ws(ws)

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice: str | None = None,
        language: TTSLanguages | str | None = None,
        optimize_streaming_latency: NotGivenOr[int] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        text_normalization: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        Args:
            voice (str, optional): The voice ID for the desired voice.
            language (TTSLanguages | str, optional): Language code for synthesis (e.g., "en", "fr", "ja").
            optimize_streaming_latency (int, optional): Latency optimization level for the xAI TTS websocket.
            speed (float, optional): Speaking-rate multiplier for the generated audio.
            text_normalization (bool, optional): Whether to normalize text before synthesis.
        """  # noqa: E501
        connection_options_before = (
            self._opts.voice,
            self._opts.language,
            self._opts.optimize_streaming_latency,
            self._opts.speed,
            self._opts.text_normalization,
        )

        self._opts.voice = voice or self._opts.voice
        self._opts.language = language or self._opts.language
        if is_given(optimize_streaming_latency):
            self._opts.optimize_streaming_latency = optimize_streaming_latency
        if is_given(speed):
            self._opts.speed = speed
        if is_given(text_normalization):
            self._opts.text_normalization = text_normalization

        connection_options_after = (
            self._opts.voice,
            self._opts.language,
            self._opts.optimize_streaming_latency,
            self._opts.speed,
            self._opts.text_normalization,
        )
        if connection_options_after != connection_options_before:
            self._pool.invalidate()

    def prewarm(self) -> None:
        self._pool.prewarm()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(
            tts=self, conn_options=conn_options, opts=replace(self._opts), pool=self._pool
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._pool.aclose()


class SynthesizeStream(tts.SynthesizeStream):
    """Stream-based text-to-speech synthesis using xAI WebSocket API.

    This implementation connects to xAI's WebSocket API for real-time streaming
    synthesis.
    """

    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
        pool: utils.ConnectionPool[http.WebSocket],
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._pool = pool

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            stream=True,
            mime_type="audio/pcm",
        )

        segments_ch = utils.aio.Chan[tokenize.WordStream]()

        async def _tokenize_input() -> None:
            input_stream = None
            async for input in self.input:
                if isinstance(input, str):
                    if input_stream is None:
                        input_stream = self._opts.tokenizer.stream()
                        segments_ch.send_nowait(input_stream)
                    input_stream.push_text(input)
                else:
                    if input_stream:
                        input_stream.end_input()
                    input_stream = None

            segments_ch.close()

        async def _run_segments() -> None:
            async for input_stream in segments_ch:
                await self._run_ws(input_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except TimeoutError:
            raise APITimeoutError() from None
        except APIConnectionError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.cancel_and_wait(*tasks)

    async def _run_ws(
        self, input_stream: tokenize.WordStream, output_emitter: tts.AudioEmitter
    ) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)
        input_ended = False

        read_deadline = self._core_attempts

        async def _send_task(ws: http.WebSocket) -> None:
            nonlocal input_ended

            async for word in input_stream:
                self.mark_started()
                await ws.send_str(json.dumps({"type": "text.delta", "delta": word.token}))
                read_deadline.expect_response()
            await ws.send_str(json.dumps({"type": "text.done"}))
            input_ended = True
            read_deadline.expect_response(final=True)

        async def _recv_task(ws: http.WebSocket) -> None:
            while True:
                msg = await ws.receive()
                if msg.type == http.WebSocketMessageType.CLOSE:
                    raise APIStatusError(
                        "xAI connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=}",
                    )

                if msg.type != http.WebSocketMessageType.TEXT:
                    logger.warning("Unexpected xAI message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                msg_type = data.get("type")
                if msg_type == "audio.delta":
                    read_deadline.audio_received()
                    output_emitter.push(base64.b64decode(data["delta"]))
                elif msg_type == "audio.done":
                    if input_ended:
                        output_emitter.end_segment()
                        break
                elif msg_type == "error":
                    raise APIStatusError(
                        data.get("message", "unknown xAI error"),
                        status_code=-1,
                        body=str(data),
                    )
                else:
                    logger.warning("Unexpected xAI message", extra={"lk.pii.data": data})

        async with self._pool.connection(conn_options=self._conn_options) as ws:
            tasks = [
                asyncio.create_task(_send_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.cancel_and_wait(*tasks)
