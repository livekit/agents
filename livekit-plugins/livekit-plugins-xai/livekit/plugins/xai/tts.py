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


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice: GrokVoices | str = DEFAULT_VOICE,
        language: TTSLanguages | str = "auto",
        tokenizer: tokenize.WordTokenizer | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of the xAI TTS.

        See [xAI TTS Documentation Link] for more documentation on all of these options.

        Args:
            voice (str, optional): The voice ID for the desired voice. Defaults to "ara".
            language (TTSLanguages | str, optional): Language code for synthesis (e.g., "en", "fr", "ja"). Defaults to "auto".
            api_key (str | None, optional): The xAI API key. If not provided, it will be read from the xAI environment variable.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
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
            tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)
        self._opts = _TTSOptions(
            voice=voice,
            language=language,
            tokenizer=tokenizer,
        )

        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "xAI"

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        params = {
            "voice": self._opts.voice,
            "language": self._opts.language,
            "codec": "pcm",
            "sample_rate": SAMPLE_RATE,
        }
        url = f"{XAI_WEBSOCKET_URL}?{urlencode(params)}"
        try:
            ws = await asyncio.wait_for(
                self._ensure_session().ws_connect(
                    url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                ),
                timeout,
            )
        except (
            aiohttp.ClientConnectorError,
            aiohttp.ClientConnectionResetError,
            asyncio.TimeoutError,
        ) as e:
            raise APIConnectionError("failed to connect to xAI") from e
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice: str | None = None,
        language: TTSLanguages | str | None = None,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        Args:
            voice (str, optional): The voice ID for the desired voice.
            language (TTSLanguages | str, optional): Language code for synthesis (e.g., "en", "fr", "ja").
        """  # noqa: E501
        self._opts.voice = voice or self._opts.voice
        self._opts.language = language or self._opts.language

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
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()


class SynthesizeStream(tts.SynthesizeStream):
    """Stream-based text-to-speech synthesis using xAI WebSocket API.

    This implementation connects to xAI's WebSocket API for real-time streaming
    synthesis.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

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
            async for input in self._input_ch:
                if isinstance(input, str):
                    if input_stream is None:
                        input_stream = self._opts.tokenizer.stream()
                        segments_ch.send_nowait(input_stream)
                    input_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
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
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self, input_stream: tokenize.WordStream, output_emitter: tts.AudioEmitter
    ) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)
        input_ended = False

        async def _send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal input_ended

            async for word in input_stream:
                self._mark_started()
                await ws.send_str(json.dumps({"type": "text.delta", "delta": word.token}))
            await ws.send_str(json.dumps({"type": "text.done"}))
            input_ended = True

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "xAI connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("Unexpected xAI message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                msg_type = data.get("type")
                if msg_type == "audio.delta":
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
                    logger.warning("Unexpected xAI message %s", data)

        ws = await self._tts._connect_ws(self._conn_options.timeout)
        tasks = [
            asyncio.create_task(_send_task(ws)),
            asyncio.create_task(_recv_task(ws)),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            await self._tts._close_ws(ws)
