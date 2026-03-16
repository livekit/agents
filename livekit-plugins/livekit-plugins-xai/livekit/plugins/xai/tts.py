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

from .log import logger
from .types import GrokVoices

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

XAI_WEBSOCKET_URL = "wss://api.x.ai/v1/realtime/audio/speech"
DEFAULT_VOICE = "ara"


@dataclass
class _TTSOptions:
    voice: GrokVoices | str
    tokenizer: tokenize.WordTokenizer


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice: GrokVoices | str = DEFAULT_VOICE,
        tokenizer: tokenize.WordTokenizer | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of the xAI TTS.

        See [xAI TTS Documentation Link] for more documentation on all of these options.

        Args:
            voice (str, optional): The voice ID for the desired voice. Defaults to "ara".
            api_key (str | None, optional): The xAI API key. If not provided, it will be read from the xAI environment variable.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError(
                "xAI API key is required, either as argument or set XAI_API_KEY"
                " environment variable"
            )
        self._api_key = api_key
        if tokenizer is None:
            tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)
        self._opts = _TTSOptions(
            voice=voice,
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
        try:
            ws = await asyncio.wait_for(
                self._create_session().ws_connect(
                    XAI_WEBSOCKET_URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                ),
                timeout,
            )
            config_msg = {
                "type": "config",
                "data": {
                    "voice_id": self._opts.voice,
                    "output_format": {"Raw": {"encoding": "Linear16"}},
                    "sample_rate_hertz": "Hz24000",
                },
            }
            await ws.send_str(json.dumps(config_msg))
        except (
            aiohttp.ClientConnectorError,
            aiohttp.ClientConnectionResetError,
            asyncio.TimeoutError,
        ) as e:
            raise APIConnectionError("failed to connect to xAI") from e
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def _create_session(self) -> aiohttp.ClientSession:
        return utils.http_context.http_session()

    def update_options(
        self,
        *,
        voice: str | None = None,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        Args:
            voice_uuid (str, optional): The voice ID for the desired voice.
        """  # noqa: E501
        self._opts.voice = voice or self._opts.voice

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None: ...

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
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            stream=True,
            mime_type="audio/pcm",
        )

        async def _tokenize_input() -> None:
            input_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if input_stream is None:
                        input_stream = self._opts.tokenizer.stream()
                        self._segments_ch.send_nowait(input_stream)
                    input_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if input_stream:
                        input_stream.end_input()
                    input_stream = None

            self._segments_ch.close()

        async def _run_segments() -> None:
            async for input_stream in self._segments_ch:
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
                text_chunk_msg = {
                    "type": "text_chunk",
                    "data": {"text": f"{word.token}", "is_last": False},
                }
                self._mark_started()
                await ws.send_str(json.dumps(text_chunk_msg))
            last_msg = {"type": "text_chunk", "data": {"text": "", "is_last": True}}
            await ws.send_str(json.dumps(last_msg))
            input_ended = True

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError("xAI connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("Unexpected xAI message type %s", msg.type)
                    continue

                msg = json.loads(msg.data)
                if msg["data"].get("type") == "audio":
                    if msg["data"]["data"].get("audio", None):
                        b64data = base64.b64decode(msg["data"]["data"]["audio"])
                        output_emitter.push(b64data)

                    if msg["data"]["data"].get("is_last") and input_ended:
                        output_emitter.end_segment()
                        break

                else:
                    logger.error("Unexpected xAI message %s", msg)

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
