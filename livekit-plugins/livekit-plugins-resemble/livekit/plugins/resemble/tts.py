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
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .log import logger

RESEMBLE_WEBSOCKET_URL = "wss://websocket.cluster.resemble.ai/stream"
RESEMBLE_REST_API_URL = "https://f.cluster.resemble.ai/synthesize"
DEFAULT_VOICE_UUID = "55592656"
BUFFERED_WORDS_COUNT = 3


@dataclass
class _TTSOptions:
    voice_uuid: str
    sample_rate: int
    tokenizer: tokenize.SentenceTokenizer


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice_uuid: str | None = None,
        tokenizer: tokenize.SentenceTokenizer | None = None,
        sample_rate: int = 44100,
        http_session: aiohttp.ClientSession | None = None,
        use_streaming: bool = True,
    ) -> None:
        """
        Create a new instance of the Resemble TTS.

        See https://docs.app.resemble.ai/docs/text_to_speech/ for more documentation on all of these options.

        Args:
            voice_uuid (str, optional): The voice UUID for the desired voice. Defaults to None.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 44100.
            api_key (str | None, optional): The Resemble API key. If not provided, it will be read from the RESEMBLE_API_KEY environment variable.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
            tokenizer (tokenize.SentenceTokenizer, optional): The tokenizer to use. Defaults to tokenize.SentenceTokenizer().
            use_streaming (bool, optional): Whether to use streaming or not. Defaults to True.
        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=use_streaming),
            sample_rate=sample_rate,
            num_channels=1,
        )

        api_key = api_key or os.environ.get("RESEMBLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Resemble API key is required, either as argument or set RESEMBLE_API_KEY"
                " environment variable"
            )
        self._api_key = api_key

        if tokenizer is None:
            tokenizer = tokenize.basic.SentenceTokenizer(min_sentence_len=BUFFERED_WORDS_COUNT)

        if voice_uuid is None:
            voice_uuid = DEFAULT_VOICE_UUID

        self._opts = _TTSOptions(
            voice_uuid=voice_uuid,
            sample_rate=sample_rate,
            tokenizer=tokenizer,
        )

        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
        )

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        return await asyncio.wait_for(
            self._ensure_session().ws_connect(
                RESEMBLE_WEBSOCKET_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
            ),
            timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse):
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
        voice_uuid: str | None = None,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        Args:
            voice_uuid (str, optional): The voice UUID for the desired voice.
        """  # noqa: E501
        self._opts.voice_uuid = voice_uuid or self._opts.voice_uuid

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
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
    """Synthesize text into speech in one go using Resemble AI's REST API."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.SynthesizedAudioEmitter):
        try:
            async with self._tts._ensure_session().post(
                RESEMBLE_REST_API_URL,
                headers={
                    "Authorization": f"Bearer {self._tts._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={
                    "voice_uuid": self._opts.voice_uuid,
                    "data": self._input_text,
                    "sample_rate": self._opts.sample_rate,
                    "precision": "PCM_16",
                },
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                resp.raise_for_status()
                response_json = await resp.json()

                if not response_json.get("success", False):
                    issues = response_json.get("issues", ["Unknown error"])
                    error_msg = "; ".join(issues)
                    raise APIError(
                        message=f"Resemble API returned failure: {error_msg}",
                        body=json.dumps(response_json),
                    )

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm"
                )

                audio_b64 = response_json["audio_content"]
                audio_bytes = base64.b64decode(audio_b64)

                output_emitter.push(audio_bytes)
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
    """Stream-based text-to-speech synthesis using Resemble AI WebSocket API.


    This implementation connects to Resemble's WebSocket API for real-time streaming
    synthesis. Note that this requires a Business plan subscription with Resemble AI.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)
        self._segments_ch = utils.aio.Chan[tokenize.SentenceStream]()

    async def _run(self, output_emitter: tts.SynthesizedAudioEmitter):
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            stream=True,
        )

        async def _tokenize_input():
            """tokenize text from the input_ch to words"""
            input_stream = None
            async for text in self._input_ch:
                if isinstance(text, str):
                    if input_stream is None:
                        # new segment (after flush for e.g)
                        input_stream = self._opts.tokenizer.stream()
                        self._segments_ch.send_nowait(input_stream)
                    input_stream.push_text(text)
                elif isinstance(text, self._FlushSentinel):
                    if input_stream is not None:
                        input_stream.end_input()
                    input_stream = None

            if input_stream is not None:
                input_stream.end_input()

            self._segments_ch.close()

        async def _process_segments():
            async for input_stream in self._segments_ch:
                await self._run_ws(input_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_process_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self, input_stream: tokenize.SentenceStream, output_emitter: tts.SynthesizedAudioEmitter
    ) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        last_index = 0
        input_ended = False

        async def _send_task(ws: aiohttp.ClientWebSocketResponse):
            nonlocal input_ended, last_index
            async for data in input_stream:
                last_index += 1
                payload = {
                    "voice_uuid": self._opts.voice_uuid,
                    "data": data.token,
                    "request_id": last_index,
                    "sample_rate": self._opts.sample_rate,
                    "precision": "PCM_16",
                    "output_format": "mp3",
                }
                self._mark_started()
                await ws.send_str(json.dumps(payload))

            input_ended = True

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse):
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError("Resemble connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("Unexpected Resemble message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                if data.get("type") == "audio":
                    if data.get("audio_content", None):
                        b64data = base64.b64decode(data["audio_content"])
                        output_emitter.push(b64data)

                elif data.get("type") == "audio_end":
                    index = data["request_id"]
                    if index == last_index and input_ended:
                        output_emitter.flush()
                        break
                else:
                    logger.error("Unexpected Resemble message %s", data)

        async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
            tasks = [
                asyncio.create_task(_send_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
