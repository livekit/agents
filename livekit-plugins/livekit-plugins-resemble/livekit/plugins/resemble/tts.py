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
from dataclasses import dataclass

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

RESEMBLE_WEBSOCKET_URL = "wss://websocket.cluster.resemble.ai/stream"
RESEMBLE_REST_API_URL = "https://f.cluster.resemble.ai/synthesize"
NUM_CHANNELS = 1
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
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice_uuid: NotGivenOr[str] = NOT_GIVEN,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        sample_rate: int = 44100,
        http_session: aiohttp.ClientSession | None = None,
        use_streaming: bool = True,
    ) -> None:
        """
        Create a new instance of the Neuphonic TTS.

        See https://docs.app.resemble.ai/docs/text_to_speech/ for more documentation on all of these options, or go to https://app.resemble.ai/ to test out different options.

        Args:
            voice_uuid (str, optional): The voice UUID for the desired voice. Defaults to None.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 44100.
            api_key (str | None, optional): The Neuphonic API key. If not provided, it will be read from the NEUPHONIC_API_TOKEN environment variable.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
            tokenizer (tokenize.SentenceTokenizer, optional): The tokenizer to use. Defaults to tokenize.SentenceTokenizer().
            use_streaming (bool, optional): Whether to use streaming or not. Defaults to True.
        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=use_streaming),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        self._api_key = api_key if is_given(api_key) else os.environ.get("RESEMBLE_API_KEY")
        if not self._api_key:
            raise ValueError("API key must be provided or set in RESEMBLE_API_KEY")

        self._opts = _TTSOptions(
            voice_uuid=voice_uuid if is_given(voice_uuid) else DEFAULT_VOICE_UUID,
            sample_rate=sample_rate,
            tokenizer=tokenizer
            if is_given(tokenizer)
            else tokenize.basic.SentenceTokenizer(min_sentence_len=BUFFERED_WORDS_COUNT),
        )

        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
        )

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()

        return await asyncio.wait_for(
            session.ws_connect(
                RESEMBLE_WEBSOCKET_URL, headers={"Authorization": f"Bearer {self._api_key}"}
            ),
            self._conn_options.timeout,
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
        voice_uuid: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        Args:
            voice_uuid (str, optional): The voice UUID for the desired voice.
            sample_rate (int, optional): The audio sample rate in Hz.
        """  # noqa: E501
        if is_given(voice_uuid):
            self._opts.voice_uuid = voice_uuid
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            api_key=self._api_key,
            session=self._ensure_session(),
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(
            tts=self,
            pool=self._pool,
            opts=self._opts,
            api_key=self._api_key,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize text into speech in one go using Resemble AI's REST API."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts, self._session, self._api_key = opts, session, api_key

    async def _run(self) -> None:
        request_id = utils.shortuuid()

        # Create request headers
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",  # Expect JSON response
        }

        # Create request payload
        payload = {
            "voice_uuid": self._opts.voice_uuid,
            "data": self._input_text,
            "sample_rate": self._opts.sample_rate,
        }

        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
        )

        try:
            async with self._session.post(
                RESEMBLE_REST_API_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as response:
                response.raise_for_status()
                response_json = await response.json()

                # Check for success
                if not response_json.get("success", False):
                    issues = response_json.get("issues", ["Unknown error"])
                    error_msg = "; ".join(issues)
                    raise APIStatusError(
                        message=f"Resemble API returned failure: {error_msg}",
                        status_code=response.status,
                        request_id=request_id,
                        body=json.dumps(response_json),
                    )

                # Extract base64-encoded audio content
                audio_content_b64 = response_json.get("audio_content")
                if not audio_content_b64:
                    raise APIStatusError(
                        message="No audio content in response",
                        status_code=response.status,
                        request_id=request_id,
                        body=json.dumps(response_json),
                    )

                # Decode base64 to get raw audio bytes
                audio_bytes = base64.b64decode(audio_content_b64)

                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                )
                for frame in audio_bstream.write(audio_bytes):
                    emitter.push(frame)
                for frame in audio_bstream.flush():
                    emitter.push(frame)
                emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=f"resemble api error: {str(e)}",
            ) from e
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streamed API using websockets"""

    def __init__(
        self,
        *,
        tts: TTS,
        opts: _TTSOptions,
        pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse],
        api_key: str,
    ):
        super().__init__(tts=tts)
        self._opts, self._pool, self._api_key = opts, pool, api_key

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()

        @utils.log_exceptions(logger=logger)
        async def _tokenize_input():
            """tokenize text from the input_ch to words"""
            input_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if input_stream is None:
                        # new segment (after flush for e.g)
                        input_stream = self._opts.tokenizer.stream()
                        self._segments_ch.send_nowait(input_stream)
                    input_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if input_stream is not None:
                        input_stream.end_input()
                    input_stream = None
            if input_stream is not None:
                input_stream.end_input()
            self._segments_ch.close()

        @utils.log_exceptions(logger=logger)
        async def _process_segments():
            async for input_stream in self._segments_ch:
                await self._run_ws(input_stream, request_id)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_process_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self,
        input_stream: tokenize.WordStream,
        request_id: str,
    ) -> None:
        async with self._pool.connection() as ws:
            segment_id = utils.shortuuid()

            @utils.log_exceptions(logger=logger)
            async def _send_task(ws: aiohttp.ClientWebSocketResponse):
                """Stream text to the websocket."""
                async for data in input_stream:
                    payload = {
                        "voice_uuid": self._opts.voice_uuid,
                        "data": data.text,
                        "request_id": request_id,
                        "sample_rate": self._opts.sample_rate,
                        "precision": "PCM_16",
                        "no_audio_header": True,
                    }
                    await ws.send_str(json.dumps(payload))

            @utils.log_exceptions(logger=logger)
            async def _recv_task(ws: aiohttp.ClientWebSocketResponse):
                audio_bstream = utils.audio.AudioByteStream(
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                )
                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                    segment_id=segment_id,
                )

                while True:
                    msg = await ws.receive()
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        raise APIStatusError(
                            "Resemble connection closed unexpectedly",
                            request_id=request_id,
                        )

                    if msg.type != aiohttp.WSMsgType.TEXT:
                        logger.warning("Unexpected Resemble message type %s", msg.type)
                        continue

                    data = json.loads(msg.data)

                    if data.get("type") == "audio":
                        b64data = base64.b64decode(data["audio_content"])
                        for frame in audio_bstream.write(b64data):
                            emitter.push(frame)

                    elif data.get("type") == "audio_end":
                        for frame in audio_bstream.flush():
                            emitter.push(frame)
                        emitter.flush()
                        break  # we are not going to receive any more audio
                    else:
                        logger.error("Unexpected Resemble message %s", data)

            tasks = [
                asyncio.create_task(_send_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]

            try:
                await asyncio.gather(*tasks)
            except asyncio.TimeoutError as e:
                raise APITimeoutError() from e
            except aiohttp.ClientResponseError as e:
                raise APIStatusError(
                    message=e.message,
                    status_code=e.status,
                    request_id=request_id,
                    body=None,
                ) from e
            except Exception as e:
                raise APIConnectionError() from e
            finally:
                await utils.aio.gracefully_cancel(*tasks)
