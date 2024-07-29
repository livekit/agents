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
from dataclasses import dataclass

import aiohttp
from livekit import rtc
from livekit.agents import tokenize, tts, utils

from .log import logger
from .models import TTSDefaultVoiceId, TTSEncoding, TTSModels

API_AUTH_HEADER = "X-API-Key"
API_VERSION_HEADER = "Cartesia-Version"
API_VERSION = "2024-06-10"


@dataclass
class _TTSOptions:
    model: TTSModels
    encoding: TTSEncoding
    sample_rate: int
    voice: str | list[float]
    api_key: str
    language: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels = "sonic-english",
        language: str = "en",
        encoding: TTSEncoding = "pcm_s16le",
        voice: str | list[float] = TTSDefaultVoiceId,
        sample_rate: int = 24000,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        api_key = api_key or os.environ.get("CARTESIA_API_KEY")
        if not api_key:
            raise ValueError("CARTESIA_API_KEY must be set")

        self._opts = _TTSOptions(
            model=model,
            language=language,
            encoding=encoding,
            sample_rate=sample_rate,
            voice=voice,
            api_key=api_key,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def synthesize(self, text: str) -> "ChunkedStream":
        return ChunkedStream(text, self._opts, self._ensure_session())
    
    def stream(self) -> "SynthesizeStream":
        return SynthesizeStream(self._ensure_session(), self._opts)


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the bytes endpoint"""
    def __init__(
        self, text: str, opts: _TTSOptions, session: aiohttp.ClientSession
    ) -> None:
        super().__init__()
        self._text, self._opts, self._session = text, opts, session

    @utils.log_exceptions(logger=logger)
    async def _main_task(self):
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate, num_channels=1
        )
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()

        voice = {}
        if isinstance(self._opts.voice, str):
            voice["mode"] = "id"
            voice["id"] = self._opts.voice
        else:
            voice["mode"] = "embedding"
            voice["embedding"] = self._opts.voice

        data = {
            "model_id": self._opts.model,
            "transcript": self._text,
            "voice": voice,
            "output_format": {
                "container": "raw",
                "encoding": self._opts.encoding,
                "sample_rate": self._opts.sample_rate,
            },
            "language": self._opts.language,
        }

        async with self._session.post(
            "https://api.cartesia.ai/tts/bytes",
            headers={
                API_AUTH_HEADER: f"{self._opts.api_key}",
                API_VERSION_HEADER: API_VERSION,
            },
            json=data,
        ) as resp:
            async for data, _ in resp.content.iter_chunks():
                for frame in bstream.write(data):
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id, segment_id=segment_id, frame=frame
                        )
                    )

            for frame in bstream.flush():
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=request_id, segment_id=segment_id, frame=frame
                    )
                )

class SynthesizeStream(tts.SynthesizeStream):
    """Streamed API using websockets"""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        opts: _TTSOptions,
    ):
        super().__init__()
        self._opts, self._session = opts, session

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()

        @utils.log_exceptions(logger=logger)
        async def _tokenize_input():
            """tokenize text from the input_ch to words"""
            word_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if word_stream is None:
                        # new segment (after flush for e.g)
                        word_stream = self._opts.word_tokenizer.stream()
                        self._segments_ch.send_nowait(word_stream)

                    word_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if word_stream is not None:
                        word_stream.end_input()

                    word_stream = None

            self._segments_ch.close()

        @utils.log_exceptions(logger=logger)
        async def _run():
            async for word_stream in self._segments_ch:
                await self._run_ws(word_stream)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self,
        word_stream: tokenize.WordStream,
        max_retry: int = 3,
    ) -> None:
        ws_conn: aiohttp.ClientWebSocketResponse | None = None
        for try_i in range(max_retry):
            retry_delay = 5
            try:
                if try_i > 0:
                    await asyncio.sleep(retry_delay)

                ws_conn = await self._session.ws_connect(
                    f"wss://api.cartesia.ai/tts/websocket?api_key={self._opts.api_key}&cartesia_version={API_VERSION}",
                )
                break
            except Exception as e:
                logger.warning(
                    f"failed to connect to Cartesia, retrying in {retry_delay}s",
                    exc_info=e,
                )

        if ws_conn is None:
            raise Exception(f"failed to connect to Cartesia after {max_retry} retries")

        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()
        
        eos_sent = False
        
        context_id = utils.shortuuid()
        
        voice = {}
        if isinstance(self._opts.voice, str):
            voice["mode"] = "id"
            voice["id"] = self._opts.voice
        else:
            voice["mode"] = "embedding"
            voice["embedding"] = self._opts.voice
        
        data_pkt = {
            "model_id": self._opts.model,
            "voice": voice,
            "output_format": {
                "container": "raw",
                "encoding": self._opts.encoding,
                "sample_rate": self._opts.sample_rate,
            },
            "language": self._opts.language,
            "context_id": context_id,
        }

        async def send_task():
            nonlocal eos_sent

            async for data in word_stream:
                data_pkt["transcript"] = f"{data.token} "
                data_pkt["continue"] = True
                await ws_conn.send_str(json.dumps(data_pkt))

            # no more token, mark eos
            eos_pkt = data_pkt.copy()
            eos_pkt["transcript"] = ""
            eos_pkt["continue"] = False
            await ws_conn.send_str(json.dumps(eos_pkt))
            eos_sent = True

        async def recv_task():
            nonlocal eos_sent

            while True:
                msg = await ws_conn.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if not eos_sent:
                        raise Exception(
                            "Cartesia connection closed unexpectedly, not all tokens have been consumed"
                        )
                    return

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Cartesia message type %s", msg.type)
                    continue

                self._process_stream_event(
                    data=json.loads(msg.data),
                    request_id=request_id,
                    segment_id=segment_id,
                    context_id=context_id,
                )

        await asyncio.gather(send_task(), recv_task())

    def _process_stream_event(
        self, *, data: dict, request_id: str, segment_id: str, context_id: str
    ) -> None:
        if data.get("context_id") != context_id: # ignore messages for other contexts, this should not happen
            logger.warning("received message for unknown context %s", data)
            return
        if data.get("audio"):
            b64data = base64.b64decode(data["audio"])
            chunk_frame = rtc.AudioFrame(
                data=b64data,
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=len(b64data) // 2,
            )
            self._event_ch.send_nowait(
                tts.SynthesizedAudio(
                    request_id=request_id,
                    segment_id=segment_id,
                    frame=chunk_frame,
                )
            )
        elif data.get("error"):
            logger.error("Cartesia reported an error: %s", data["error"])
        elif not data.get("done"):
            logger.error("unexpected Cartesia message %s", data)
