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
from typing import Any

import aiohttp
from livekit.agents import tokenize, tts, utils

from .log import logger
from .models import TTSDefaultVoiceId, TTSEncoding, TTSModels

API_AUTH_HEADER = "X-API-Key"
API_VERSION_HEADER = "Cartesia-Version"
API_VERSION = "2024-06-10"

NUM_CHANNELS = 1
BUFFERED_WORDS_COUNT = 8


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
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
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
        return SynthesizeStream(self._opts, self._ensure_session())


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
            sample_rate=self._opts.sample_rate, num_channels=NUM_CHANNELS
        )
        request_id, segment_id = utils.shortuuid(), utils.shortuuid()

        data = _to_cartesia_options(self._opts)
        data["transcript"] = self._text

        async with self._session.post(
            "https://api.cartesia.ai/tts/bytes",
            headers={
                API_AUTH_HEADER: self._opts.api_key,
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
    def __init__(
        self,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
    ):
        super().__init__()
        self._opts, self._session = opts, session
        self._sent_tokenizer_stream = tokenize.basic.SentenceTokenizer(
            min_sentence_len=BUFFERED_WORDS_COUNT
        ).stream()

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        retry_count = 0
        max_retry = 3
        while self._input_ch.qsize() or not self._input_ch.closed:
            try:
                url = f"wss://api.cartesia.ai/tts/websocket?api_key={self._opts.api_key}&cartesia_version={API_VERSION}"
                ws = await self._session.ws_connect(url)
                retry_count = 0  # connected successfully, reset the retry_count

                await self._run_ws(ws)
            except Exception as e:
                if retry_count >= max_retry:
                    logger.exception(
                        f"failed to connect to Cartesia after {max_retry} tries"
                    )
                    break

                retry_delay = min(retry_count * 2, 10)  # max 10s
                retry_count += 1

                logger.warning(
                    f"Cartesia connection failed, retrying in {retry_delay}s",
                    exc_info=e,
                )
                await asyncio.sleep(retry_delay)

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        request_id = utils.shortuuid()

        async def sentence_stream_task():
            base_pkt = _to_cartesia_options(self._opts)
            async for ev in self._sent_tokenizer_stream:
                token_pkt = base_pkt.copy()
                token_pkt["context_id"] = request_id
                token_pkt["transcript"] = ev.token + " "
                token_pkt["continue"] = True
                await ws.send_str(json.dumps(token_pkt))

            end_pkt = base_pkt.copy()
            end_pkt["context_id"] = request_id
            end_pkt["transcript"] = " "
            end_pkt["continue"] = False
            await ws.send_str(json.dumps(end_pkt))

        async def input_task():
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._sent_tokenizer_stream.flush()
                    continue
                self._sent_tokenizer_stream.push_text(data)
            self._sent_tokenizer_stream.end_input()

        async def recv_task():
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=NUM_CHANNELS,
            )

            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise Exception("Cartesia connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Cartesia message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                segment_id = data.get("context_id")
                # Once we receive audio for a segment, we can start a new segment
                if data.get("data"):
                    b64data = base64.b64decode(data["data"])
                    for frame in audio_bstream.write(b64data):
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                segment_id=segment_id,
                                frame=frame,
                            )
                        )
                elif data.get("done"):
                    for frame in audio_bstream.flush():
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                segment_id=segment_id,
                                frame=frame,
                            )
                        )

                    if segment_id == request_id:
                        # we're not going to receive more frames, close the connection
                        await ws.close()
                        break
                else:
                    logger.error("unexpected Cartesia message %s", data)

        tasks = [
            asyncio.create_task(input_task()),
            asyncio.create_task(sentence_stream_task()),
            asyncio.create_task(recv_task()),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)


def _to_cartesia_options(opts: _TTSOptions) -> dict[str, Any]:
    voice: dict[str, Any] = {}
    if isinstance(opts.voice, str):
        voice["mode"] = "id"
        voice["id"] = opts.voice
    else:
        voice["mode"] = "embedding"
        voice["embedding"] = opts.voice

    return {
        "model_id": opts.model,
        "voice": voice,
        "output_format": {
            "container": "raw",
            "encoding": opts.encoding,
            "sample_rate": opts.sample_rate,
        },
        "language": opts.language,
    }
