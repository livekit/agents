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
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)

from .log import logger
from .models import (
    TTSDefaultVoiceId,
    TTSEncoding,
    TTSModels,
    TTSVoiceEmotion,
    TTSVoiceSpeed,
)

API_AUTH_HEADER = "X-API-Key"
API_VERSION_HEADER = "Cartesia-Version"
API_VERSION = "2024-06-10"

NUM_CHANNELS = 1
BUFFERED_WORDS_COUNT = 3


@dataclass
class _TTSOptions:
    model: TTSModels | str
    encoding: TTSEncoding
    sample_rate: int
    voice: str | list[float]
    speed: TTSVoiceSpeed | float | None
    emotion: list[TTSVoiceEmotion | str] | None
    api_key: str
    language: str
    base_url: str

    def get_http_url(self, path: str) -> str:
        protocol = "http" if "localhost" in self.base_url else "https"
        return f"{protocol}://{self.base_url}{path}"

    def get_ws_url(self, path: str) -> str:
        protocol = "ws" if "localhost" in self.base_url else "wss"
        return f"{protocol}://{self.base_url}{path}"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = "sonic",
        language: str = "en",
        encoding: TTSEncoding = "pcm_s16le",
        voice: str | list[float] = TTSDefaultVoiceId,
        speed: TTSVoiceSpeed | float | None = None,
        emotion: list[TTSVoiceEmotion | str] | None = None,
        sample_rate: int = 24000,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = "api.cartesia.ai",
    ) -> None:
        """
        Create a new instance of Cartesia TTS.

        See https://docs.cartesia.ai/reference/web-socket/stream-speech/stream-speech for more details on the the Cartesia API.

        Args:
            model (TTSModels, optional): The Cartesia TTS model to use. Defaults to "sonic-english".
            language (str, optional): The language code for synthesis. Defaults to "en".
            encoding (TTSEncoding, optional): The audio encoding format. Defaults to "pcm_s16le".
            voice (str | list[float], optional): The voice ID or embedding array.
            speed (TTSVoiceSpeed | float, optional): Voice Control - Speed (https://docs.cartesia.ai/user-guides/voice-control)
            emotion (list[TTSVoiceEmotion], optional): Voice Control - Emotion (https://docs.cartesia.ai/user-guides/voice-control)
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            api_key (str, optional): The Cartesia API key. If not provided, it will be read from the CARTESIA_API_KEY environment variable.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
            base_url (str, optional): The base URL for the Cartesia API. Defaults to "api.cartesia.ai".
        """

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
            speed=speed,
            emotion=emotion,
            api_key=api_key,
            base_url=base_url,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def update_options(
        self,
        *,
        model: TTSModels | None = None,
        language: str | None = None,
        voice: str | list[float] | None = None,
        speed: TTSVoiceSpeed | float | None = None,
        emotion: list[TTSVoiceEmotion | str] | None = None,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        This method allows updating the TTS settings, including model type, language, voice, speed,
        and emotion. If any parameter is not provided, the existing value will be retained.

        Args:
            model (TTSModels, optional): The Cartesia TTS model to use. Defaults to "sonic-english".
            language (str, optional): The language code for synthesis. Defaults to "en".
            voice (str | list[float], optional): The voice ID or embedding array.
            speed (TTSVoiceSpeed | float, optional): Voice Control - Speed (https://docs.cartesia.ai/user-guides/voice-control)
            emotion (list[TTSVoiceEmotion], optional): Voice Control - Emotion (https://docs.cartesia.ai/user-guides/voice-control)
        """
        self._opts.model = model or self._opts.model
        self._opts.language = language or self._opts.language
        self._opts.voice = voice or self._opts.voice
        self._opts.speed = speed or self._opts.speed
        if emotion is not None:
            self._opts.emotion = emotion

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
            session=self._ensure_session(),
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "SynthesizeStream":
        return SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the bytes endpoint"""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts, self._session = opts, session

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate, num_channels=NUM_CHANNELS
        )

        json = _to_cartesia_options(self._opts)
        json["transcript"] = self._input_text

        headers = {
            API_AUTH_HEADER: self._opts.api_key,
            API_VERSION_HEADER: API_VERSION,
        }

        try:
            async with self._session.post(
                self._opts.get_http_url("/tts/bytes"),
                headers=headers,
                json=json,
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                resp.raise_for_status()
                async for data, _ in resp.content.iter_chunks():
                    for frame in bstream.write(data):
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                frame=frame,
                            )
                        )

                for frame in bstream.flush():
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(request_id=request_id, frame=frame)
                    )
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts, self._session = opts, session
        self._sent_tokenizer_stream = tokenize.basic.SentenceTokenizer(
            min_sentence_len=BUFFERED_WORDS_COUNT
        ).stream()

    async def _run(self) -> None:
        request_id = utils.shortuuid()

        async def _sentence_stream_task(ws: aiohttp.ClientWebSocketResponse):
            base_pkt = _to_cartesia_options(self._opts)
            async for ev in self._sent_tokenizer_stream:
                token_pkt = base_pkt.copy()
                token_pkt["context_id"] = request_id
                token_pkt["transcript"] = ev.token + " "
                token_pkt["continue"] = True
                self._mark_started()
                await ws.send_str(json.dumps(token_pkt))

            end_pkt = base_pkt.copy()
            end_pkt["context_id"] = request_id
            end_pkt["transcript"] = " "
            end_pkt["continue"] = False
            await ws.send_str(json.dumps(end_pkt))

        async def _input_task():
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._sent_tokenizer_stream.flush()
                    continue
                self._sent_tokenizer_stream.push_text(data)
            self._sent_tokenizer_stream.end_input()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse):
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=NUM_CHANNELS,
            )

            last_frame: rtc.AudioFrame | None = None

            def _send_last_frame(*, segment_id: str, is_final: bool) -> None:
                nonlocal last_frame
                if last_frame is not None:
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id,
                            segment_id=segment_id,
                            frame=last_frame,
                            is_final=is_final,
                        )
                    )

                    last_frame = None

            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Cartesia connection closed unexpectedly",
                        request_id=request_id,
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Cartesia message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                segment_id = data.get("context_id")

                if data.get("data"):
                    b64data = base64.b64decode(data["data"])
                    for frame in audio_bstream.write(b64data):
                        _send_last_frame(segment_id=segment_id, is_final=False)
                        last_frame = frame
                elif data.get("done"):
                    for frame in audio_bstream.flush():
                        _send_last_frame(segment_id=segment_id, is_final=False)
                        last_frame = frame

                    _send_last_frame(segment_id=segment_id, is_final=True)

                    if segment_id == request_id:
                        # we're not going to receive more frames, close the connection
                        await ws.close()
                        break
                else:
                    logger.error("unexpected Cartesia message %s", data)

        url = self._opts.get_ws_url(f"/tts/websocket?api_key={self._opts.api_key}&cartesia_version={API_VERSION}")

        ws: aiohttp.ClientWebSocketResponse | None = None

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(url), self._conn_options.timeout
            )

            tasks = [
                asyncio.create_task(_input_task()),
                asyncio.create_task(_sentence_stream_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
        finally:
            if ws is not None:
                await ws.close()


def _to_cartesia_options(opts: _TTSOptions) -> dict[str, Any]:
    voice: dict[str, Any] = {}
    if isinstance(opts.voice, str):
        voice["mode"] = "id"
        voice["id"] = opts.voice
    else:
        voice["mode"] = "embedding"
        voice["embedding"] = opts.voice

    voice_controls: dict = {}
    if opts.speed is not None:
        voice_controls["speed"] = opts.speed
    if opts.emotion is not None:
        voice_controls["emotion"] = opts.emotion

    if voice_controls:
        voice["__experimental_controls"] = voice_controls

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
