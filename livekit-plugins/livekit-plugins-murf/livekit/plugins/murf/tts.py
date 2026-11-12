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
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    http,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import (
    TTSDefaultVoiceId,
    TTSDefaultVoiceStyle,
    TTSEncoding,
    TTSLocales,
    TTSModels,
    TTSStyles,
)

API_AUTH_HEADER = "api-key"


@dataclass
class _TTSOptions:
    api_key: str
    locale: TTSLocales | str | None
    model: TTSModels | str
    voice: str
    style: str | None
    speed: int | None
    pitch: int | None
    sample_rate: int
    encoding: TTSEncoding
    base_url: str
    min_buffer_size: int
    max_buffer_delay_in_ms: int

    def get_http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_ws_url(self, path: str) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}{path}"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = "FALCON",
        locale: TTSLocales | str | None = None,
        voice: str = TTSDefaultVoiceId,
        style: TTSStyles | str | None = None,
        speed: int | None = None,
        pitch: int | None = None,
        sample_rate: int = 24000,
        encoding: TTSEncoding = "pcm",
        base_url: str = "https://global.api.murf.ai",
        http_session: http.HttpSession | None = None,
        min_buffer_size: int = 3,
        max_buffer_delay_in_ms: int = 0,
        streaming: bool = True,
    ) -> None:
        """
        Create a new instance of Murf AI TTS.

        See https://murf.ai/api/docs/api-reference/text-to-speech/stream-input for more details on the the Murf AI API.

        Args:
            api_key (str | None, optional): The Murf AI API key. If not provided, it will be read from the MURF_API_KEY environment variable.
            model (TTSModels | str, optional): The Murf AI TTS model to use. Defaults to "FALCON".
            locale (str | None, optional): The locale for synthesis (e.g., "en-US", "en-UK"). If not provided, will be inferred from voice.
            voice (str, optional): The voice ID from Murf AI's voice library (e.g., "en-US-matthew"). Defaults to TTSDefaultVoiceId.
            style (TTSStyles | str | None, optional): The voice style to apply (e.g., "Conversation"). Can be None for default style.
            speed (int | None, optional): The speech speed control. Higher values = faster speech. None for default speed.
            pitch (int | None, optional): The speech pitch control. Higher values = higher pitch. None for default pitch.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            encoding (str, optional): The audio encoding format. Defaults to "pcm".
            http_session (http.HttpSession | None, optional): An existing HttpSession to use. If not provided, the job's shared session is used.
            base_url (str, optional): The base URL for the Murf AI API. Defaults to "https://global.api.murf.ai".
            min_buffer_size (int, optional):Minimum characters to buffer before sending text to audio when no sentence boundary is detected. Higher values improve quality; lower values reduce TTFB. Defaults to 3.
            max_buffer_delay_in_ms (int, optional): Maximum wait time before sending buffered text if min_buffer_size isn’t reached. Defaults to 0.
            streaming (bool, optional): If True, uses WebSocket streaming for real-time audio. If False, uses HTTP requests. Defaults to True.
        """  # noqa: E501

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=streaming),
            sample_rate=sample_rate,
            num_channels=1,
        )

        murf_api_key = api_key or os.environ.get("MURF_API_KEY")
        if not murf_api_key:
            raise ValueError("MURF_API_KEY must be set")

        self._opts = _TTSOptions(
            api_key=murf_api_key,
            model=model,
            locale=locale,
            voice=voice,
            style=style or TTSDefaultVoiceStyle,
            speed=speed,
            pitch=pitch,
            sample_rate=sample_rate,
            encoding=encoding,
            base_url=base_url,
            min_buffer_size=min_buffer_size,
            max_buffer_delay_in_ms=max_buffer_delay_in_ms,
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
        return "Murf"

    async def _connect_ws(self, conn_options: APIConnectOptions) -> http.WebSocket:
        session = self._ensure_session()
        url = self._opts.get_ws_url(
            f"/v1/speech/stream-input?sample_rate={self._opts.sample_rate}&format={self._opts.encoding}&model={self._opts.model}"
        )
        headers = {API_AUTH_HEADER: self._opts.api_key}
        return await session.ws_connect(url, headers=headers, conn_options=conn_options)

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
        locale: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        style: NotGivenOr[str | None] = NOT_GIVEN,
        speed: NotGivenOr[int | None] = NOT_GIVEN,
        pitch: NotGivenOr[int | None] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        This method allows updating the TTS settings, including model, locale, voice, style,
        speed and pitch. If any parameter is not provided, the existing value will be retained.

        Args:
            locale (str, optional): The locale for synthesis (e.g., "en-US", "en-UK").
            voice (str, optional): The voice ID from Murf AI's voice library. (e.g. "en-US-matthew")
            style (str | None, optional): The voice style to apply (e.g., "Conversation", "Promo").
            speed (int | None, optional): Controls the speech speed. Positive values increase speed, negative values decrease it. Valid range: -50 to 50.
            pitch (int | None, optional): Controls the speech pitch. Positive values raise pitch, negative values lower it. Valid range: -50 to 50.
        """
        if is_given(locale):
            self._opts.locale = locale
        if is_given(voice):
            self._opts.voice = voice
        if is_given(style):
            self._opts.style = style
        if is_given(speed):
            self._opts.speed = speed
        if is_given(pitch):
            self._opts.pitch = pitch

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            get_session=self._ensure_session,
            opts=replace(self._opts),
            tts=self,
            input_text=chunk.text,
            conn_options=conn_options,
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(
            pool=self._pool,
            opts=replace(self._opts),
            tts=self,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the http streaming output endpoint"""

    def __init__(
        self,
        *,
        get_session: Callable[[], http.HttpSession],
        opts: _TTSOptions,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = opts
        self._get_session = get_session

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()

        try:
            async with self._get_session().post(
                self._opts.get_http_url("/v1/speech/stream"),
                headers={API_AUTH_HEADER: self._opts.api_key},
                json={
                    "text": self._input_text,
                    "model": self._opts.model,
                    "multiNativeLocale": self._opts.locale,
                    "voice_id": self._opts.voice,
                    "style": self._opts.style,
                    "rate": self._opts.speed,
                    "pitch": self._opts.pitch,
                    "format": self._opts.encoding,
                    "sample_rate": self._opts.sample_rate,
                },
                conn_options=self._conn_options,
            ) as resp:
                await resp.raise_for_status()

                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
                )

                async for data in resp.iter_bytes():
                    output_emitter.push(data)

                output_emitter.flush()
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


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

        read_deadline = self._core_attempts

        async def _send_task(ws: http.WebSocket) -> None:
            nonlocal input_done
            context_id = utils.shortuuid()
            base_pkt = _to_murf_websocket_pkt(self._opts)
            async for data in self.input:
                if isinstance(data, self.Flush):
                    continue
                token_pkt = base_pkt.copy()
                token_pkt["context_id"] = context_id
                token_pkt["text"] = data.text + " "
                self.mark_started()
                await ws.send_str(json.dumps(token_pkt))
                read_deadline.expect_response()
                input_sent_event.set()

            input_done = True
            end_pkt = base_pkt.copy()
            end_pkt["context_id"] = context_id
            end_pkt["end"] = True
            await ws.send_str(json.dumps(end_pkt))
            read_deadline.expect_response(final=True)
            input_sent_event.set()

        async def _recv_task(ws: http.WebSocket) -> None:
            current_segment_id: str | None = None
            await input_sent_event.wait()
            while True:
                msg = await ws.receive()
                if msg.type is http.WebSocketMessageType.CLOSE:
                    raise APIStatusError(
                        "Murf AI connection closed unexpectedly", request_id=request_id
                    )

                if msg.type is not http.WebSocketMessageType.TEXT:
                    logger.warning("unexpected Murf AI message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                if current_segment_id is None:
                    current_segment_id = request_id
                    output_emitter.start_segment(segment_id=request_id)
                if data.get("audio"):
                    b64data = base64.b64decode(data["audio"])
                    read_deadline.audio_received()
                    output_emitter.push(b64data)
                elif data.get("final"):
                    if input_done:
                        # close only if the input stream is closed
                        output_emitter.end_input()
                        break
                else:
                    logger.warning("unexpected Murf message", extra={"lk.pii.data": data})

        try:
            async with self._pool.connection(conn_options=self._conn_options) as ws:
                tasks = [
                    asyncio.create_task(_send_task(ws)),
                    asyncio.create_task(_recv_task(ws)),
                ]

                try:
                    await asyncio.gather(*tasks)
                finally:
                    input_sent_event.set()
                    await utils.aio.cancel_and_wait(*tasks)
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


def _to_murf_websocket_pkt(opts: _TTSOptions) -> dict[str, Any]:
    voice_config: dict[str, Any] = {}

    if opts.voice:
        voice_config["voice_id"] = opts.voice

    if opts.style:
        voice_config["style"] = opts.style

    if opts.speed:
        voice_config["rate"] = opts.speed

    if opts.pitch:
        voice_config["pitch"] = opts.pitch

    if opts.locale:
        voice_config["multi_native_locale"] = opts.locale

    return {
        "voice_config": voice_config,
        "min_buffer_size": opts.min_buffer_size,
        "max_buffer_delay_in_ms": opts.max_buffer_delay_in_ms,
    }
