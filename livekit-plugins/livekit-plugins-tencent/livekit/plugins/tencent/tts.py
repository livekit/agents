# Copyright 2026 LiveKit, Inc.
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
import hashlib
import hmac
import json
import os
import time
import urllib.parse
import weakref
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

DEFAULT_VOICE_TYPE = 601010
DEFAULT_CODEC = "pcm"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_NUM_CHANNELS = 1
DEFAULT_HOST = "tts.cloud.tencent.com"
DEFAULT_PATH = "/stream_ws"
DEFAULT_ACTION = "TextToStreamAudioWS"


@dataclass
class _TTSOptions:
    app_id: str
    secret_id: str
    secret_key: str
    voice_type: int
    codec: str
    sample_rate: int
    speed: float
    volume: float
    model_type: int
    enable_subtitle: bool
    emotion_category: str
    emotion_intensity: int
    segment_rate: int
    fast_voice_type: str
    host: str
    path: str
    action: str
    proxy_url: str | None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        app_id: NotGivenOr[str] = NOT_GIVEN,
        secret_id: NotGivenOr[str] = NOT_GIVEN,
        secret_key: NotGivenOr[str] = NOT_GIVEN,
        voice_type: int = DEFAULT_VOICE_TYPE,
        codec: str = DEFAULT_CODEC,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        speed: float = 0.0,
        volume: float = 0.0,
        model_type: int = 0,
        enable_subtitle: bool = False,
        emotion_category: str = "",
        emotion_intensity: int = 0,
        segment_rate: int = 0,
        fast_voice_type: str = "",
        host: str = DEFAULT_HOST,
        path: str = DEFAULT_PATH,
        action: str = DEFAULT_ACTION,
        proxy_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Tencent Cloud TTS instance."""
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=DEFAULT_NUM_CHANNELS,
        )

        resolved_app_id = app_id if utils.is_given(app_id) else os.getenv("TENCENT_TTS_APP_ID", "")
        resolved_secret_id = (
            secret_id if utils.is_given(secret_id) else os.getenv("TENCENT_TTS_SECRET_ID", "")
        )
        resolved_secret_key = (
            secret_key if utils.is_given(secret_key) else os.getenv("TENCENT_TTS_SECRET_KEY", "")
        )

        if not resolved_app_id or not resolved_secret_id or not resolved_secret_key:
            raise ValueError(
                "Tencent TTS credentials are required: set TENCENT_TTS_APP_ID, "
                "TENCENT_TTS_SECRET_ID, and TENCENT_TTS_SECRET_KEY or pass them explicitly"
            )

        self._opts = _TTSOptions(
            app_id=resolved_app_id,
            secret_id=resolved_secret_id,
            secret_key=resolved_secret_key,
            voice_type=voice_type,
            codec=codec,
            sample_rate=sample_rate,
            speed=speed,
            volume=volume,
            model_type=model_type,
            enable_subtitle=enable_subtitle,
            emotion_category=emotion_category,
            emotion_intensity=emotion_intensity,
            segment_rate=segment_rate,
            fast_voice_type=fast_voice_type,
            host=host,
            path=path,
            action=action,
            proxy_url=proxy_url,
        )
        self._session = http_session
        self._streams: weakref.WeakSet[SynthesizeStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return f"voice_type:{self._opts.voice_type}"

    @property
    def provider(self) -> str:
        return "Tencent"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

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

    def _build_url(
        self,
        opts: _TTSOptions,
        *,
        text: str,
        session_id: str,
        now: int | None = None,
    ) -> str:
        timestamp = int(time.time() if now is None else now)

        sign_query = _build_query(
            opts,
            text=text,
            session_id=session_id,
            timestamp=timestamp,
            escape_text=False,
        )
        sign_url = f"{opts.host}{opts.path}?{sign_query}"
        signature = _sign_url(sign_url, opts.secret_key)

        url_query = _build_query(
            opts,
            text=text,
            session_id=session_id,
            timestamp=timestamp,
            escape_text=True,
        )
        escaped_signature = urllib.parse.quote_plus(signature)
        return f"wss://{opts.host}{opts.path}?{url_query}&Signature={escaped_signature}"

    async def _synthesize_once(
        self,
        opts: _TTSOptions,
        *,
        text: str,
        session_id: str,
        output_emitter: tts.AudioEmitter,
        conn_options: APIConnectOptions,
    ) -> None:
        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            ws = await asyncio.wait_for(
                self._ensure_session().ws_connect(
                    self._build_url(opts, text=text, session_id=session_id),
                    proxy=opts.proxy_url,
                ),
                timeout=conn_options.timeout,
            )

            handshake = await asyncio.wait_for(ws.receive(), timeout=conn_options.timeout)
            if handshake.type != aiohttp.WSMsgType.TEXT:
                raise APIConnectionError(
                    f"Unexpected Tencent TTS handshake message: {handshake.type}"
                )
            _raise_for_tencent_error(_json_loads(handshake.data), request_id=session_id)

            while True:
                msg = await asyncio.wait_for(ws.receive(), timeout=conn_options.timeout)
                if msg.type == aiohttp.WSMsgType.BINARY:
                    output_emitter.push(msg.data)
                    continue

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = _json_loads(msg.data)
                    _raise_for_tencent_error(data, request_id=session_id)
                    if data.get("final") == 1:
                        return
                    continue

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIConnectionError("Tencent TTS WebSocket closed before final message")

                if msg.type == aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError(f"Tencent TTS WebSocket error: {ws.exception()}")

        except TimeoutError as e:
            raise APITimeoutError("Timeout connecting to Tencent TTS API") from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=session_id,
                body=None,
            ) from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Tencent TTS connection error: {e}") from e
        finally:
            if ws is not None:
                await ws.close()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=DEFAULT_NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        if not self._input_text.strip():
            return

        await self._tts._synthesize_once(
            self._opts,
            text=self._input_text,
            session_id=request_id,
            output_emitter=output_emitter,
            conn_options=self._conn_options,
        )
        output_emitter.flush()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    def push_text(self, token: str) -> None:
        if not token or self._input_ch.closed:
            return

        self._pushed_text += token

        if self._metrics_task is None:
            self._metrics_task = asyncio.create_task(
                self._metrics_monitor_task(self._monitor_aiter), name="TTS._metrics_task"
            )

        if not self._mtc_text:
            self._num_segments += 1

        self._mtc_text += token
        self._input_ch.send_nowait(token)
        self._input_buffer.append(token)

    def flush(self) -> None:
        if self._input_ch.closed:
            return

        if self._mtc_text:
            self._mtc_pending_texts.append(self._mtc_text)
            self._mtc_text = ""

        sentinel = self._FlushSentinel()
        self._input_ch.send_nowait(sentinel)
        self._input_buffer.append(sentinel)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=DEFAULT_NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )

        pending_text = ""

        async def _flush_segment() -> None:
            nonlocal pending_text
            text = pending_text
            pending_text = ""
            if not text.strip():
                return

            segment_id = utils.shortuuid()
            output_emitter.start_segment(segment_id=segment_id)
            self._mark_started()
            await self._tts._synthesize_once(
                self._opts,
                text=text,
                session_id=segment_id,
                output_emitter=output_emitter,
                conn_options=self._conn_options,
            )
            output_emitter.end_segment()

        async for data in self._input_ch:
            if isinstance(data, str):
                pending_text += data
            elif isinstance(data, self._FlushSentinel):
                await _flush_segment()

        await _flush_segment()


def _build_query(
    opts: _TTSOptions,
    *,
    text: str,
    session_id: str,
    timestamp: int,
    escape_text: bool,
) -> str:
    query = {
        "Action": opts.action,
        "AppId": opts.app_id,
        "SecretId": opts.secret_id,
        "Timestamp": str(timestamp),
        "Expired": str(timestamp + 24 * 60 * 60),
        "Text": urllib.parse.quote_plus(text) if escape_text else text,
        "FastVoiceType": opts.fast_voice_type,
        "SessionId": session_id,
        "ModelType": str(opts.model_type),
        "VoiceType": str(opts.voice_type),
        "SampleRate": str(opts.sample_rate),
        "Speed": _format_float(opts.speed),
        "Volume": _format_float(opts.volume),
        "Codec": opts.codec,
        "EnableSubtitle": str(opts.enable_subtitle).lower(),
        "EmotionCategory": opts.emotion_category,
        "EmotionIntensity": str(opts.emotion_intensity),
        "SegmentRate": str(opts.segment_rate),
    }
    return "&".join(f"{key}={query[key]}" for key in sorted(query))


def _sign_url(sign_url: str, secret_key: str) -> str:
    digest = hmac.new(
        secret_key.encode("utf-8"),
        f"GET{sign_url}".encode(),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def _format_float(value: float) -> str:
    return format(value, "g")


def _json_loads(data: str) -> dict[str, Any]:
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as e:
        raise APIConnectionError(f"Invalid Tencent TTS JSON message: {data}") from e
    if not isinstance(decoded, dict):
        raise APIConnectionError(f"Unexpected Tencent TTS JSON payload: {decoded!r}")
    return decoded


def _raise_for_tencent_error(data: dict[str, Any], *, request_id: str) -> None:
    code = _as_int(data.get("code"), default=0)
    if code == 0:
        return

    raise APIStatusError(
        message=str(data.get("message") or "Tencent TTS error"),
        status_code=code,
        request_id=str(data.get("request_id") or data.get("session_id") or request_id),
        body=data,
    )


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
