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
import binascii
import json
import os
import uuid
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ._utils import API_BASE_URL, parse_json_response, unwrap_data, validate_language
from .models import Language
from .scribe import _validate_ticket_url
from .version import __version__

SAMPLE_RATE = 44_100
NUM_CHANNELS = 1
DEFAULT_VOICE = "am-hamen"


@dataclass
class _StreamingOptions:
    language: Language
    voice: str
    base_url: str
    max_text_characters: int
    max_audio_seconds: int | None
    session_timeout: float


class StreamingTTS(tts.TTS):
    """Persistent Addis Voices 2 WebSocket synthesis."""

    def __init__(
        self,
        *,
        language: Language | str = "am",
        voice: str = DEFAULT_VOICE,
        api_key: str | None = None,
        base_url: str = API_BASE_URL,
        max_text_characters: int = 5000,
        max_audio_seconds: int | None = None,
        session_timeout: float = 220.0,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        if not voice:
            raise ValueError("voice must not be empty")
        if not 1 <= max_text_characters <= 5000:
            raise ValueError("max_text_characters must be between 1 and 5000")
        if max_audio_seconds is not None and not 1 <= max_audio_seconds <= 600:
            raise ValueError("max_audio_seconds must be between 1 and 600")
        if session_timeout <= 0:
            raise ValueError("session_timeout must be positive")

        resolved_api_key = api_key or os.environ.get("ADDIS_API_KEY")
        if not resolved_api_key:
            raise ValueError("AddisAI API key is required, either as api_key or ADDIS_API_KEY")

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._api_key = resolved_api_key
        self._opts = _StreamingOptions(
            language=validate_language(language),
            voice=voice,
            base_url=base_url.rstrip("/"),
            max_text_characters=max_text_characters,
            max_audio_seconds=max_audio_seconds,
            session_timeout=session_timeout,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return "addis-voices-2-realtime"

    @property
    def provider(self) -> str:
        return "AddisAI"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        language: NotGivenOr[Language | str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = validate_language(language)
        if is_given(voice):
            if not voice:
                raise ValueError("voice must not be empty")
            self._opts.voice = voice

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return SynthesizeStream(
            tts_instance=self,
            conn_options=conn_options,
            options=replace(self._opts),
        )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        raise NotImplementedError(
            "StreamingTTS supports token streaming only; use stream() or use TTS for completed clips"
        )


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts_instance: StreamingTTS,
        conn_options: APIConnectOptions,
        options: _StreamingOptions,
    ) -> None:
        super().__init__(tts=tts_instance, conn_options=conn_options)
        self._addisai_tts = tts_instance
        self._opts = options
        self._request_id = str(uuid.uuid4())

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        body: dict[str, object] = {
            "voice_id": self._opts.voice,
            "language": self._opts.language,
            "audio_format": "mp3",
            "max_text_characters": self._opts.max_text_characters,
        }
        if self._opts.max_audio_seconds is not None:
            body["max_audio_seconds"] = self._opts.max_audio_seconds

        try:
            async with self._addisai_tts._ensure_session().post(
                f"{self._opts.base_url}/api/v1/realtime/sessions",
                json=body,
                headers={
                    "x-api-key": self._addisai_tts._api_key,
                    "Accept": "application/json",
                    "X-Addis-Client": f"livekit-plugins-addisai/{__version__}",
                },
                timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
            ) as response:
                ticket = unwrap_data(await parse_json_response(response))

            websocket_url = str(ticket.get("websocket_url") or "")
            _validate_ticket_url(websocket_url, "/api/v1/realtime/voice")
            ws = await self._addisai_tts._ensure_session().ws_connect(
                websocket_url,
                timeout=aiohttp.ClientWSTimeout(
                    ws_close=self._conn_options.timeout,
                    ws_receive=self._opts.session_timeout,
                ),
            )
            try:
                await ws.send_json(
                    {"type": "session.authenticate", "token": ticket.get("token")}
                )
                initial = await _receive_event(ws)
                if initial.get("type") != "session.created":
                    _raise_event_error(initial, self._request_id)
                    raise APIConnectionError("AddisAI streaming voice session was not admitted")

                self._acquire_time = 0.0
                self._connection_reused = False
                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/mpeg",
                    stream=True,
                )

                async def send_text() -> None:
                    committed = False
                    async for item in self._input_ch:
                        if isinstance(item, self._FlushSentinel):
                            if not committed:
                                await ws.send_json(
                                    {"type": "text.commit", "request_id": self._request_id}
                                )
                                committed = True
                            continue
                        self._mark_started()
                        await ws.send_json({"type": "text.append", "text": item})
                    if not committed:
                        await ws.send_json(
                            {"type": "text.commit", "request_id": self._request_id}
                        )

                async def receive_audio() -> None:
                    while True:
                        event = await _receive_event(ws)
                        _raise_event_error(event, self._request_id)
                        if event.get("request_id") not in (None, self._request_id):
                            continue
                        event_type = event.get("type")
                        if event_type == "audio.delta":
                            if event.get("format") != "mp3":
                                raise APIConnectionError(
                                    "AddisAI returned an unsupported streaming audio format"
                                )
                            try:
                                audio = base64.b64decode(str(event.get("audio") or ""), validate=True)
                            except (ValueError, binascii.Error) as exc:
                                raise APIConnectionError(
                                    "AddisAI returned invalid base64 audio"
                                ) from exc
                            output_emitter.push(audio)
                        elif event_type in ("speech.completed", "speech.cancelled"):
                            data = event.get("data")
                            if isinstance(data, dict) and data.get("id"):
                                output_emitter._note_provider_request_id(str(data["id"]))
                            output_emitter.flush()
                            return

                await asyncio.gather(send_text(), receive_audio())
            finally:
                await ws.close()
        except asyncio.TimeoutError as exc:
            raise APITimeoutError() from exc
        except APIError:
            raise
        except aiohttp.ClientError as exc:
            raise APIConnectionError("AddisAI streaming voice connection failed") from exc


async def _receive_event(ws: aiohttp.ClientWebSocketResponse) -> dict[str, Any]:
    message = await ws.receive()
    if message.type != aiohttp.WSMsgType.TEXT:
        raise APIConnectionError("AddisAI streaming voice WebSocket closed unexpectedly")
    try:
        event = json.loads(message.data)
    except (TypeError, json.JSONDecodeError) as exc:
        raise APIConnectionError("AddisAI streaming voice returned invalid JSON") from exc
    if not isinstance(event, dict):
        raise APIConnectionError("AddisAI streaming voice returned an invalid event")
    return event


def _raise_event_error(event: dict[str, Any], request_id: str) -> None:
    if event.get("type") != "error":
        return
    raw_error = event.get("error")
    error = raw_error if isinstance(raw_error, dict) else {}
    raise APIStatusError(
        str(error.get("message") or "AddisAI streaming voice returned an error"),
        status_code=int(error.get("status") or event.get("status") or 500),
        request_id=str(event.get("request_id") or request_id),
        body=event,
    )
