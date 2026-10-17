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
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from ._utils import API_BASE_URL, parse_json_response, unwrap_data
from .version import __version__

ScribeBackend = Literal["standard", "turbo"]
ScribeChunk = Literal["320ms", "1120ms"]


@dataclass
class _ScribeOptions:
    backend: ScribeBackend
    chunk: ScribeChunk
    speakers: bool
    base_url: str
    request_timeout: float


class ScribeSTT(stt.STT):
    """Addis Scribe file and live Amharic transcription."""

    def __init__(
        self,
        *,
        backend: ScribeBackend = "standard",
        chunk: ScribeChunk = "320ms",
        speakers: bool = False,
        api_key: str | None = None,
        base_url: str = API_BASE_URL,
        request_timeout: float = 600.0,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        if backend not in ("standard", "turbo"):
            raise ValueError("backend must be 'standard' or 'turbo'")
        if chunk not in ("320ms", "1120ms"):
            raise ValueError("chunk must be '320ms' or '1120ms'")
        if speakers and backend != "turbo":
            raise ValueError("speaker labels require backend='turbo'")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")

        resolved_api_key = api_key or os.environ.get("ADDIS_API_KEY")
        if not resolved_api_key:
            raise ValueError("AddisAI API key is required, either as api_key or ADDIS_API_KEY")

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=speakers,
                aligned_transcript=False,
            )
        )
        self._api_key = resolved_api_key
        self._opts = _ScribeOptions(
            backend=backend,
            chunk=chunk,
            speakers=speakers,
            base_url=base_url.rstrip("/"),
            request_timeout=request_timeout,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return f"scribe-{self._opts.backend}"

    @property
    def provider(self) -> str:
        return "AddisAI"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    @staticmethod
    def _validate_language(language: NotGivenOr[str]) -> None:
        if is_given(language) and language != "am":
            raise ValueError("Addis Scribe currently supports only 'am' (Amharic)")

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        self._validate_language(language)
        request_id = str(uuid.uuid4())
        form = aiohttp.MultipartWriter("form-data")
        audio_part = form.append(
            rtc.combine_audio_frames(buffer).to_wav_bytes(),
            {"Content-Type": "audio/wav"},
        )
        audio_part.set_content_disposition("form-data", name="audio", filename="audio.wav")

        params: dict[str, str] = {
            "backend": self._opts.backend,
            "chunk": self._opts.chunk,
            "request_id": request_id,
            "stream": "false",
        }
        if self._opts.speakers:
            params["speakers"] = "true"

        try:
            async with self._ensure_session().post(
                f"{self._opts.base_url}/api/v1/scribe/transcribe",
                params=params,
                data=form,
                headers=self._headers,
                timeout=aiohttp.ClientTimeout(
                    total=self._opts.request_timeout,
                    sock_connect=conn_options.timeout,
                ),
            ) as response:
                data = unwrap_data(await parse_json_response(response))
            _ensure_settled(data)
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=request_id,
                alternatives=[
                    stt.SpeechData(
                        text=str(data.get("text") or ""),
                        language=LanguageCode("am"),
                        metadata={"usage": data.get("usage"), "scribe": data},
                    )
                ],
            )
        except asyncio.TimeoutError as exc:
            raise APITimeoutError() from exc
        except APIError:
            raise
        except aiohttp.ClientError as exc:
            raise APIConnectionError("failed to connect to Addis Scribe") from exc

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        self._validate_language(language)
        return SpeechStream(stt_instance=self, conn_options=conn_options)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "Accept": "application/json",
            "X-Addis-Client": f"livekit-plugins-addisai/{__version__}",
        }


class SpeechStream(stt.SpeechStream):
    def __init__(self, *, stt_instance: ScribeSTT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt_instance, conn_options=conn_options, sample_rate=16_000)
        self._scribe = stt_instance

    async def _run(self) -> None:
        iterator = self._input_ch.__aiter__()
        while True:
            try:
                first = await iterator.__anext__()
            except StopAsyncIteration:
                return
            if isinstance(first, self._FlushSentinel):
                continue
            await self._run_segment(first, iterator)

    async def _run_segment(
        self,
        first: rtc.AudioFrame,
        iterator: Any,
    ) -> None:
        request_id = str(uuid.uuid4())
        body: dict[str, object] = {
            "backend": self._scribe._opts.backend,
            "chunk": self._scribe._opts.chunk,
            "request_id": request_id,
        }
        if self._scribe._opts.speakers:
            body["speakers"] = True

        try:
            async with self._scribe._ensure_session().post(
                f"{self._scribe._opts.base_url}/api/v1/scribe/sessions",
                json=body,
                headers=self._scribe._headers,
                timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
            ) as response:
                ticket = unwrap_data(await parse_json_response(response))

            websocket_url = str(ticket.get("websocket_url") or "")
            _validate_ticket_url(websocket_url, "/api/v1/scribe/stream")
            ws = await self._scribe._ensure_session().ws_connect(
                websocket_url,
                timeout=aiohttp.ClientWSTimeout(
                    ws_close=self._conn_options.timeout,
                    ws_receive=self._scribe._opts.request_timeout,
                ),
            )
            try:
                await ws.send_json(
                    {"type": "session.authenticate", "token": ticket.get("token")}
                )
                initial = await _receive_event(ws)
                if initial.get("type") == "transcript.completed":
                    self._emit_event(initial, request_id)
                    return
                if initial.get("type") != "session.created":
                    raise APIConnectionError("Addis Scribe session was not admitted")
                self._report_connection_acquired(0.0, False)

                async def send_audio() -> None:
                    await ws.send_bytes(first.data.tobytes())
                    async for item in iterator:
                        if isinstance(item, self._FlushSentinel):
                            break
                        await ws.send_bytes(item.data.tobytes())
                    await ws.send_json({"type": "audio.finish"})

                async def receive_events() -> None:
                    while True:
                        event = await _receive_event(ws)
                        self._emit_event(event, request_id)
                        if event.get("type") == "transcript.completed":
                            return

                await asyncio.gather(send_audio(), receive_events())
            finally:
                await ws.close()
        except asyncio.TimeoutError as exc:
            raise APITimeoutError() from exc
        except APIError:
            raise
        except aiohttp.ClientError as exc:
            raise APIConnectionError("Addis Scribe WebSocket connection failed") from exc

    def _emit_event(self, event: dict[str, Any], request_id: str) -> None:
        event_type = event.get("type")
        if event_type == "error":
            raw_error = event.get("error")
            error = raw_error if isinstance(raw_error, dict) else event
            raise APIStatusError(
                str(error.get("message") or "Addis Scribe returned an error"),
                status_code=int(event.get("status") or error.get("status") or 500),
                request_id=request_id,
                body=event,
            )
        if event_type == "transcript.partial":
            text = event.get("uncommitted") or event.get("text") or ""
            if text:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        request_id=request_id,
                        alternatives=[
                            stt.SpeechData(text=str(text), language=LanguageCode("am"))
                        ],
                    )
                )
            return
        if event_type == "transcript.completed":
            data = event.get("data")
            if not isinstance(data, dict):
                raise APIConnectionError("Addis Scribe returned an invalid completion")
            _ensure_settled(data)
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=[
                        stt.SpeechData(
                            text=str(data.get("text") or ""),
                            language=LanguageCode("am"),
                            metadata={"usage": data.get("usage"), "scribe": data},
                        )
                    ],
                )
            )


def _ensure_settled(data: dict[str, Any]) -> None:
    usage = data.get("usage")
    if not isinstance(data.get("text"), str) or not isinstance(usage, dict) or usage.get("settled") is not True:
        raise APIConnectionError(
            "Addis Scribe ended without settled billing; recover the same request ID"
        )


async def _receive_event(ws: aiohttp.ClientWebSocketResponse) -> dict[str, Any]:
    message = await ws.receive()
    if message.type != aiohttp.WSMsgType.TEXT:
        raise APIConnectionError("Addis Scribe WebSocket closed unexpectedly")
    try:
        event = json.loads(message.data)
    except (TypeError, json.JSONDecodeError) as exc:
        raise APIConnectionError("Addis Scribe returned invalid JSON") from exc
    if not isinstance(event, dict):
        raise APIConnectionError("Addis Scribe returned an invalid event")
    return cast(dict[str, Any], event)


def _validate_ticket_url(url: str, expected_path: str) -> None:
    parsed = urlsplit(url)
    local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    if (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path != expected_path
        or (parsed.scheme != "wss" and not (local and parsed.scheme == "ws"))
    ):
        raise APIConnectionError("AddisAI returned an invalid WebSocket URL")
