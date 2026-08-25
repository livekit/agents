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

"""Floe speech-to-text for LiveKit Agents.

Unlike the LLM and TTS legs (OpenAI-compatible base-URL swaps), Floe streaming
STT is a dedicated plugin: it opens a WebSocket to Floe, streams raw PCM audio
up, and receives JSON transcript messages back — keyless, with Floe fronting the
upstream (Deepgram) key and metering per audio-second on your Floe balance.

Wire protocol:
    Connect: ``wss://credit-api.floelabs.xyz/v1/audio/transcriptions/stream``
        with query params ``model``, ``encoding``, ``sample_rate``, ``language``.
    Auth: ``Authorization: Bearer <FLOE_API_KEY>`` (agent key; ``floe_live_``
        developer keys are rejected).
    Client -> server: raw binary PCM frames in the declared encoding.
    Server -> client: JSON ``{"type":"transcript","text","is_final","speech_final"}``
        and ``{"type":"error","code","message"}`` (followed by a socket close).

A non-streaming ``recognize()`` fallback is also provided over the batch
``POST /v1/audio/transcriptions`` endpoint (OpenAI-shaped multipart).
"""

from __future__ import annotations

import asyncio
import json
import os
from urllib.parse import urlencode, urlparse

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger
from .models import STTModels

# Streaming-STT WebSocket (keyless): raw PCM up, JSON transcripts down.
FLOE_STT_WS_URL = "wss://credit-api.floelabs.xyz/v1/audio/transcriptions/stream"

# Batch STT (OpenAI-shaped multipart) for the non-streaming recognize() fallback.
FLOE_BATCH_STT_URL = "https://credit-api.floelabs.xyz/v1/audio/transcriptions"

# Usage tag stamped on emitted metrics so a mixed session can tell Floe-routed
# speech apart from other providers. Mirrors the LLM/TTS legs' provider tag.
FLOE_PROVIDER = "floe"

# LiveKit transports deliver 16-bit PCM; that is the only encoding the plugin
# streams. The Floe endpoint also accepts mulaw/alaw, but LiveKit does not
# produce them, so the wire encoding is fixed.
_ENCODING = "linear16"

_MIN_SAMPLE_RATE = 8000
_MAX_SAMPLE_RATE = 48000


def _speech_events(
    data: dict, language: str, speaking: bool
) -> tuple[list[stt.SpeechEvent], bool]:
    """Map one Floe ``transcript`` frame to LiveKit speech events.

    Pure so the wire-protocol translation is unit-testable without a socket.
    Returns the events to emit and the updated ``speaking`` state. An empty
    transcript yields nothing. START_OF_SPEECH is synthesized before the first
    non-empty segment; END_OF_SPEECH after a ``speech_final`` final.
    """
    events: list[stt.SpeechEvent] = []
    text = data.get("text", "")
    if not text:
        return events, speaking

    is_final = bool(data.get("is_final"))
    speech_final = bool(data.get("speech_final"))
    lang = LanguageCode(language or "")

    if not speaking:
        speaking = True
        events.append(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

    sd = stt.SpeechData(language=lang, text=text)
    if is_final:
        events.append(
            stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[sd])
        )
        if speech_final:
            speaking = False
            events.append(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
    else:
        events.append(
            stt.SpeechEvent(type=stt.SpeechEventType.INTERIM_TRANSCRIPT, alternatives=[sd])
        )
    return events, speaking


def _build_ws_url(base_url: str, *, model: str, sample_rate: int, language: str | None) -> str:
    """Build the streaming-STT connect URL with query parameters."""
    params: dict[str, str] = {
        "model": model,
        "encoding": _ENCODING,
        "sample_rate": str(sample_rate),
    }
    if language:
        params["language"] = language
    return f"{base_url}?{urlencode(params)}"


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: STTModels | str = "deepgram/nova-3",
        language: str = "en",
        sample_rate: int = 16000,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        task_id: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = FLOE_STT_WS_URL,
        batch_url: str = FLOE_BATCH_STT_URL,
    ) -> None:
        """Create a new instance of the Floe STT.

        Streaming transcription over Floe's WebSocket — keyless, metered per
        audio-second on your Floe balance and bounded by any spend cap on the
        agent key. A non-streaming :meth:`recognize` fallback hits the batch
        ``/v1/audio/transcriptions`` endpoint.

        Args:
            model: Fully qualified ``provider/model`` STT id, e.g.
                ``"deepgram/nova-3"``.
            language: BCP-47 language hint (e.g. ``"en"``). Empty enables
                auto-detection where the model supports it.
            sample_rate: Audio sample rate in Hz (8000-48000). Frames pushed at
                another rate are resampled to this before being streamed.
            api_key: Your Floe **agent** key (``floe_...``). Falls back to the
                ``FLOE_API_KEY`` environment variable. Developer keys
                (``floe_live_...``) are rejected — streaming STT is agent-scoped.
            task_id: Optional Floe task id sent as ``X-Floe-Task-Id`` so a
                per-task budget can bound one conversation.
            base_url: Floe streaming-STT WebSocket URL. Override to point at a
                self-hosted Floe. Must be ``wss`` (``ws`` allowed for loopback).
            batch_url: Batch STT URL used by the :meth:`recognize` fallback.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                offline_recognize=True,
            )
        )
        self._api_key = _get_agent_key(api_key)
        if not _MIN_SAMPLE_RATE <= sample_rate <= _MAX_SAMPLE_RATE:
            raise ValueError(
                f"sample_rate must be between {_MIN_SAMPLE_RATE} and {_MAX_SAMPLE_RATE} Hz, "
                f"got {sample_rate}."
            )
        _require_secure_ws_url(base_url)
        _require_secure_http_url(batch_url)

        self._model = model
        self._language = language
        self._sample_rate = sample_rate
        self._task_id = task_id if is_given(task_id) else None
        self._base_url = base_url
        self._batch_url = batch_url
        self._session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return str(self._model)

    @property
    def provider(self) -> str:
        return FLOE_PROVIDER

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._task_id:
            headers["X-Floe-Task-Id"] = self._task_id
        return headers

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        return SpeechStream(
            stt=self,
            conn_options=conn_options,
            language=language if is_given(language) else self._language,
        )

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        lang = language if is_given(language) else self._language
        wav = rtc.combine_audio_frames(buffer).to_wav_bytes()

        form = aiohttp.FormData()
        form.add_field("file", wav, filename="audio.wav", content_type="audio/wav")
        form.add_field("model", str(self._model))
        if lang:
            form.add_field("language", lang)

        try:
            async with self._ensure_session().post(
                self._batch_url,
                data=form,
                headers=self._auth_headers(),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                body = await resp.read()
                if resp.status != 200:
                    raise APIStatusError(
                        message=body.decode("utf-8", "replace") or "batch STT request failed",
                        status_code=resp.status,
                    )
                cost = resp.headers.get("X-Floe-Cost-USDC")
                if cost:
                    logger.debug("floe batch STT cost: $%s USDC", cost)
                payload = json.loads(body) if body else {}
                text = payload.get("text", "") if isinstance(payload, dict) else ""
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language=LanguageCode(lang or ""), text=text)],
                )
        except APIError:
            raise
        except aiohttp.ClientError as e:
            raise APIConnectionError() from e

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        language: str,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._sample_rate)
        self._stt: STT = stt
        self._language = language
        self._speaking = False

    async def _run(self) -> None:
        stt_impl = self._stt
        url = _build_ws_url(
            stt_impl._base_url,
            model=str(stt_impl._model),
            sample_rate=stt_impl._sample_rate,
            language=self._language or None,
        )
        try:
            ws = await asyncio.wait_for(
                self._stt._ensure_session().ws_connect(url, headers=stt_impl._auth_headers()),
                self._conn_options.timeout,
            )
        except aiohttp.ClientError as e:
            raise APIConnectionError("failed to connect to Floe STT") from e

        try:
            send = asyncio.create_task(self._send_task(ws))
            recv = asyncio.create_task(self._recv_task(ws))
            try:
                done, _ = await asyncio.wait(
                    (send, recv), return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    task.result()  # surface send/recv errors (drives retry)
            finally:
                await utils.aio.gracefully_cancel(send, recv)
        finally:
            await ws.close()

    async def _send_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                # Floe endpoints server-side (Deepgram); no explicit flush frame.
                continue
            await ws.send_bytes(data.data.tobytes())
        # Input ended: close so the recv loop unwinds cleanly.
        await ws.close()

    async def _recv_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                continue  # transcripts-only downstream
            if msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            ):
                return
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning("floe STT: non-JSON message: %s", msg.data)
                continue

            mtype = data.get("type")
            if mtype == "transcript":
                events, self._speaking = _speech_events(data, self._language, self._speaking)
                for ev in events:
                    self._event_ch.send_nowait(ev)
            elif mtype == "error":
                code = data.get("code", "unknown")
                message = data.get("message", code)
                # Server closes the socket after an error (e.g. insufficient
                # balance); surface it and don't retry a policy refusal.
                raise APIError(f"Floe STT error [{code}]: {message}", retryable=False)


def _require_secure_ws_url(url: str) -> None:
    """Refuse to send the Floe agent key over a cleartext WebSocket.

    Allows wss to any host (incl. self-hosted Floe on a custom domain); allows
    ws only for loopback (local dev). Fail closed on anything else.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "wss":
        return
    if parsed.scheme == "ws" and host in {"localhost", "127.0.0.1", "::1"}:
        return
    raise ValueError(
        f"Floe requires a wss base_url; got {url!r}. Refusing to send your Floe "
        "API key over a non-TLS WebSocket."
    )


def _require_secure_http_url(url: str) -> None:
    """Refuse to send the Floe agent key + audio over a cleartext HTTP request.

    Mirrors :func:`_require_secure_ws_url` for the batch endpoint: allows https
    to any host (incl. self-hosted Floe on a custom domain); allows http only
    for loopback (local dev). Fail closed on anything else.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and host in {"localhost", "127.0.0.1", "::1"}:
        return
    raise ValueError(
        f"Floe requires an https batch_url; got {url!r}. Refusing to send your "
        "Floe API key or audio over a non-TLS connection."
    )


def _get_agent_key(key: NotGivenOr[str]) -> str:
    floe_api_key = key if is_given(key) else os.environ.get("FLOE_API_KEY")
    if not floe_api_key:
        raise ValueError(
            "FLOE_API_KEY is required, either as argument or set FLOE_API_KEY environment variable"  # noqa: E501
        )
    if not floe_api_key.startswith("floe_") or floe_api_key.startswith("floe_live_"):
        raise ValueError(
            "Floe streaming STT requires an agent key (floe_...), not a developer "
            "key (floe_live_...). Streaming STT is agent-scoped."
        )
    return floe_api_key
