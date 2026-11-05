"""FonadaLabs TTS plugin for LiveKit — Klone V2 Flash / Klone V2 Pro.

- ``klone_v2_flash`` (default): streams raw 24 kHz PCM over a pooled WebSocket
  (``wss://api.fonada.ai/v1/voice-clone/ws``). One connection carries many
  utterances, so it is the lowest-latency option for voice agents.
- ``klone_v2_pro``: one-shot ``POST /v1/voice-clone/chunks`` (Pro does not
  support streaming or the WebSocket). Audio is requested as PCM and pushed to
  LiveKit as the response body arrives.

Docs: https://fonadalabs.ai/docs/text-to-speech/klone-v2-flash
      https://fonadalabs.ai/docs/text-to-speech/klone-v2-pro
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from ._utils import parse_error
from .log import logger
from .models import (
    LANGUAGE_ALIASES,
    TTS_LANGUAGES,
    TTSFlashLanguages,
    TTSModels,
    resolve_language,
)

FONADALABS_API_BASE_URL = "https://api.fonada.ai"

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

# Per-request text limit for both Klone V2 models.
_MAX_TEXT_CHARS = 5000
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[।.!?\n])\s*")

# The server closes idle WebSockets after 5 minutes.
_WS_MAX_IDLE_SECS = 240.0


class _StaleConnectionError(APIConnectionError):
    """A pooled WebSocket died before the server answered; safe to reconnect."""


def _split_text(text: str, max_chars: int = _MAX_TEXT_CHARS) -> list[str]:
    """Split *text* at sentence boundaries into chunks of at most *max_chars*."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    buf = ""
    for sent in (s.strip() for s in _SENTENCE_SPLIT_RE.split(text)):
        if not sent:
            continue
        while len(sent) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(sent[:max_chars])
            sent = sent[max_chars:]
        if buf and len(buf) + 1 + len(sent) > max_chars:
            chunks.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}" if buf else sent
    if buf:
        chunks.append(buf)
    return chunks


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


@dataclass
class _TTSOptions:
    model: str
    language: str
    share_id: str
    instruct: str | None
    num_step: int | None
    speed: float | None


class TTS(tts.TTS):
    """FonadaLabs Text-to-Speech plugin for LiveKit (Klone V2 Flash by default).

    Args:
        api_key:      FonadaLabs API key (or ``FONADALABS_API_KEY`` env var).
        model:        ``"klone_v2_flash"`` (default, streaming) or ``"klone_v2_pro"``.
        language:     ISO code or display name, e.g. ``"hi"`` / ``"Hindi"``.
                      See :data:`~.models.KLONE_V2_FLASH_LANGUAGES` and
                      :data:`~.models.KLONE_V2_PRO_LANGUAGES`.
        share_id:     Voice Arena voice ``share_id`` (8 characters, e.g. ``"UFNPsnEd"``).
                      Required; voice names are not accepted.
        instruct:     Style hint, e.g. ``"calm, friendly"`` (max 500 chars).
        num_step:     Diffusion steps, 8–64 (Flash only).
        speed:        Speaking rate, 0.25–4.0. FonadaLabs notes this is currently
                      unreliable on Flash.
        base_url:     Override the API base URL.
        http_session: Reuse an existing :class:`aiohttp.ClientSession`.
    """

    def __init__(
        self,
        *,
        share_id: str,
        api_key: str | None = None,
        model: TTSModels | str = "klone_v2_flash",
        language: TTSFlashLanguages | str = "hi",
        instruct: str | None = None,
        num_step: int | None = None,
        speed: float | None = None,
        base_url: str = FONADALABS_API_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        self._api_key = api_key or os.environ.get("FONADALABS_API_KEY")
        if not self._api_key:
            raise ValueError(
                "FonadaLabs API key is required. "
                "Pass it directly or set the FONADALABS_API_KEY environment variable."
            )

        if model not in TTS_LANGUAGES:
            raise ValueError(
                f"Unsupported TTS model {model!r}. Supported: {', '.join(TTS_LANGUAGES)}"
            )

        self._opts = _TTSOptions(
            model=model,
            language=_resolve_tts_language(model, language),
            share_id=_validate_share_id(share_id),
            instruct=_validate_instruct(instruct),
            num_step=_validate_num_step(num_step),
            speed=_validate_speed(speed),
        )
        if model == "klone_v2_pro" and num_step is not None:
            logger.warning("num_step is only supported by klone_v2_flash; it will be ignored")

        self._base_url = base_url.rstrip("/")
        self._ws_url = (
            self._base_url.replace("https://", "wss://").replace("http://", "ws://")
            + "/v1/voice-clone/ws"
        )
        self._chunks_url = f"{self._base_url}/v1/voice-clone/chunks"
        self._session = http_session

        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=_WS_MAX_IDLE_SECS,
            mark_refreshed_on_get=True,
        )
        self._streams: set[SynthesizeStream] = set()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "FonadaLabs"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        share_id: NotGivenOr[str] = NOT_GIVEN,
        instruct: NotGivenOr[str | None] = NOT_GIVEN,
        num_step: NotGivenOr[int | None] = NOT_GIVEN,
        speed: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        if utils.is_given(language):
            self._opts.language = _resolve_tts_language(self._opts.model, language)
        if utils.is_given(share_id):
            self._opts.share_id = _validate_share_id(share_id)
        if utils.is_given(instruct):
            self._opts.instruct = _validate_instruct(instruct)
        if utils.is_given(num_step):
            self._opts.num_step = _validate_num_step(num_step)
        if utils.is_given(speed):
            self._opts.speed = _validate_speed(speed)

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        try:
            ws = await asyncio.wait_for(
                self._ensure_session().ws_connect(
                    self._ws_url, headers={"Authorization": f"Bearer {self._api_key}"}
                ),
                timeout,
            )
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(
                message=f"FonadaLabs TTS WebSocket handshake failed: {e.message}",
                status_code=e.status,
            ) from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Timed out connecting to FonadaLabs TTS") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Failed to connect to FonadaLabs TTS: {e}") from e

        try:
            msg = await ws.receive(timeout=timeout)
        except asyncio.TimeoutError as e:
            await ws.close()
            raise APITimeoutError("Timed out waiting for FonadaLabs TTS ready message") from e

        if msg.type != aiohttp.WSMsgType.TEXT:
            await ws.close()
            raise APIConnectionError(f"FonadaLabs TTS closed before ready ({msg.type})")
        data = json.loads(msg.data)
        if data.get("type") != "ready":
            await ws.close()
            code, message = parse_error(data)
            raise APIStatusError(
                message=f"FonadaLabs TTS rejected connection [{code}]: {message}",
                status_code=int(data.get("status") or 403),
                retryable=False,
            )
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def prewarm(self) -> None:
        if self._opts.model == "klone_v2_flash":
            self._pool.prewarm()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

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
        await super().aclose()


# ---------------------------------------------------------------------------
# SynthesizeStream
# ---------------------------------------------------------------------------


class SynthesizeStream(tts.SynthesizeStream):
    """Synthesizes one segment (one agent reply) per stream.

    The whole segment is sent as a single request: the server splits it by
    sentence and starts streaming audio for the first sentence immediately.
    Sending one request per reply (instead of per sentence) keeps usage within
    the default rate limit of 10 requests per minute.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._text: str | None = None

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        # _run may be retried; the input channel can only be consumed once.
        if self._text is None:
            parts: list[str] = []
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    break
                parts.append(data)
            self._text = "".join(parts)

        chunks = _split_text(self._text)
        if not chunks:
            return

        self._mark_started()

        if self._opts.model == "klone_v2_flash":
            await self._run_ws(chunks, output_emitter)
        else:
            for chunk in chunks:
                await self._run_http(chunk, output_emitter)

        output_emitter.flush()

    async def _run_ws(self, chunks: list[str], output_emitter: tts.AudioEmitter) -> None:
        ws = await self._tts._pool.get(timeout=self._conn_options.timeout)
        # Only return the connection to the pool if every utterance finished
        # cleanly; otherwise leftover audio would leak into the next reply.
        reusable = False
        try:
            for i, chunk in enumerate(chunks):
                try:
                    await self._speak_ws(ws, chunk, output_emitter)
                except _StaleConnectionError as e:
                    # The server closed an idle pooled socket. Nothing was sent
                    # or received for this utterance, so reconnect and resend
                    # instead of surfacing an error and a framework retry.
                    if i != 0:
                        raise
                    logger.debug(f"TTS pooled WebSocket was stale ({e}); reconnecting")
                    self._tts._pool.remove(ws)
                    ws = await self._tts._pool.get(timeout=self._conn_options.timeout)
                    await self._speak_ws(ws, chunk, output_emitter)
            reusable = True
        except APIStatusError:
            reusable = not ws.closed  # server keeps the socket open after an error
            raise
        finally:
            if reusable:
                self._tts._pool.put(ws)
            else:
                self._tts._pool.remove(ws)

    async def _speak_ws(
        self, ws: aiohttp.ClientWebSocketResponse, text: str, output_emitter: tts.AudioEmitter
    ) -> None:
        request_id = utils.shortuuid()
        payload: dict[str, Any] = {
            "text": text,
            "share_id": self._opts.share_id,
            "language": self._opts.language,
            "output_audio_codec": "pcm",
            "request_id": request_id,
        }
        if self._opts.instruct:
            payload["instruct"] = self._opts.instruct
        if self._opts.num_step is not None:
            payload["num_step"] = self._opts.num_step
        if self._opts.speed is not None:
            payload["speed"] = self._opts.speed

        if ws.closed:
            raise _StaleConnectionError("FonadaLabs TTS WebSocket already closed")
        try:
            await ws.send_str(json.dumps(payload))
        except (aiohttp.ClientError, ConnectionResetError) as e:
            raise _StaleConnectionError(f"FonadaLabs TTS send failed: {e}") from e
        got_reply = False

        logger.debug(f"TTS utterance sent ({len(text)} chars)")

        while True:
            try:
                msg = await ws.receive(timeout=self._conn_options.timeout)
            except asyncio.TimeoutError as e:
                raise APITimeoutError("FonadaLabs TTS timed out waiting for audio") from e

            if msg.type == aiohttp.WSMsgType.BINARY:
                got_reply = True
                output_emitter.push(msg.data)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                got_reply = True
                data = json.loads(msg.data)
                msg_type = data.get("type")
                if data.get("request_id") not in (None, request_id):
                    continue
                if msg_type == "start":
                    continue
                if msg_type == "end":
                    logger.debug(
                        f"TTS utterance done — first_audio={data.get('first_audio_ms')}ms "
                        f"total={data.get('total_ms')}ms bytes={data.get('audio_bytes')}"
                    )
                    return
                if msg_type == "error":
                    code, message = parse_error(data)
                    raise APIStatusError(
                        message=f"FonadaLabs TTS error [{code}]: {message}",
                        status_code=int(data.get("status") or 500),
                        body=data,
                    )
            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            ):
                err_cls = APIConnectionError if got_reply else _StaleConnectionError
                raise err_cls(
                    f"FonadaLabs TTS WebSocket closed unexpectedly (code={ws.close_code})"
                )
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError(f"FonadaLabs TTS WebSocket error: {ws.exception()}")

    async def _run_http(self, text: str, output_emitter: tts.AudioEmitter) -> None:
        form: dict[str, str] = {
            "model": self._opts.model,
            "text": text,
            "share_id": self._opts.share_id,
            "language": self._opts.language,
            "output_audio_codec": "pcm",
        }
        if self._opts.instruct:
            form["instruct"] = self._opts.instruct
        if self._opts.speed is not None:
            form["speed"] = str(self._opts.speed)

        data = aiohttp.FormData()
        for key, value in form.items():
            data.add_field(key, value)

        try:
            async with self._tts._ensure_session().post(
                self._tts._chunks_url,
                data=data,
                headers={"Authorization": f"Bearer {self._tts._api_key}"},
                # Pro is non-streaming: the whole clip is generated before the
                # response starts, and generation time grows with text length
                # (~2.4s per 100 chars measured). The default 10s conn timeout
                # would fail most replies over ~400 chars, so allow at least 60s.
                timeout=aiohttp.ClientTimeout(total=max(self._conn_options.timeout, 60.0)),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    code, message = parse_error(body)
                    raise APIStatusError(
                        message=f"FonadaLabs TTS error [{code}]: {message}",
                        status_code=resp.status,
                        body=body,
                    )
                async for data_chunk in resp.content.iter_chunked(4096):
                    output_emitter.push(data_chunk)
        except asyncio.TimeoutError as e:
            raise APITimeoutError("FonadaLabs TTS request timed out") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"FonadaLabs TTS connection failed: {e}") from e

    async def aclose(self) -> None:
        self._tts._streams.discard(self)
        await super().aclose()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _resolve_tts_language(model: str, language: str) -> str:
    supported = TTS_LANGUAGES[model]
    code = resolve_language(language, supported, LANGUAGE_ALIASES.get(model))
    if code is None:
        raise ValueError(
            f"Language {language!r} is not supported by {model}. "
            f"Supported: {', '.join(f'{c} ({n})' for c, n in supported.items())}"
        )
    return code


def _validate_share_id(share_id: str) -> str:
    share_id = (share_id or "").strip()
    if not share_id:
        raise ValueError(
            "share_id is required: pass the Voice Arena share_id of the voice "
            '(e.g. share_id="UFNPsnEd")'
        )
    return share_id


def _validate_instruct(instruct: str | None) -> str | None:
    if not instruct:
        return None
    if len(instruct) > 500:
        logger.warning("instruct is longer than 500 characters and will be truncated")
        return instruct[:500]
    return instruct


def _validate_num_step(num_step: int | None) -> int | None:
    if num_step is not None and not 8 <= num_step <= 64:
        raise ValueError("num_step must be between 8 and 64")
    return num_step


def _validate_speed(speed: float | None) -> float | None:
    if speed is not None and not 0.25 <= speed <= 4.0:
        raise ValueError("speed must be between 0.25 and 4.0")
    return speed
