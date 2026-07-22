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
import io
import json
import os
import re
import time
import wave
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlparse, urlunparse

import websockets
import websockets.exceptions

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import (
    DEFAULT_CFG_VALUE,
    DEFAULT_CHUNK_NOTATION,
    DEFAULT_CONNECT_TIMEOUT_S,
    DEFAULT_FLUSH_RECV_TIMEOUT_S,
    DEFAULT_FRAME_MS,
    DEFAULT_INFERENCE_TIMESTEPS,
    DEFAULT_POST_TEXT_DRAIN_S,
    DEFAULT_RECV_IDLE_TIMEOUT_S,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEAKER_ID,
    DEFAULT_STREAM_MODEL,
    DEFAULT_TURN_TIMEOUT_S,
)
from .version import __version__

USER_AGENT = f"livekit-plugins-avaz/{__version__}"


def build_auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "X-API-Key": api_key,
    }


# websockets default max_size is 1 MiB; long utterances return base64 WAV frames above that.
_WS_MAX_SIZE = 16 * 1024 * 1024
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_uuid(value: str) -> bool:
    return bool(_UUID_RE.match(value.strip()))


def _stream_model_from_agent_name(name: str) -> str:
    return name.strip().lower().replace(" ", "")


def _derive_ws_url_from_base(base_url: str) -> str:
    parsed = urlparse(base_url.rstrip("/"))
    if parsed.scheme in ("http", "https"):
        scheme = "wss" if parsed.scheme == "https" else "ws"
    else:
        scheme = parsed.scheme or "wss"
    path = parsed.path.rstrip("/") + "/tts/stream-input"
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))


def _resolve_base_url(value: NotGivenOr[str]) -> str:
    if is_given(value) and value:
        return str(value).rstrip("/")
    for env_name in ("AVAZ_BASE_URL",):
        env = os.environ.get(env_name, "").strip()
        if env:
            return env.rstrip("/")
    return ""


def _resolve_ws_url(
    ws_url: NotGivenOr[str],
    base_url: NotGivenOr[str],
) -> str:
    if is_given(ws_url) and ws_url:
        return str(ws_url).rstrip("/")
    resolved_base = _resolve_base_url(base_url)
    if resolved_base:
        return _derive_ws_url_from_base(resolved_base)
    env = os.environ.get("TTS_WS_URI", "").strip()
    if env:
        return env.rstrip("/")
    raise ValueError(
        "Avaz TTS WebSocket URL is required. Pass ws_url=..., base_url=..., or set AVAZ_BASE_URL."
    )


def _resolve_api_key(api_key: NotGivenOr[str]) -> str:
    if is_given(api_key) and api_key:
        return str(api_key).strip()
    for env_name in ("AVAZ_API_KEY",):
        env = os.environ.get(env_name, "").strip()
        if env:
            return env
    bundled = ""
    return bundled if bundled else ""


def _resolve_model_id(value: NotGivenOr[str]) -> str:
    if is_given(value) and value:
        return str(value).strip()
    env = os.environ.get("AVAZ_AGENT_MODEL_ID", "").strip()
    if env:
        return env
    return "".strip()


def _resolve_stream_model(
    *,
    stream_model: NotGivenOr[str],
    agent_model_id: str,
) -> str:
    """Resolve upstream WebSocket model string (avaz1/2/3).

    ``agent_model_id`` is the dashboard catalog id (usually a UUID). When it is
    a non-UUID agent name, derive the upstream stream model from it.
    """
    if is_given(stream_model) and stream_model:
        return str(stream_model).strip()
    env = os.environ.get("AVAZ_STREAM_MODEL", "").strip()
    if env:
        return env
    if agent_model_id and not _is_uuid(agent_model_id):
        return _stream_model_from_agent_name(agent_model_id)
    return DEFAULT_STREAM_MODEL


def _ws_connect_kwargs(api_key: str) -> dict[str, Any]:
    if not api_key:
        return {}
    return {"additional_headers": build_auth_headers(api_key)}


def _wav_pcm(wav_bytes: bytes) -> tuple[int, bytes]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    if channels != 1:
        raise APIConnectionError(f"Avaz TTS expects mono WAV, got {channels} channels")
    if sample_width != 2:
        raise APIConnectionError(f"Avaz TTS expects 16-bit PCM, got sample_width={sample_width}")
    return sample_rate, pcm


def _build_init_message(opts: _TTSOptions) -> dict[str, Any]:
    return {
        "model_settings": {
            "model_id": opts.stream_model,
            "speaker_id": opts.speaker_id,
            "cfg_value": opts.cfg_value,
            "inference_timesteps": opts.inference_timesteps,
        },
        "voice_settings": {"chunk_notation": opts.chunk_notation},
    }


def _summarize_server_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "audio" and isinstance(value, str):
            summary[key] = f"<base64 {len(value)} chars>"
        else:
            summary[key] = value
    return summary


def _log_server_payload(payload: dict[str, Any], *, phase: str = "") -> None:
    tag = "[Avaz TTS] recv"
    if phase:
        tag += f" ({phase})"
    logger.debug(
        "%s payload keys=%s summary=%s",
        tag,
        list(payload.keys()),
        _summarize_server_payload(payload),
    )


def _normalize_text_for_chunk_notation(text: str, chunk_notation: str) -> str:
    """Avaz only synthesizes after a chunk_notation boundary (often '.').

    Utterances ending with ``?`` / ``!`` / no punctuation produce
    ``chunks_generated: 0`` on flush when ``chunk_notation`` is ``'.'``.
    """
    normalized = text.strip()
    if not normalized:
        return normalized
    notation = chunk_notation or "."
    if normalized[-1] in notation:
        return normalized
    primary = notation[0]
    if normalized[-1] in "?!" and primary == ".":
        # Replace (not append) so the final char is a chunk boundary the server accepts.
        # Appending "?. " still yields chunks_generated: 0 on some Avaz builds.
        return normalized[:-1] + primary
    return normalized + primary


def _parse_init_response(raw: str | bytes) -> dict[str, Any]:
    text = raw.decode() if isinstance(raw, bytes) else raw
    logger.debug("[Avaz TTS] init response: %s", text[:500])
    try:
        init_payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise APIConnectionError(f"Avaz TTS invalid init response: {text[:500]}") from exc
    if not isinstance(init_payload, dict):
        raise APIConnectionError(f"Avaz TTS invalid init response type: {type(init_payload)}")
    _log_server_payload(init_payload, phase="init")
    if "error" in init_payload:
        raise APIConnectionError(f"Avaz TTS init error: {init_payload}")
    if init_payload.get("status") not in (None, "ready", "ok", "initialized"):
        logger.warning("[Avaz TTS] unexpected init response: %s", init_payload)
    return init_payload


@dataclass
class _TTSOptions:
    ws_url: str
    base_url: str
    api_key: str
    agent_model_id: str
    stream_model: str
    speaker_id: int
    cfg_value: float
    inference_timesteps: int
    chunk_notation: str
    connect_timeout_s: float
    turn_timeout_s: float
    post_text_drain_s: float
    recv_idle_timeout_s: float
    flush_recv_timeout_s: float


async def _warmup_turn(opts: _TTSOptions, *, timeout_s: float = 15.0) -> bool:
    """Minimal Avaz synthesis to warm server-side model weights."""
    uri = opts.ws_url
    init_msg = _build_init_message(opts)
    warmup_text = opts.chunk_notation[0] if opts.chunk_notation else "."
    got_audio = False
    deadline = time.monotonic() + max(1.0, timeout_s)

    async with websockets.connect(
        uri,
        open_timeout=opts.connect_timeout_s,
        max_size=_WS_MAX_SIZE,
        **_ws_connect_kwargs(opts.api_key),
    ) as ws:
        await ws.send(json.dumps(init_msg))
        init_resp = await asyncio.wait_for(ws.recv(), timeout=opts.connect_timeout_s)
        _parse_init_response(init_resp)

        await ws.send(json.dumps({"text": warmup_text}))

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=min(0.4, remaining))
            except asyncio.TimeoutError:
                if got_audio:
                    break
                continue
            except websockets.exceptions.ConnectionClosed as exc:
                raise APIConnectionError(
                    f"Avaz TTS WebSocket closed during warm-up: {exc}"
                ) from exc
            try:
                payload = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if "audio" in payload:
                got_audio = True
            elif "error" in payload:
                raise APIConnectionError(f"Avaz TTS warm-up server error: {payload}")

        await ws.send(json.dumps({"flush": True}))
        flush_deadline = time.monotonic() + min(2.0, max(0.5, deadline - time.monotonic()))
        while time.monotonic() < flush_deadline:
            remaining = flush_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=min(0.4, remaining))
            except asyncio.TimeoutError:
                if got_audio:
                    break
                continue
            except websockets.exceptions.ConnectionClosed:
                break
            try:
                payload = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and "audio" in payload:
                got_audio = True

    return got_audio


class TTS(tts.TTS):
    """Avaz text-to-speech over WebSocket (stream-input protocol).

    Protocol (see Tests/test_ws_avaz3.py):
      1. Connect to ``/tts/stream-input`` (dashboard proxy or direct upstream)
      2. Send ``model_settings`` + ``voice_settings`` (WS ``model_id`` is upstream string)
      3. Stream ``{"text": "..."}`` chunks; receive base64 WAV in ``{"audio": ...}``
      4. Send ``{"flush": true}`` to finish the turn

    Dashboard mode: pass ``api_key``, ``base_url``, and ``model_id`` (UUID).
    Override WebSocket URL via ``ws_url=`` or ``AVAZ_BASE_URL``.

    Timing: ``recv_idle_timeout_s`` and ``post_text_drain_s`` are WebSocket recv
    idle windows (not fixed sleeps). Tune them if the server needs longer gaps
    between audio chunks before ``flush``.
    """

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        model_id: NotGivenOr[str] = NOT_GIVEN,
        stream_model: NotGivenOr[str] = NOT_GIVEN,
        ws_url: NotGivenOr[str] = NOT_GIVEN,
        speaker_id: int = DEFAULT_SPEAKER_ID,
        cfg_value: float = DEFAULT_CFG_VALUE,
        inference_timesteps: int = DEFAULT_INFERENCE_TIMESTEPS,
        chunk_notation: str = DEFAULT_CHUNK_NOTATION,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        turn_timeout_s: float = DEFAULT_TURN_TIMEOUT_S,
        post_text_drain_s: float = DEFAULT_POST_TEXT_DRAIN_S,
        recv_idle_timeout_s: float = DEFAULT_RECV_IDLE_TIMEOUT_S,
        flush_recv_timeout_s: float = DEFAULT_FLUSH_RECV_TIMEOUT_S,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        resolved_base = _resolve_base_url(base_url)
        resolved_model_id = _resolve_model_id(model_id)
        resolved_stream_model = _resolve_stream_model(
            stream_model=stream_model,
            agent_model_id=resolved_model_id,
        )
        resolved_ws_url = _resolve_ws_url(ws_url, base_url)
        resolved_api_key = _resolve_api_key(api_key)
        if resolved_base and not resolved_api_key:
            raise ValueError(
                "Avaz TTS API key is required when using dashboard base_url. "
                "Pass api_key=..., or set AVAZ_API_KEY."
            )
        self._opts = _TTSOptions(
            ws_url=resolved_ws_url,
            base_url=resolved_base,
            api_key=resolved_api_key,
            agent_model_id=resolved_model_id,
            stream_model=resolved_stream_model,
            speaker_id=int(speaker_id),
            cfg_value=cfg_value,
            inference_timesteps=int(inference_timesteps),
            chunk_notation=chunk_notation,
            connect_timeout_s=connect_timeout_s,
            turn_timeout_s=turn_timeout_s,
            post_text_drain_s=post_text_drain_s,
            recv_idle_timeout_s=recv_idle_timeout_s,
            flush_recv_timeout_s=flush_recv_timeout_s,
        )
        self._prewarm_task: asyncio.Task[bool] | None = None
        self._warmed = False

    @property
    def model(self) -> str:
        return self._opts.agent_model_id or self._opts.stream_model

    @property
    def provider(self) -> str:
        return "avaz"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)

    def set_voice_ids(
        self,
        *,
        model_id: NotGivenOr[str | int | None] = NOT_GIVEN,
        speaker_id: NotGivenOr[str | int | None] = NOT_GIVEN,
    ) -> None:
        if is_given(model_id) and model_id is not None:
            mid = str(model_id)
            if _is_uuid(mid):
                self._opts.agent_model_id = mid
            else:
                self._opts.stream_model = mid
        if is_given(speaker_id) and speaker_id is not None:
            try:
                self._opts.speaker_id = int(speaker_id)
            except (TypeError, ValueError):
                logger.warning("Avaz speaker_id must be int, got %r", speaker_id)

    async def warmup(self, *, timeout_s: float = 15.0) -> bool:
        """Pre-warm WS connect, model init, and first inference before greeting.

        Uses shorter drain timeouts than production turns so startup stays fast.
        """
        t0 = time.monotonic()
        try:
            got_audio = await _warmup_turn(self._opts, timeout_s=timeout_s)
            elapsed_ms = (time.monotonic() - t0) * 1000
            if got_audio:
                logger.info(
                    "[Avaz TTS] warm-up done in %.0fms (ws=%s)",
                    elapsed_ms,
                    self._opts.ws_url,
                )
            else:
                logger.warning(
                    "[Avaz TTS] warm-up finished without audio in %.0fms (ws=%s)",
                    elapsed_ms,
                    self._opts.ws_url,
                )
            return got_audio
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.warning(
                "[Avaz TTS] warm-up failed in %.0fms (non-critical): %s",
                elapsed_ms,
                exc,
            )
            return False

    async def _warmup_and_mark(self) -> bool:
        ok = await self.warmup()
        self._warmed = ok
        return ok

    async def _ensure_warmed(self) -> None:
        if self._warmed:
            return
        task = self._prewarm_task
        if task is not None:
            try:
                await task
            except Exception:
                pass
            if self._warmed:
                return
        self._warmed = await self.warmup()

    def prewarm(self) -> None:
        """LiveKit AgentActivity calls this at session start."""
        if self._warmed:
            return
        if self._prewarm_task is not None and not self._prewarm_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        logger.info("[Avaz TTS] pre-warming...")
        self._prewarm_task = loop.create_task(self._warmup_and_mark())

    async def aclose(self) -> None:
        task = self._prewarm_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await super().aclose()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._sent_text_cache: str | None = None

    async def _collect_text_parts(self) -> list[str]:
        """Drain agent text before WebSocket work.

        Batching is intentional: it preserves LiveKit retry replay of
        ``_input_buffer`` and matches the validated single-utterance flow in
        test_ws_avaz3. Avaz accepts incremental ``{"text": ...}`` frames; we
        can forward tokens incrementally in a follow-up without protocol changes.
        """
        parts: list[str] = []
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                continue
            if data:
                parts.append(data)
        return parts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        await self._tts._ensure_warmed()
        node_start = time.monotonic()
        uri = self._opts.ws_url
        init_msg = _build_init_message(self._opts)

        pcm_accum = bytearray()
        first_text_time: float | None = None
        first_audio_time: float | None = None
        total_text_chars = 0
        audio_chunk_count = 0
        emitter_ready = False
        sample_rate = self._tts.sample_rate
        bytes_per_frame = max(1, int(sample_rate * DEFAULT_FRAME_MS / 1000)) * 2
        leftover = b""
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()
        last_text_chunk = ""

        text_parts = await self._collect_text_parts()
        sent_text = "".join(text_parts).strip()
        if sent_text:
            self._sent_text_cache = sent_text
        elif self._sent_text_cache:
            sent_text = self._sent_text_cache
            logger.info(
                "[Avaz TTS] retry using cached text (%d chars): %r",
                len(sent_text),
                sent_text[:120] + ("..." if len(sent_text) > 120 else ""),
            )
        else:
            raise APIConnectionError("Avaz TTS: no text received from agent stream")
        raw_sent_text = sent_text
        sent_text = _normalize_text_for_chunk_notation(sent_text, self._opts.chunk_notation)
        if sent_text != raw_sent_text:
            logger.info(
                "[Avaz TTS] normalized text for chunk_notation=%r: %r -> %r",
                self._opts.chunk_notation,
                raw_sent_text[:120] + ("..." if len(raw_sent_text) > 120 else ""),
                sent_text[:120] + ("..." if len(sent_text) > 120 else ""),
            )
        total_text_chars = len(sent_text)
        first_text_time = time.monotonic()
        logger.info(
            "[LATENCY] Avaz TTS received %d text chunks (%d chars) at %.0fms after node start",
            len(text_parts) if text_parts else 1,
            total_text_chars,
            (first_text_time - node_start) * 1000,
        )
        logger.debug(
            "[Avaz TTS] synthesizing %d chars: %r",
            total_text_chars,
            sent_text[:120] + ("..." if len(sent_text) > 120 else ""),
        )

        async def _push_pcm(pcm: bytes, sr: int) -> None:
            nonlocal emitter_ready, sample_rate, bytes_per_frame, leftover
            if not pcm:
                return
            if not emitter_ready:
                sample_rate = sr
                samples_per_frame = max(1, int(sample_rate * DEFAULT_FRAME_MS / 1000))
                bytes_per_frame = samples_per_frame * 2
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
                    stream=True,
                )
                output_emitter.start_segment(segment_id=segment_id)
                emitter_ready = True
            pcm_accum.extend(pcm)
            buf = leftover + pcm
            offset = 0
            while offset + bytes_per_frame <= len(buf):
                output_emitter.push(buf[offset : offset + bytes_per_frame])
                offset += bytes_per_frame
            leftover = buf[offset:]

        async def _handle_audio_payload(data: dict[str, Any]) -> None:
            nonlocal first_audio_time, audio_chunk_count, last_text_chunk
            b64 = data.get("audio")
            if not b64:
                return
            try:
                wav_bytes = base64.b64decode(b64)
            except Exception as exc:
                raise APIConnectionError(f"Avaz TTS invalid base64 audio: {exc}") from exc
            try:
                sr, pcm = _wav_pcm(wav_bytes)
            except wave.Error as exc:
                raise APIConnectionError(f"Avaz TTS WAV decode failed: {exc}") from exc

            if first_audio_time is None:
                first_audio_time = time.monotonic()
                if first_text_time is not None:
                    ttfa_ms = (first_audio_time - first_text_time) * 1000
                    logger.info(
                        "[LATENCY] Avaz TTS time-to-first-audio: %.0fms (from first text chunk)",
                        ttfa_ms,
                    )
            audio_chunk_count += 1
            text_chunk = str(data.get("text_chunk", "") or "")
            if text_chunk:
                last_text_chunk = text_chunk
            chunk_index = data.get("chunk_index", audio_chunk_count - 1)
            logger.debug(
                "[Avaz TTS] chunk %s: %d pcm bytes - %r",
                chunk_index,
                len(pcm),
                text_chunk[:60],
            )
            await _push_pcm(pcm, sr)

        async def _drain_audio(ws: Any, *, timeout: float) -> bool:
            """Return True if at least one audio message was received."""
            got_audio = False
            deadline = time.monotonic() + max(0.0, timeout)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                except websockets.exceptions.ConnectionClosed as exc:
                    # Server often closes cleanly after flush/status=closed; treat that as
                    # end-of-stream (matches warm-up). Only fail if we never got audio.
                    if (
                        isinstance(exc, websockets.exceptions.ConnectionClosedOK)
                        or got_audio
                        or emitter_ready
                    ):
                        logger.debug("[Avaz TTS] WebSocket closed during drain: %s", exc)
                        break
                    raise APIConnectionError(f"Avaz TTS WebSocket closed: {exc}") from exc
                try:
                    payload = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    logger.debug("[Avaz TTS] non-JSON frame ignored: %r", raw[:120])
                    continue
                if not isinstance(payload, dict):
                    logger.debug("[Avaz TTS] non-dict JSON frame ignored: %r", payload)
                    continue
                _log_server_payload(payload, phase="drain")
                if "audio" in payload:
                    got_audio = True
                    await _handle_audio_payload(payload)
                elif "error" in payload:
                    raise APIConnectionError(f"Avaz TTS server error: {payload}")
            return got_audio

        async def _run_turn(ws: Any) -> None:
            await ws.send(json.dumps(init_msg))
            init_resp = await asyncio.wait_for(ws.recv(), timeout=self._opts.connect_timeout_s)
            _parse_init_response(init_resp)

            # Send full utterance in one frame (integration tests / greeting path).
            await ws.send(json.dumps({"text": sent_text}))
            self._mark_started()
            await _drain_audio(ws, timeout=self._opts.recv_idle_timeout_s)

            # Trailing chunks: recv idle window (not a fixed sleep) before flush.
            if self._opts.post_text_drain_s > 0:
                await _drain_audio(ws, timeout=self._opts.post_text_drain_s)

            await ws.send(json.dumps({"flush": True}))
            flush_status: dict[str, Any] | None = None
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=self._opts.flush_recv_timeout_s)
                flush_text = raw.decode() if isinstance(raw, bytes) else raw
                payload = json.loads(flush_text)
                if isinstance(payload, dict):
                    _log_server_payload(payload, phase="flush")
                    if "audio" in payload:
                        await _handle_audio_payload(payload)
                    elif "error" in payload:
                        raise APIConnectionError(f"Avaz TTS server error: {payload}")
                    elif "status" in payload:
                        flush_status = payload
                else:
                    logger.debug("[Avaz TTS] non-dict flush response ignored: %r", payload)
            except asyncio.TimeoutError:
                pass
            except websockets.exceptions.ConnectionClosed as exc:
                if isinstance(exc, websockets.exceptions.ConnectionClosedOK) or emitter_ready:
                    logger.debug("[Avaz TTS] WebSocket closed during flush recv: %s", exc)
                else:
                    raise APIConnectionError(f"Avaz TTS WebSocket closed: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise APIConnectionError(
                    f"Avaz TTS invalid flush response: {flush_text[:500]!r}"
                ) from exc

            await _drain_audio(ws, timeout=self._opts.flush_recv_timeout_s)

            # Server may still be synthesizing after status=closed (test client waits 2s more).
            if not emitter_ready:
                await _drain_audio(ws, timeout=self._opts.flush_recv_timeout_s)

            if not emitter_ready and flush_status is not None:
                chunks_gen = int(flush_status.get("chunks_generated", -1))
                if chunks_gen == 0:
                    logger.error(
                        "[Avaz TTS] server returned 0 chunks for text (%d chars): %r",
                        len(sent_text),
                        sent_text[:200],
                    )
                    raise APIConnectionError(f"Avaz TTS produced no audio: {flush_status}")

        async def _connect_and_run_turn() -> None:
            async with websockets.connect(
                uri,
                open_timeout=self._opts.connect_timeout_s,
                max_size=_WS_MAX_SIZE,
                **_ws_connect_kwargs(self._opts.api_key),
            ) as ws:
                await _run_turn(ws)

        try:
            await asyncio.wait_for(
                _connect_and_run_turn(),
                timeout=self._opts.turn_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise APIConnectionError(
                f"Avaz TTS turn timed out after {self._opts.turn_timeout_s:.0f}s ({uri})"
            ) from exc
        except APIConnectionError:
            raise
        except Exception as exc:
            raise APIConnectionError(f"Avaz TTS connection failed: {exc}") from exc

        if not emitter_ready:
            raise APIConnectionError(
                f"Avaz TTS produced no audio for {total_text_chars} text chars ({uri})"
            )

        if leftover:
            padded = bytes(leftover) + b"\x00" * (bytes_per_frame - len(leftover))
            output_emitter.push(padded)
            pcm_accum.extend(b"\x00" * (bytes_per_frame - len(leftover)))

        total_elapsed_ms = (time.monotonic() - node_start) * 1000
        audio_total_ms = (
            (len(pcm_accum) / 2 / sample_rate) * 1000 if sample_rate and pcm_accum else 0.0
        )
        logger.info(
            "[LATENCY] Avaz TTS turn complete: text_chars=%d, audio_chunks=%d, "
            "audio=%.0fms, total=%.0fms",
            total_text_chars,
            audio_chunk_count,
            audio_total_ms,
            total_elapsed_ms,
        )


ChunkedStream = tts.ChunkedStream
