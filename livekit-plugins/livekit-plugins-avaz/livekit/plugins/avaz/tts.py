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
import ssl
import time
import wave
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlparse, urlunparse

import websockets
import websockets.exceptions

from livekit import rtc
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
    DEFAULT_INFERENCE_TIMESTEPS,
    DEFAULT_INLINE_WARMUP_TIMEOUT_S,
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
    """Build dashboard authentication headers for the Avaz API.

    Args:
        api_key: Avaz dashboard API token.

    Returns:
        Headers carrying the token as both ``Authorization`` bearer and ``X-API-Key``.
    """
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
    raw = base_url.strip().rstrip("/")
    # urlparse treats "host:port/path" as scheme=host; require a real URL scheme.
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"Avaz base_url must include http:// or https:// and a host (got {base_url!r})"
        )
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path.rstrip("/") + "/tts/stream-input"
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))


def _assert_secure_ws_for_credentials(ws_url: str, api_key: str) -> None:
    """Refuse sending API credentials over any plaintext ``ws://`` URL.

    Dashboard tokens must use ``wss://`` (or an ``https`` ``base_url``). Local
    TTS without auth can still use ``ws://`` by omitting ``api_key`` or passing
    ``api_key=""`` to suppress ``AVAZ_API_KEY``.
    """
    if not api_key:
        return
    parsed = urlparse(ws_url)
    if parsed.scheme != "ws":
        return
    raise ValueError(
        "Avaz TTS refuses to send API credentials over unencrypted ws://. "
        "Use an https base_url (or wss:// ws_url), or omit api_key for local plaintext."
    )


def _resolve_base_url(value: NotGivenOr[str]) -> str:
    if is_given(value) and value:
        return str(value).rstrip("/")
    env = os.environ.get("AVAZ_BASE_URL", "").strip()
    return env.rstrip("/") if env else ""


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
    # An explicitly passed value (including "") opts out of the env fallback so a
    # local plaintext ws:// upstream stays usable when AVAZ_API_KEY is exported.
    if is_given(api_key):
        return str(api_key or "").strip()
    return os.environ.get("AVAZ_API_KEY", "").strip()


def _resolve_model_id(value: NotGivenOr[str]) -> str:
    if is_given(value) and value:
        return str(value).strip()
    return os.environ.get("AVAZ_AGENT_MODEL_ID", "").strip()


def _resolve_stream_model(
    *,
    stream_model: NotGivenOr[str],
    agent_model_id: str,
    model_id_explicit: bool = False,
) -> str:
    """Resolve upstream WebSocket model string (avaz1/2/3).

    Precedence: explicit ``stream_model`` > explicit non-UUID ``model_id`` >
    ``AVAZ_STREAM_MODEL`` > non-UUID env/catalog name > default.
    """
    if is_given(stream_model) and stream_model:
        return str(stream_model).strip()
    if model_id_explicit and agent_model_id and not _is_uuid(agent_model_id):
        return _stream_model_from_agent_name(agent_model_id)
    env = os.environ.get("AVAZ_STREAM_MODEL", "").strip()
    if env:
        return env
    if agent_model_id and not _is_uuid(agent_model_id):
        return _stream_model_from_agent_name(agent_model_id)
    return DEFAULT_STREAM_MODEL


def _ws_connect_kwargs(api_key: str, *, ws_url: str = "") -> dict[str, Any]:
    """Build ``websockets.connect`` kwargs (auth headers + explicit TLS context)."""
    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["additional_headers"] = build_auth_headers(api_key)
    if ws_url and urlparse(ws_url).scheme == "wss":
        # Explicit default context so certificate verification is not implicit.
        kwargs["ssl"] = ssl.create_default_context()
    return kwargs


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
    model_settings: dict[str, Any] = {
        # Upstream synthesis model string (avaz1/2/3).
        "model_id": opts.stream_model,
        "speaker_id": opts.speaker_id,
        "cfg_value": opts.cfg_value,
        "inference_timesteps": opts.inference_timesteps,
    }
    # Dashboard catalog UUID — proxy uses this to select voice/config.
    if opts.agent_model_id:
        model_settings["agent_model_id"] = opts.agent_model_id
    return {
        "model_settings": model_settings,
        "voice_settings": {"chunk_notation": opts.chunk_notation},
    }


_REDACT_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "token",
    "secret",
    "password",
    "credential",
    "session",
    "cookie",
    "bearer",
)


def _is_sensitive_log_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(fragment in normalized for fragment in _REDACT_KEY_FRAGMENTS)


def _summarize_server_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a DEBUG-safe summary: redact secrets, truncate large blobs."""
    summary: dict[str, Any] = {}
    for key, value in payload.items():
        if _is_sensitive_log_key(key):
            summary[key] = "<redacted>"
        elif key == "audio" and isinstance(value, str):
            summary[key] = f"<base64 {len(value)} chars>"
        elif isinstance(value, dict):
            summary[key] = _summarize_server_payload(value)
        elif isinstance(value, list):
            summary[key] = f"<list len={len(value)}>"
        elif isinstance(value, str) and len(value) > 200:
            summary[key] = f"<str {len(value)} chars>"
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


def _chunk_boundary_to_append(text: str, chunk_notation: str) -> str:
    """Return the chunk-boundary character to send before flush, or empty.

    Uses ``strip()`` so leading/trailing whitespace on streamed tokens does not
    hide a missing boundary (unlike slicing ``normalized[len(raw):]``).
    """
    stripped = text.strip()
    if not stripped:
        return ""
    notation = chunk_notation or "."
    if stripped[-1] in notation:
        return ""
    return notation[0]


def _normalize_text_for_chunk_notation(text: str, chunk_notation: str) -> str:
    """Ensure text ends with a chunk_notation boundary Avaz will synthesize.

    Utterances with no trailing boundary produce ``chunks_generated: 0`` on flush
    when ``chunk_notation`` is ``'.'``. Trailing ``?`` / ``!`` are preserved for
    prosody; the primary notation character is appended (no space) so the server
    still sees a chunk boundary.
    """
    normalized = text.strip()
    if not normalized:
        return normalized
    boundary = _chunk_boundary_to_append(normalized, chunk_notation)
    return normalized + boundary if boundary else normalized


def _parse_init_response(raw: str | bytes) -> dict[str, Any]:
    text = raw.decode() if isinstance(raw, bytes) else raw
    try:
        init_payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise APIConnectionError(f"Avaz TTS invalid init response: {text[:120]}") from exc
    if not isinstance(init_payload, dict):
        raise APIConnectionError(f"Avaz TTS invalid init response type: {type(init_payload)}")
    _log_server_payload(init_payload, phase="init")
    if "error" in init_payload:
        raise APIConnectionError(f"Avaz TTS init error: {_summarize_server_payload(init_payload)}")
    # Some servers may emit audio without a dedicated init ack — keep the payload.
    if "audio" in init_payload:
        return init_payload
    if init_payload.get("status") not in (None, "ready", "ok", "initialized"):
        logger.warning(
            "[Avaz TTS] unexpected init response: %s",
            _summarize_server_payload(init_payload),
        )
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
    """Minimal Avaz synthesis to warm server-side model weights.

    Receive loops exit on a short idle window (not the full ``timeout_s``) so a
    no-audio warm-up cannot stall every spoken reply for ~15s.
    """
    uri = opts.ws_url
    init_msg = _build_init_message(opts)
    # Bare chunk_notation (e.g. ".") often yields chunks_generated: 0; use a
    # short utterance that still ends with the required boundary.
    warmup_text = _normalize_text_for_chunk_notation("warmup", opts.chunk_notation)
    got_audio = False
    deadline = time.monotonic() + max(1.0, timeout_s)
    idle_s = min(1.0, max(0.3, opts.recv_idle_timeout_s))

    async with websockets.connect(
        uri,
        open_timeout=opts.connect_timeout_s,
        max_size=_WS_MAX_SIZE,
        **_ws_connect_kwargs(opts.api_key, ws_url=uri),
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
                raw = await asyncio.wait_for(ws.recv(), timeout=min(idle_s, remaining))
            except asyncio.TimeoutError:
                # Idle window elapsed — stop waiting even if no audio arrived.
                break
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
                raise APIConnectionError(
                    f"Avaz TTS warm-up server error: {_summarize_server_payload(payload)}"
                )
            elif payload.get("status") in ("closed", "done", "complete"):
                break

        try:
            await ws.send(json.dumps({"flush": True}))
        except websockets.exceptions.ConnectionClosed:
            return got_audio
        flush_deadline = time.monotonic() + min(2.0, max(0.5, deadline - time.monotonic()))
        while time.monotonic() < flush_deadline:
            remaining = flush_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=min(idle_s, remaining))
            except asyncio.TimeoutError:
                break
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

    Timing: ``recv_idle_timeout_s``, ``flush_recv_timeout_s``, and
    ``post_text_drain_s`` are WebSocket recv idle windows (not fixed sleeps).
    Tune them if the server needs longer gaps between audio chunks after flush.
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
        """Create a new Avaz dashboard / upstream WebSocket TTS instance.

        Args:
            api_key: Dashboard API token. Falls back to ``AVAZ_API_KEY``.
            base_url: Dashboard HTTP(S) base used to derive ``/tts/stream-input``.
            model_id: Dashboard agent/model id (UUID) or upstream name.
            stream_model: Upstream WebSocket ``model_id`` (e.g. ``avaz3``).
            ws_url: Explicit WebSocket URL; overrides ``base_url`` derivation.
            speaker_id: Upstream speaker index.
            cfg_value: Guidance scale for synthesis.
            inference_timesteps: Diffusion / sampling steps.
            chunk_notation: Characters treated as chunk boundaries by Avaz.
            connect_timeout_s: WebSocket open / init timeout.
            turn_timeout_s: Max duration for one synthesis turn.
            post_text_drain_s: Extra recv idle after flush for trailing audio.
            recv_idle_timeout_s: Recv idle window while text is still streaming.
            flush_recv_timeout_s: Base recv idle window after flush.
            sample_rate: Output PCM sample rate in Hz.
        """
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
            model_id_explicit=is_given(model_id) and bool(model_id),
        )
        explicit_ws = is_given(ws_url) and bool(ws_url)
        resolved_ws_url = _resolve_ws_url(ws_url, base_url)
        resolved_api_key = _resolve_api_key(api_key)
        # When an explicit ws_url is provided, ignore dashboard base_url/env for
        # API-key requirements (local unauthenticated upstream is valid).
        if resolved_base and not resolved_api_key and not explicit_ws:
            raise ValueError(
                "Avaz TTS API key is required when using dashboard base_url. "
                "Pass api_key=..., or set AVAZ_API_KEY."
            )
        if explicit_ws:
            # Avoid retaining an unused env base that disagrees with ws_url.
            resolved_base = ""
        _assert_secure_ws_for_credentials(resolved_ws_url, resolved_api_key)
        self._opts = _TTSOptions(
            ws_url=resolved_ws_url,
            base_url=resolved_base,
            api_key=resolved_api_key,
            agent_model_id=resolved_model_id if _is_uuid(resolved_model_id) else "",
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
        # Separate from _warmed so a no-audio / failed warm-up is not retried
        # before every spoken reply (each attempt can open a new WS).
        self._warmup_attempted = False
        self._warmup_lock = asyncio.Lock()

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
        """Update dashboard model id and/or upstream speaker at runtime.

        Takes effect on the next :meth:`stream` / :meth:`synthesize` call
        (in-flight streams keep the options they started with).

        Args:
            model_id: Dashboard UUID (sent as ``agent_model_id`` in WebSocket
                init) or upstream stream model name when not a UUID. A non-UUID
                value clears any previous ``agent_model_id`` so the dashboard
                does not keep the old catalog voice.
            speaker_id: Integer speaker index for ``model_settings``.
        """
        if is_given(model_id) and model_id is not None:
            mid = str(model_id)
            if _is_uuid(mid):
                self._opts.agent_model_id = mid
            else:
                self._opts.stream_model = mid
                self._opts.agent_model_id = ""
        if is_given(speaker_id) and speaker_id is not None:
            try:
                self._opts.speaker_id = int(speaker_id)
            except (TypeError, ValueError):
                logger.warning("Avaz speaker_id must be int, got %r", speaker_id)

    async def warmup(self, *, timeout_s: float = 15.0) -> bool:
        """Pre-warm WS connect, model init, and first inference before greeting.

        Warm-up is non-critical: failures are logged and return ``False``.
        Uses short recv-idle windows so a no-audio response cannot burn the
        full ``timeout_s`` window.
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

    async def _warmup_and_mark(self, *, timeout_s: float = 15.0) -> bool:
        try:
            ok = await self.warmup(timeout_s=timeout_s)
            self._warmed = ok
            return ok
        finally:
            self._warmup_attempted = True

    async def _ensure_warmed(self) -> None:
        if self._warmed or self._warmup_attempted:
            return
        async with self._warmup_lock:
            if self._warmed or self._warmup_attempted:
                return
            task = self._prewarm_task
            if task is not None:
                try:
                    # Bound the wait so a slow prewarm cannot stall the greeting.
                    # Leave the task running; it still marks attempted when finished.
                    await asyncio.wait_for(
                        asyncio.shield(task),
                        timeout=DEFAULT_INLINE_WARMUP_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    # Spent the inline budget once; do not re-wait on later turns.
                    # Leave the prewarm task running — it still sets _warmed when done.
                    self._warmup_attempted = True
                    logger.debug(
                        "[Avaz TTS] prewarm still running after %.0fs; continuing with first turn",
                        DEFAULT_INLINE_WARMUP_TIMEOUT_S,
                    )
                    return
                except Exception:
                    pass
                if self._warmed or self._warmup_attempted:
                    return
            # Bound inline warm-up so the first spoken turn cannot stall ~15s.
            # wait_for is required: timeout_s alone does not cover WS connect /
            # init-ack waits (governed by connect_timeout_s).
            try:
                await asyncio.wait_for(
                    self._warmup_and_mark(timeout_s=DEFAULT_INLINE_WARMUP_TIMEOUT_S),
                    timeout=DEFAULT_INLINE_WARMUP_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                self._warmup_attempted = True
                logger.debug(
                    "[Avaz TTS] inline warm-up exceeded %.0fs; continuing with first turn",
                    DEFAULT_INLINE_WARMUP_TIMEOUT_S,
                )

    def prewarm(self) -> None:
        """LiveKit AgentActivity calls this at session start."""
        if self._warmed or self._warmup_attempted:
            return
        if self._prewarm_task is not None and not self._prewarm_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        logger.info("[Avaz TTS] pre-warming...")
        self._prewarm_task = loop.create_task(self._warmup_and_mark(timeout_s=15.0))

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
    """Streaming synthesizer for Avaz WebSocket TTS.

    Forwards agent text chunks as they arrive over stream-input, then emits
    PCM frames through the LiveKit TTS output emitter.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        """Create a stream bound to an :class:`TTS` instance.

        Args:
            tts: Parent Avaz TTS plugin.
            conn_options: Retry / connect options from the agents framework.
        """
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._sent_text_cache: str | None = None

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # Pick up set_voice_ids / option changes made after stream() construction.
        self._opts = replace(self._tts._opts)
        node_start = time.monotonic()
        uri = self._opts.ws_url
        init_msg = _build_init_message(self._opts)
        notation = self._opts.chunk_notation or "."

        pcm_byte_count = 0
        first_text_time: float | None = None
        first_audio_time: float | None = None
        total_text_chars = 0
        audio_chunk_count = 0
        emitter_ready = False
        declared_sample_rate = self._tts.sample_rate
        sample_rate = declared_sample_rate
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()
        ws_closed = False
        resampler: rtc.AudioResampler | None = None
        input_rate: int | None = None
        flush_status: dict[str, Any] | None = None

        # Buffer only until the first non-empty token so empty/tool turns skip
        # connect, while real replies start the WebSocket as soon as LLM text
        # begins (instead of waiting for end_input).
        pending: list[str | SynthesizeStream._FlushSentinel] = []
        has_text = False
        async for data in self._input_ch:
            pending.append(data)
            if isinstance(data, str) and data.strip():
                has_text = True
                break

        if not has_text:
            if self._sent_text_cache:
                pending = [self._sent_text_cache, self._FlushSentinel()]
                has_text = True
                logger.debug(
                    "[Avaz TTS] retry using cached text (%d chars)",
                    len(self._sent_text_cache),
                )
            else:
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
                    stream=True,
                )
                output_emitter.start_segment(segment_id=segment_id)
                logger.debug("[Avaz TTS] no text for this turn; skipping synthesis")
                return

        await self._tts._ensure_warmed()

        async def _input_events() -> Any:
            for item in pending:
                yield item
            async for item in self._input_ch:
                yield item

        async def _push_pcm(pcm: bytes, sr: int) -> None:
            nonlocal emitter_ready, sample_rate, pcm_byte_count, resampler, input_rate
            if not pcm:
                return
            if not emitter_ready:
                # Always declare the constructor sample_rate so synthesize()
                # (_ChunkedStreamFromStream) and streaming agree on playback rate.
                sample_rate = declared_sample_rate
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
                    stream=True,
                )
                output_emitter.start_segment(segment_id=segment_id)
                emitter_ready = True
                input_rate = sr
                if sr != declared_sample_rate:
                    logger.warning(
                        "[Avaz TTS] resampling WAV sample_rate=%s -> declared %s",
                        sr,
                        declared_sample_rate,
                    )
                    resampler = rtc.AudioResampler(
                        input_rate=sr,
                        output_rate=declared_sample_rate,
                        num_channels=1,
                    )
            elif input_rate is not None and sr != input_rate:
                logger.warning(
                    "[Avaz TTS] mid-turn sample_rate change %s -> %s; dropping chunk",
                    input_rate,
                    sr,
                )
                return

            if resampler is not None:
                frame = rtc.AudioFrame(
                    data=pcm,
                    sample_rate=input_rate or sr,
                    num_channels=1,
                    samples_per_channel=len(pcm) // 2,
                )
                for out in resampler.push(frame):
                    out_bytes = out.data.tobytes()
                    pcm_byte_count += len(out_bytes)
                    output_emitter.push(out_bytes)
            else:
                pcm_byte_count += len(pcm)
                output_emitter.push(pcm)

        async def _handle_audio_payload(data: dict[str, Any]) -> None:
            nonlocal first_audio_time, audio_chunk_count
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
                    logger.debug(
                        "[Avaz TTS] time-to-first-audio: %.0fms",
                        (first_audio_time - first_text_time) * 1000,
                    )
            audio_chunk_count += 1
            text_chunk = str(data.get("text_chunk", "") or "")
            chunk_index = data.get("chunk_index", audio_chunk_count - 1)
            logger.debug(
                "[Avaz TTS] chunk %s: %d pcm bytes - %r",
                chunk_index,
                len(pcm),
                text_chunk[:60],
            )
            await _push_pcm(pcm, sr)

        async def _handle_payload(payload: dict[str, Any], *, phase: str) -> bool:
            """Handle one server frame. Returns True when the turn is finished."""
            nonlocal flush_status
            _log_server_payload(payload, phase=phase)
            if "audio" in payload:
                await _handle_audio_payload(payload)
                return False
            if "error" in payload:
                raise APIConnectionError(
                    f"Avaz TTS server error: {_summarize_server_payload(payload)}"
                )
            if payload.get("status") in ("closed", "done", "complete"):
                flush_status = payload
                return True
            if "status" in payload:
                flush_status = payload
            return False

        async def _run_turn(ws: Any) -> None:
            nonlocal ws_closed, first_text_time, total_text_chars
            await ws.send(json.dumps(init_msg))
            init_resp = await asyncio.wait_for(ws.recv(), timeout=self._opts.connect_timeout_s)
            init_payload = _parse_init_response(init_resp)
            if "audio" in init_payload:
                await _handle_audio_payload(init_payload)

            sent_parts: list[str] = []
            flush_sent = False
            send_done = asyncio.Event()
            terminal = asyncio.Event()

            async def send_task() -> None:
                nonlocal first_text_time, total_text_chars, flush_sent, ws_closed
                try:
                    async for data in _input_events():
                        if isinstance(data, self._FlushSentinel):
                            if not sent_parts:
                                continue
                            boundary = _chunk_boundary_to_append("".join(sent_parts), notation)
                            if boundary and not ws_closed:
                                try:
                                    await ws.send(json.dumps({"text": boundary}))
                                    sent_parts.append(boundary)
                                    logger.debug(
                                        "[Avaz TTS] appended chunk boundary %r",
                                        boundary,
                                    )
                                except websockets.exceptions.ConnectionClosed as exc:
                                    ws_closed = True
                                    if emitter_ready:
                                        logger.debug(
                                            "[Avaz TTS] WebSocket closed while "
                                            "sending chunk boundary: %s",
                                            exc,
                                        )
                                    else:
                                        raise APIConnectionError(
                                            f"Avaz TTS WebSocket closed: {exc}"
                                        ) from exc
                            if not ws_closed and not flush_sent:
                                try:
                                    await ws.send(json.dumps({"flush": True}))
                                    flush_sent = True
                                except websockets.exceptions.ConnectionClosed as exc:
                                    ws_closed = True
                                    if emitter_ready:
                                        logger.debug(
                                            "[Avaz TTS] WebSocket already closed before flush: %s",
                                            exc,
                                        )
                                    else:
                                        raise APIConnectionError(
                                            f"Avaz TTS WebSocket closed: {exc}"
                                        ) from exc
                            continue

                        if not data:
                            continue
                        if first_text_time is None:
                            first_text_time = time.monotonic()
                            self._mark_started()
                            logger.debug(
                                "[Avaz TTS] first text after %.0fms",
                                (first_text_time - node_start) * 1000,
                            )
                        sent_parts.append(data)
                        try:
                            await ws.send(json.dumps({"text": data}))
                        except websockets.exceptions.ConnectionClosed as exc:
                            ws_closed = True
                            if emitter_ready:
                                logger.debug(
                                    "[Avaz TTS] WebSocket closed while sending text: %s",
                                    exc,
                                )
                                break
                            raise APIConnectionError(f"Avaz TTS WebSocket closed: {exc}") from exc
                finally:
                    if sent_parts:
                        joined = "".join(sent_parts)
                        self._sent_text_cache = joined
                        total_text_chars = len(joined)
                        if not flush_sent and not ws_closed:
                            boundary = _chunk_boundary_to_append(joined, notation)
                            if boundary:
                                try:
                                    await ws.send(json.dumps({"text": boundary}))
                                    sent_parts.append(boundary)
                                except websockets.exceptions.ConnectionClosed as exc:
                                    ws_closed = True
                                    if not emitter_ready:
                                        raise APIConnectionError(
                                            f"Avaz TTS WebSocket closed: {exc}"
                                        ) from exc
                            if not ws_closed:
                                try:
                                    await ws.send(json.dumps({"flush": True}))
                                    flush_sent = True
                                except websockets.exceptions.ConnectionClosed as exc:
                                    ws_closed = True
                                    if not emitter_ready:
                                        raise APIConnectionError(
                                            f"Avaz TTS WebSocket closed: {exc}"
                                        ) from exc
                    send_done.set()

            async def recv_task() -> None:
                nonlocal ws_closed
                idle = max(0.05, self._opts.recv_idle_timeout_s)
                # After flush, allow a longer idle before giving up on trailing audio.
                flush_idle = max(idle, self._opts.flush_recv_timeout_s) + max(
                    0.0, self._opts.post_text_drain_s
                )
                while not terminal.is_set():
                    if ws_closed:
                        break
                    # Snapshot before await: flush may flip during a short pre-flush wait.
                    using_post_flush = flush_sent or send_done.is_set()
                    timeout = flush_idle if using_post_flush else idle
                    # Before any text is sent, wait longer so init→first-token
                    # races do not abort the receive loop.
                    if not sent_parts and not send_done.is_set():
                        timeout = max(timeout, self._opts.connect_timeout_s)
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    except asyncio.TimeoutError:
                        # Still streaming text: keep waiting for audio.
                        if not flush_sent and not send_done.is_set():
                            continue
                        # Flush landed during a short pre-flush wait — honour the
                        # full post-flush window instead of exiting early.
                        if not using_post_flush:
                            continue
                        break
                    except websockets.exceptions.ConnectionClosed as exc:
                        ws_closed = True
                        if (
                            isinstance(exc, websockets.exceptions.ConnectionClosedOK)
                            or emitter_ready
                        ):
                            logger.debug("[Avaz TTS] WebSocket closed during recv: %s", exc)
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
                    if await _handle_payload(payload, phase="recv"):
                        terminal.set()
                        break

            send_t = asyncio.create_task(send_task())
            recv_t = asyncio.create_task(recv_task())
            try:
                await asyncio.gather(send_t, recv_t)
            finally:
                for t in (send_t, recv_t):
                    if not t.done():
                        t.cancel()
                await asyncio.gather(send_t, recv_t, return_exceptions=True)

            if not emitter_ready and flush_status is not None:
                chunks_gen = int(flush_status.get("chunks_generated", -1))
                if chunks_gen == 0:
                    logger.error(
                        "[Avaz TTS] server returned 0 chunks for text (%d chars)",
                        total_text_chars,
                    )
                    raise APIConnectionError(
                        f"Avaz TTS produced no audio: {_summarize_server_payload(flush_status)}"
                    )

        async def _connect_and_run_turn() -> None:
            async with websockets.connect(
                uri,
                open_timeout=self._opts.connect_timeout_s,
                max_size=_WS_MAX_SIZE,
                **_ws_connect_kwargs(self._opts.api_key, ws_url=uri),
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

        if resampler is not None:
            for out in resampler.flush():
                out_bytes = out.data.tobytes()
                pcm_byte_count += len(out_bytes)
                output_emitter.push(out_bytes)

        if not emitter_ready:
            raise APIConnectionError(
                f"Avaz TTS produced no audio for {total_text_chars} text chars ({uri})"
            )

        total_elapsed_ms = (time.monotonic() - node_start) * 1000
        audio_total_ms = (
            (pcm_byte_count / 2 / sample_rate) * 1000 if sample_rate and pcm_byte_count else 0.0
        )
        logger.debug(
            "[Avaz TTS] turn complete: text_chars=%d, audio_chunks=%d, audio=%.0fms, total=%.0fms",
            total_text_chars,
            audio_chunk_count,
            audio_total_ms,
            total_elapsed_ms,
        )


ChunkedStream = tts.ChunkedStream
