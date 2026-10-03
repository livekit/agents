"""Synthesize workspace voices through the 60db HTTP API."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import wave
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlsplit

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

_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_SAMPLE_RATE = 24000


@dataclass
class _Options:
    voice_id: str
    speed: float


def _validate_options(voice_id: str, speed: float) -> _Options:
    if not isinstance(voice_id, str) or not voice_id.strip():
        raise ValueError("voice_id must be an explicit 60db workspace voice ID")
    if isinstance(speed, bool) or not math.isfinite(speed) or not 0.5 <= speed <= 2.0:
        raise ValueError("speed must be finite and between 0.5 and 2.0")
    return _Options(voice_id=voice_id, speed=speed)


def _validate_metadata(record: dict[str, Any]) -> None:
    if record.get("success") is False or record.get("type") == "error" or record.get("error"):
        raise ValueError("60db reported a synthesis error")
    for key in ("encoding", "audio_encoding", "output_format"):
        if key in record and str(record[key]).lower() not in {"linear16", "pcm", "pcm16", "wav"}:
            raise ValueError("60db returned incompatible audio encoding")
    for key, expected in (
        ("sample_rate", _SAMPLE_RATE),
        ("sample_rate_hertz", _SAMPLE_RATE),
        ("channels", 1),
        ("bit_depth", 16),
    ):
        if key in record and record[key] != expected:
            raise ValueError("60db returned incompatible audio metadata")
    if "audio_config" in record:
        config = record["audio_config"]
        if not isinstance(config, dict):
            raise ValueError("60db returned invalid audio configuration")
        _validate_metadata(config)


def _record_audio(record: Any) -> bytes:
    if not isinstance(record, dict):
        raise ValueError("60db returned an invalid response object")
    _validate_metadata(record)
    result = record.get("result", record.get("backendResponse", record))
    if not isinstance(result, dict):
        raise ValueError("60db returned an invalid audio result")
    _validate_metadata(result)
    value = result.get("audioContent", result.get("audio_base64"))
    if value is None:
        return b""
    if not isinstance(value, str):
        raise ValueError("60db audio must be base64 text")
    audio = base64.b64decode(value, validate=True)
    # The SDK also accepts an initial chunk containing a base64 JSON envelope.
    if audio.startswith(b"{"):
        try:
            inner = json.loads(audio)
        except (ValueError, UnicodeDecodeError):
            return audio
        if not isinstance(inner, dict) or not any(
            key in inner for key in ("audioContent", "audio_base64", "result", "backendResponse")
        ):
            raise ValueError("60db audio envelope contains no audio")
        return _record_audio(inner)
    return audio


def _pcm(audio: bytes) -> bytes:
    if audio.startswith(b"RIFF"):
        decoded = bytearray()
        offset = 0
        while offset < len(audio):
            if audio[offset : offset + 4] != b"RIFF" or len(audio) - offset < 12:
                raise ValueError("60db returned invalid WAV framing")
            size = int.from_bytes(audio[offset + 4 : offset + 8], "little") + 8
            if size < 12 or size > len(audio) - offset:
                raise ValueError("60db returned truncated WAV audio")
            with wave.open(io.BytesIO(audio[offset : offset + size]), "rb") as wav:
                if (
                    wav.getnchannels(),
                    wav.getsampwidth(),
                    wav.getframerate(),
                    wav.getcomptype(),
                ) != (
                    1,
                    2,
                    _SAMPLE_RATE,
                    "NONE",
                ):
                    raise ValueError("60db WAV must be mono PCM16 at 24000 Hz")
                frames = wav.getnframes()
                pcm = wav.readframes(frames)
                if len(pcm) != frames * 2:
                    raise ValueError("60db returned truncated WAV audio")
                decoded.extend(pcm)
            offset += size
        audio = bytes(decoded)
    elif audio.startswith((b"ID3", b"OggS", b"fLaC")):
        raise ValueError("60db returned compressed audio instead of PCM")
    if not audio or len(audio) % 2:
        raise ValueError("60db returned empty or incomplete PCM16 audio")
    return audio


class TTS(tts.TTS):
    """60db HTTP synthesis with mono PCM16 output at 24 kHz.

    Audio is validated before emission. AgentSession uses its built-in sentence
    stream adapter for incremental text; this plugin does not open WebSockets.
    """

    def __init__(
        self,
        *,
        voice_id: str,
        api_key: str | None = None,
        speed: float = 1.0,
        base_url: str = "https://api.60db.ai",
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Configure synthesis with a workspace voice and API key.

        Args:
            voice_id: Voice ID belonging to the authenticated workspace.
            api_key: API key, or SIXTYDB_API_KEY when omitted.
            speed: Speaking speed between 0.5 and 2.0.
            base_url: HTTPS API root; HTTP is allowed only on loopback for tests.
            http_session: Optional caller-owned aiohttp session.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
        )
        self._opts = _validate_options(voice_id, speed)
        self._api_key = api_key if api_key is not None else os.environ.get("SIXTYDB_API_KEY", "")
        if not self._api_key.strip():
            raise ValueError("Provide api_key or SIXTYDB_API_KEY")
        url = urlsplit(base_url)
        if (
            not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.scheme not in {"http", "https"}
            or (url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"})
        ):
            raise ValueError(
                "base_url must be HTTPS, or HTTP on loopback, without credentials or query"
            )
        self._base_url = base_url.rstrip("/")
        self._session = http_session

    @property
    def provider(self) -> str:
        return "60db"

    @property
    def model(self) -> str:
        return "workspace-voice"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is not None:
            return self._session
        return utils.http_context.http_session()

    def update_options(
        self, *, voice_id: NotGivenOr[str] = NOT_GIVEN, speed: NotGivenOr[float] = NOT_GIVEN
    ) -> None:
        """Update voice or speed for subsequent synthesis requests."""
        self._opts = _validate_options(
            voice_id if is_given(voice_id) else self._opts.voice_id,
            speed if is_given(speed) else self._opts.speed,
        )

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        """Synthesize text, splitting requests at the provider's 5000-character limit."""
        if not text.strip():
            raise ValueError("text must not be empty")
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    """Ordered HTTP synthesis requests with bounded response buffering."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            audio = bytearray()
            remaining = self._input_text.rstrip()
            while remaining:
                end = min(len(remaining), 5000)
                if end < len(remaining):
                    # Prefer a word boundary, preserving every character.
                    boundary = max(
                        remaining.rfind(" ", 2500, end), remaining.rfind("\n", 2500, end)
                    )
                    if boundary >= 0:
                        end = boundary + 1
                piece = remaining[:end]
                if piece.strip():
                    audio.extend(await self._synthesize_piece(piece))
                if len(audio) > _MAX_RESPONSE_BYTES:
                    raise ValueError("60db audio exceeds 32 MiB")
                remaining = remaining[end:]
            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=_SAMPLE_RATE,
                num_channels=1,
                mime_type="audio/pcm",
            )
            output_emitter.push(bytes(audio))
        except asyncio.TimeoutError:
            raise APITimeoutError("60db synthesis timed out") from None
        except APIError:
            raise
        except (aiohttp.ClientError, OSError):
            raise APIConnectionError("60db synthesis connection failed") from None
        except (ValueError, TypeError, KeyError, RecursionError, EOFError, wave.Error):
            raise APIError("60db returned invalid audio", retryable=False) from None

    async def _synthesize_piece(self, text: str) -> bytes:
        async with self._tts._ensure_session().post(
            self._tts._base_url + "/tts-synthesize",
            headers={"Authorization": "Bearer " + self._tts._api_key},
            json={
                "text": text,
                "voice_id": self._opts.voice_id,
                "speed": self._opts.speed,
                "timestamp_type": "NONE",
                "audio_config": {
                    "audio_encoding": "LINEAR16",
                    "sample_rate_hertz": _SAMPLE_RATE,
                },
            },
            timeout=aiohttp.ClientTimeout(
                total=60,
                sock_connect=self._conn_options.timeout,
                sock_read=self._conn_options.timeout,
            ),
            allow_redirects=False,
        ) as response:
            if not 200 <= response.status < 300:
                raise APIStatusError(
                    "60db synthesis request failed",
                    status_code=response.status,
                    body=None,
                    request_id=None,
                )
            _validate_metadata(
                {
                    key: int(response.headers[header])
                    for key, header in (
                        ("sample_rate", "X-Sample-Rate"),
                        ("channels", "X-Channels"),
                        ("bit_depth", "X-Bit-Depth"),
                    )
                    if header in response.headers
                }
            )
            data = bytearray()
            async for chunk in response.content.iter_chunked(8192):
                data.extend(chunk)
                if len(data) > _MAX_RESPONSE_BYTES:
                    raise ValueError("60db response exceeds 32 MiB")
            content_type = response.content_type
            if content_type in {"application/x-ndjson", "application/ndjson", "text/plain"}:
                audio = b"".join(
                    _record_audio(json.loads(line)) for line in data.splitlines() if line.strip()
                )
            elif content_type == "application/json":
                audio = _record_audio(json.loads(data))
            elif content_type in {
                "audio/wav",
                "audio/x-wav",
                "audio/pcm",
                "application/octet-stream",
            }:
                audio = bytes(data)
            else:
                raise ValueError("60db returned an unsupported content type")
            return _pcm(audio)
