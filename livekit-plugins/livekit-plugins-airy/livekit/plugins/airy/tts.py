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
import re
from dataclasses import dataclass, replace
from typing import Literal
from urllib.parse import urlsplit

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .version import __version__

Language = Literal["ko", "en"]
Style = Literal["normal", "bright", "calm", "whisper"]

DEFAULT_BASE_URL = "https://api.airy.so"
DEFAULT_MODEL = "airy-tts-v1"
DEFAULT_VOICE = "a597bb7a98fc9ec1"
DEFAULT_STYLE: Style = "normal"

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
MAX_TEXT_CHARACTERS = 1280
_SPEECH_PATH = "/v1/audio/speech/stream"
_SAFE_ERROR_LABEL = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


@dataclass(frozen=True)
class _TTSOptions:
    language: Language
    model: str
    voice: str
    style: Style
    api_key: str
    base_url: str


def _validate_choice(value: str, *, name: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        valid = ", ".join(choices)
        raise ValueError(f"{name} must be one of: {valid}")


def _validate_non_empty(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _normalize_base_url(base_url: str) -> str:
    _validate_non_empty(base_url, name="base_url")
    normalized = base_url.rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP or HTTPS URL")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not include a query string or fragment")
    if parsed.path.rstrip("/").endswith("/v1"):
        raise ValueError("base_url is the API root and must not include the /v1 path")
    return normalized


def _validate_text(text: str) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must contain at least one non-whitespace character")
    if len(text) > MAX_TEXT_CHARACTERS:
        raise ValueError(
            f"text must contain at most {MAX_TEXT_CHARACTERS} Unicode characters; "
            f"received {len(text)}"
        )


def _safe_error_value(value: object) -> str | None:
    if isinstance(value, str) and _SAFE_ERROR_LABEL.fullmatch(value):
        return value
    if isinstance(value, int):
        return str(value)
    return None


def _safe_request_id(value: object) -> str | None:
    return value if isinstance(value, str) and 0 < len(value) <= 256 else None


def _request_id_from_payload(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    return _safe_request_id(payload.get("request_id"))


async def _status_error(response: aiohttp.ClientResponse) -> APIStatusError:
    raw = bytearray()
    while len(raw) <= 65536:
        chunk = await response.content.read(min(8192, 65537 - len(raw)))
        if not chunk:
            break
        raw.extend(chunk)

    payload: object | None = None
    if len(raw) <= 65536:
        try:
            payload = json.loads(bytes(raw).decode("utf-8")) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass

    error = payload.get("error") if isinstance(payload, dict) else None
    error = error if isinstance(error, dict) else {}
    error_type = _safe_error_value(error.get("type"))
    error_code = _safe_error_value(error.get("code"))
    error_param = _safe_error_value(error.get("param"))

    request_id = _safe_request_id(response.headers.get("X-Request-Id")) or (
        _request_id_from_payload(payload)
    )
    safe_body: dict[str, object] = {"status_code": response.status}
    if error_type:
        safe_body["error_type"] = error_type
    if error_code:
        safe_body["error_code"] = error_code
    if error_param:
        safe_body["error_param"] = error_param
    if response.status == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.replace(".", "", 1).isdigit():
            safe_body["retry_after"] = retry_after

    message = f"Airy TTS request failed with status {response.status}"
    if error_code:
        message += f" (error code: {error_code})"
    return APIStatusError(
        message,
        status_code=response.status,
        request_id=request_id,
        body=safe_body,
    )


def _protocol_error(
    message: str, *, request_id: str | None, field: str, received: str | None = None
) -> APIStatusError:
    body: dict[str, object] = {"field": field}
    if received is not None:
        body["received"] = received
    return APIStatusError(
        message,
        status_code=502,
        request_id=request_id,
        body=body,
        retryable=False,
    )


def _validate_audio_response(response: aiohttp.ClientResponse, request_id: str | None) -> None:
    content_type = response.headers.get("Content-Type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "audio/pcm":
        raise _protocol_error(
            "Airy TTS returned an unexpected Content-Type",
            request_id=request_id,
            field="Content-Type",
            received=content_type or None,
        )

    expected_headers = {
        "X-Audio-Sample-Rate": "24000",
        "X-Audio-Channels": "1",
        "X-Audio-Sample-Format": "s16le",
    }
    for header, expected in expected_headers.items():
        value = response.headers.get(header)
        if value is not None and value.strip().lower() != expected:
            raise _protocol_error(
                f"Airy TTS returned an unexpected {header}",
                request_id=request_id,
                field=header,
                received=value,
            )


class TTS(tts.TTS):
    """Airy text-to-speech synthesis for LiveKit Agents.

    Airy accepts a complete utterance per HTTP request and streams raw PCM audio
    in the response. LiveKit automatically wraps this non-streaming-input TTS in
    a sentence-based :class:`livekit.agents.tts.StreamAdapter` when needed.
    """

    def __init__(
        self,
        *,
        language: Language,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        style: Style = DEFAULT_STYLE,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create an Airy TTS instance.

        Args:
            language: Synthesis and billing language, either ``"ko"`` or ``"en"``.
            model: Airy model identifier.
            voice: Airy voice identifier.
            style: Speaking style: ``normal``, ``bright``, ``calm``, or ``whisper``.
            api_key: Airy API key. Defaults to the ``AIRY_API_KEY`` environment variable.
            base_url: Airy API root. The plugin appends ``/v1/audio/speech/stream``.
            http_session: Optional existing aiohttp session. The plugin never closes it.
        """
        resolved_key = api_key if api_key is not None else os.environ.get("AIRY_API_KEY")
        if not resolved_key or not resolved_key.strip():
            raise ValueError("Airy API key is required, either as api_key= or AIRY_API_KEY")

        _validate_choice(language, name="language", choices=("ko", "en"))
        _validate_choice(style, name="style", choices=("normal", "bright", "calm", "whisper"))
        _validate_non_empty(model, name="model")
        _validate_non_empty(voice, name="voice")

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False, aligned_transcript=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._opts = _TTSOptions(
            language=language,
            model=model,
            voice=voice,
            style=style,
            api_key=resolved_key,
            base_url=_normalize_base_url(base_url),
        )
        self._session = http_session

    @property
    def model(self) -> str:
        """Return the configured Airy model identifier."""
        return self._opts.model

    @property
    def provider(self) -> str:
        """Return the provider name used in LiveKit metrics."""
        return "Airy"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is not None:
            return self._session
        return utils.http_context.http_session()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """Synthesize one complete utterance and progressively emit its PCM response."""
        _validate_text(text)
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        """Release plugin resources.

        HTTP sessions are injected by the caller or owned by LiveKit's shared
        HTTP context, so this plugin does not close them.
        """


class ChunkedStream(tts.ChunkedStream):
    """One-shot Airy synthesis over its streaming HTTP PCM endpoint."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload = {
            "input": self._input_text,
            "language": self._opts.language,
            "model": self._opts.model,
            "voice": self._opts.voice,
            "style": self._opts.style,
        }
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"LiveKit-Agents-Airy/{__version__}",
        }
        received_audio = False
        try:
            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}{_SPEECH_PATH}",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=self._conn_options.timeout,
                    sock_read=self._conn_options.timeout,
                ),
            ) as response:
                if response.status >= 400:
                    raise await _status_error(response)

                provider_request_id = _safe_request_id(response.headers.get("X-Request-Id"))
                _validate_audio_response(response, provider_request_id)
                request_id = provider_request_id or utils.shortuuid()
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )
                if provider_request_id:
                    output_emitter._note_provider_request_id(provider_request_id)

                pending = b""
                total_bytes = 0
                async for chunk in response.content.iter_any():
                    if not chunk:
                        continue
                    received_audio = True
                    total_bytes += len(chunk)
                    data = pending + chunk
                    complete_length = len(data) - (len(data) % 2)
                    if complete_length:
                        output_emitter.push(data[:complete_length])
                    pending = data[complete_length:]

                if pending:
                    raise _protocol_error(
                        "Airy TTS returned incomplete 16-bit PCM audio",
                        request_id=provider_request_id,
                        field="audio_length",
                        received=str(total_bytes),
                    )
                if total_bytes == 0:
                    raise APIStatusError(
                        "Airy TTS returned an empty audio response",
                        status_code=502,
                        request_id=provider_request_id,
                        body={"status_code": 502},
                    )
        except (APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except asyncio.TimeoutError:
            raise APITimeoutError(
                "Airy TTS request timed out", retryable=not received_audio
            ) from None
        except aiohttp.ClientError:
            raise APIConnectionError(
                "Airy TTS connection error", retryable=not received_audio
            ) from None
