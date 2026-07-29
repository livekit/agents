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
import os
import uuid
from dataclasses import dataclass, replace
from urllib.parse import unquote_to_bytes

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ._utils import (
    API_BASE_URL,
    parse_json_response,
    raise_for_audio_response,
    response_request_id,
    unwrap_data,
    validate_language,
)
from .models import Language, TTSOutputFormat
from .version import __version__

NUM_CHANNELS = 1
DEFAULT_VOICE = "am-hamen"

_OUTPUT_FORMATS: dict[TTSOutputFormat, tuple[int, str]] = {
    "mp3_44100": (44_100, "audio/mpeg"),
    "wav_44100": (44_100, "audio/wav"),
    "pcm_16000": (16_000, "audio/wav"),
}


@dataclass
class _TTSOptions:
    language: Language
    voice: str
    output_format: TTSOutputFormat
    speed: float | None
    base_url: str
    generation_timeout: float
    download_timeout: float


class TTS(tts.TTS):
    """Non-streaming speech synthesis using Addis Voices 2."""

    def __init__(
        self,
        *,
        language: Language | str = "am",
        voice: str = DEFAULT_VOICE,
        output_format: TTSOutputFormat = "pcm_16000",
        speed: float | None = None,
        api_key: str | None = None,
        base_url: str = API_BASE_URL,
        generation_timeout: float = 95.0,
        download_timeout: float = 30.0,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create an AddisAI text-to-speech client.

        Args:
            language: Synthesis language: ``"am"`` for Amharic or ``"om"`` for
                Afaan Oromo.
            voice: Voice ID from the dynamic AddisAI voice catalog.
            output_format: ``"mp3_44100"``, ``"wav_44100"``, or ``"pcm_16000"``.
                The default is WAV-wrapped 16 kHz PCM for speech pipelines.
            speed: Optional Addis Voices 2 speed setting.
            api_key: AddisAI API key. Falls back to ``ADDIS_API_KEY``.
            base_url: AddisAI API base URL.
            generation_timeout: Maximum generation time in seconds.
            download_timeout: Maximum signed-URL download time in seconds.
            http_session: Optional shared :class:`aiohttp.ClientSession`.

        Raises:
            ValueError: If required configuration is missing or unsupported.
        """
        if output_format not in _OUTPUT_FORMATS:
            raise ValueError("output_format must be 'mp3_44100', 'wav_44100', or 'pcm_16000'")
        if not voice:
            raise ValueError("voice must not be empty")
        if generation_timeout <= 0 or download_timeout <= 0:
            raise ValueError("generation_timeout and download_timeout must be positive")

        resolved_api_key = api_key or os.environ.get("ADDIS_API_KEY")
        if not resolved_api_key:
            raise ValueError("AddisAI API key is required, either as api_key or ADDIS_API_KEY")

        sample_rate, _ = _OUTPUT_FORMATS[output_format]
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        self._api_key = resolved_api_key
        self._opts = _TTSOptions(
            language=validate_language(language),
            voice=voice,
            output_format=output_format,
            speed=speed,
            base_url=base_url.rstrip("/"),
            generation_timeout=generation_timeout,
            download_timeout=download_timeout,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return "addis-voices-2"

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
        speed: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        """Update synthesis options for subsequent requests."""
        if is_given(language):
            self._opts.language = validate_language(language)
        if is_given(voice):
            if not voice:
                raise ValueError("voice must not be empty")
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = speed

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            options=replace(self._opts),
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        options: _TTSOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._addisai_tts = tts
        self._opts = options
        # This ID intentionally belongs to the stream rather than a retry attempt.
        self._client_request_id = str(uuid.uuid4())

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload: dict[str, object] = {
            "text": self.input_text,
            "voice_id": self._opts.voice,
            "language": self._opts.language,
            "output_format": self._opts.output_format,
            "client_request_id": self._client_request_id,
        }
        if self._opts.speed is not None:
            payload["voice_settings"] = {"speed": self._opts.speed}

        try:
            async with self._addisai_tts._ensure_session().post(
                f"{self._opts.base_url}/api/v1/voice/generations",
                json=payload,
                headers={
                    "x-api-key": self._addisai_tts._api_key,
                    "Accept": "application/json",
                    "X-Addis-Client": f"livekit-plugins-addisai/{__version__}",
                },
                timeout=aiohttp.ClientTimeout(
                    total=self._opts.generation_timeout,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as response:
                response_payload = await parse_json_response(response)
                data = unwrap_data(response_payload)
                provider_request_id = (
                    str(data.get("id") or "")
                    or response_request_id(response)
                    or self._client_request_id
                )

            # The provider request ID is stable across an idempotent retry. LiveKit
            # needs a new output request ID for each attempt so downstream consumers
            # can discard any partial audio emitted by a failed download.
            output_emitter._note_provider_request_id(provider_request_id)
            stream_request_id = utils.shortuuid()
            audio_url = _audio_url(data)
            await self._download_audio(
                audio_url=audio_url,
                request_id=stream_request_id,
                provider_mime_type=data.get("mime_type"),
                output_emitter=output_emitter,
            )
            output_emitter.flush()
        except asyncio.TimeoutError as exc:
            raise APITimeoutError() from exc
        except APIError:
            raise
        except aiohttp.ClientError as exc:
            raise APIConnectionError("failed to connect to AddisAI TTS") from exc
        except Exception as exc:
            raise APIConnectionError("failed to synthesize speech with AddisAI") from exc

    async def _download_audio(
        self,
        *,
        audio_url: str,
        request_id: str,
        provider_mime_type: object,
        output_emitter: tts.AudioEmitter,
    ) -> None:
        sample_rate, default_mime_type = _OUTPUT_FORMATS[self._opts.output_format]
        mime_type = (
            provider_mime_type
            if isinstance(provider_mime_type, str) and provider_mime_type
            else default_mime_type
        )

        if audio_url.startswith("data:"):
            data_mime_type, audio = _decode_data_url(audio_url)
            output_emitter.initialize(
                request_id=request_id,
                sample_rate=sample_rate,
                num_channels=NUM_CHANNELS,
                mime_type=data_mime_type or mime_type,
            )
            output_emitter.push(audio)
            return

        async with self._addisai_tts._ensure_session().get(
            audio_url,
            timeout=aiohttp.ClientTimeout(
                total=self._opts.download_timeout,
                sock_connect=self._conn_options.timeout,
            ),
        ) as response:
            await raise_for_audio_response(response)
            response_mime_type = response.headers.get("Content-Type", "").split(";", 1)[0]
            if response_mime_type and response_mime_type != "application/octet-stream":
                mime_type = response_mime_type

            output_emitter.initialize(
                request_id=request_id,
                sample_rate=sample_rate,
                num_channels=NUM_CHANNELS,
                mime_type=mime_type,
            )
            async for chunk in response.content.iter_chunked(64 * 1024):
                output_emitter.push(chunk)


def _audio_url(data: dict[str, object]) -> str:
    playback = data.get("playback")
    playback_url = playback.get("url") if isinstance(playback, dict) else None
    value = data.get("audio_url") or data.get("audio_data_url") or playback_url
    if not isinstance(value, str) or not value:
        raise APIConnectionError("AddisAI TTS response did not include an audio URL")
    return value


def _decode_data_url(url: str) -> tuple[str | None, bytes]:
    try:
        header, encoded = url.split(",", 1)
    except ValueError as exc:
        raise APIConnectionError("AddisAI returned an invalid audio data URL") from exc

    media_type = header[5:].split(";", 1)[0] or None
    try:
        if ";base64" in header:
            return media_type, base64.b64decode(encoded, validate=True)
        return media_type, unquote_to_bytes(encoded)
    except (ValueError, binascii.Error) as exc:
        raise APIConnectionError("AddisAI returned invalid inline audio data") from exc
