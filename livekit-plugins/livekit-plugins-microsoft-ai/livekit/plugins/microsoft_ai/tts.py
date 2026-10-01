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
#
# Adapted from the OpenAI TTS plugin and Azure plugin's REST transport pattern:
# Microsoft AI SSML/complete-WAV requests, without catalogs or streaming claims.

from __future__ import annotations

import asyncio
import io
import re
import wave
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from ._http import Configuration, HTTPClient, positive_timeout, status_error

_WAV_FORMATS = {
    8000: "riff-8khz-16bit-mono-pcm",
    22050: "riff-22050hz-16bit-mono-pcm",
    24000: "riff-24khz-16bit-mono-pcm",
    44100: "riff-44100hz-16bit-mono-pcm",
    48000: "riff-48khz-16bit-mono-pcm",
}


@dataclass(frozen=True)
class _TTSOptions:
    model: str
    voice: str
    sample_rate: int
    language: str


def _endpoint(config: Configuration, url: str | None, region: str | None) -> str:
    configured_url = url if url is not None else config.get("MICROSOFT_AI_TTS_URL")
    if configured_url is not None:
        return config.required(configured_url, "MICROSOFT_AI_TTS_URL")
    selected_region = region if region is not None else config.get("MICROSOFT_AI_TTS_REGION")
    if not selected_region:
        raise ValueError("Set MICROSOFT_AI_TTS_URL or MICROSOFT_AI_TTS_REGION, or pass url/region")
    selected_region = selected_region.lower()
    if re.fullmatch(r"[a-z][a-z0-9-]{0,62}", selected_region) is None:
        raise ValueError("region must be a public-cloud Azure region identifier")
    return f"https://{selected_region}.tts.speech.microsoft.com/cognitiveservices/v1"


def _ssml(text: str, opts: _TTSOptions) -> bytes:
    """Build plain-text SSML structurally so text/attributes cannot inject markup."""
    speak = ElementTree.Element(
        "speak",
        {
            "version": "1.0",
            "xmlns": "http://www.w3.org/2001/10/synthesis",
            "xml:lang": opts.language,
        },
    )
    voice = ElementTree.SubElement(speak, "voice", {"name": opts.voice})
    voice.text = text
    return ElementTree.tostring(speak, encoding="unicode").encode("utf-8")


def _decode_wav(data: bytes, sample_rate: int) -> bytes:
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getframerate() != sample_rate
                or audio.getcomptype() != "NONE"
            ):
                raise APIError(
                    "Microsoft AI TTS requires PCM16 mono WAV at the configured sample rate",
                    retryable=False,
                )
            expected_bytes = audio.getnframes() * 2
            pcm = audio.readframes(audio.getnframes())
            if not expected_bytes or len(pcm) != expected_bytes:
                raise APIError(
                    "Microsoft AI TTS returned empty or truncated audio", retryable=False
                )
            return pcm
    except (wave.Error, EOFError):
        raise APIError("Microsoft AI TTS returned invalid WAV audio", retryable=False) from None


class TTS(tts.TTS):
    """Microsoft AI voices through Azure Speech's SSML REST API.

    Args:
        url: Full synthesis POST endpoint, or MICROSOFT_AI_TTS_URL. Its host/path
            are used exactly as supplied and take precedence over region.
        region: Public-cloud Azure Speech region, or MICROSOFT_AI_TTS_REGION.
            Used only when the URL is unset, not when it is empty or whitespace.
        model: Model metadata/validation, or MICROSOFT_AI_TTS_MODEL. It must match
            the full voice ID's model suffix (case-insensitively), not an alias.
        voice: Full Azure Speech voice ID, or MICROSOFT_AI_TTS_VOICE. This selects
            the model and voice in SSML.
        language: SSML language, default en-US.
        sample_rate: Expected WAV sample rate, or MICROSOFT_AI_TTS_SAMPLE_RATE.
        api_key: Azure Speech subscription key, or MICROSOFT_AI_TTS_API_KEY.
        headers: Explicit headers instead of api_key/environment lookup.
        http_session: Optional caller-owned aiohttp session.
        env_file: Explicit dotenv file, or MICROSOFT_AI_ENV_FILE. Constructor
            arguments override environment variables, which override this file.
        request_timeout: Total request/body deadline in seconds.
        max_text_length: Client-side character limit, not a claimed provider limit.
        max_audio_bytes: Maximum complete WAV response size in bytes.

    AgentSession wraps this nonstreaming provider in its existing sentence
    StreamAdapter. Native text/audio streaming and JSON/base64 responses are unsupported.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        region: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        language: str = "en-US",
        sample_rate: int | None = None,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        http_session: aiohttp.ClientSession | None = None,
        env_file: str | Path | None = None,
        request_timeout: float = 30.0,
        max_text_length: int = 4096,
        max_audio_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        config = Configuration(env_file)
        if sample_rate is None:
            try:
                sample_rate = int(config.get("MICROSOFT_AI_TTS_SAMPLE_RATE") or "")
            except ValueError:
                raise ValueError("Set MICROSOFT_AI_TTS_SAMPLE_RATE or pass sample_rate") from None
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate not in _WAV_FORMATS
        ):
            raise ValueError("sample_rate must be one of 8000, 22050, 24000, 44100, 48000")
        for name, value in (
            ("max_text_length", max_text_length),
            ("max_audio_bytes", max_audio_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        positive_timeout(request_timeout, "request_timeout")
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        model = config.required(model, "MICROSOFT_AI_TTS_MODEL")
        voice = config.required(voice, "MICROSOFT_AI_TTS_VOICE")
        voice_name, separator, voice_model = voice.rpartition(":")
        if not voice_name or not separator or voice_model.casefold() != model.casefold():
            raise ValueError("voice must be a full voice ID ending in the configured model name")
        if re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", language) is None:
            raise ValueError("language must be a language tag such as en-US")
        self._opts = _TTSOptions(
            model=model, voice=voice, sample_rate=sample_rate, language=language
        )
        self._client = HTTPClient(
            config=config,
            service="TTS",
            url=_endpoint(config, url, region),
            api_key=api_key,
            headers=headers,
            http_session=http_session,
        )
        self._client.headers = {
            key: value
            for key, value in self._client.headers.items()
            if key.lower() not in ("accept", "content-type", "x-microsoft-outputformat")
        }
        self._client.headers.update(
            {
                "Accept": "audio/wav",
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": _WAV_FORMATS[sample_rate],
            }
        )
        self._request_timeout = request_timeout
        self._max_text_length = max_text_length
        self._max_audio_bytes = max_audio_bytes
        self._streams: weakref.WeakSet[ChunkedStream] = weakref.WeakSet()
        self._closed = False

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Microsoft AI"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        if self._closed:
            raise RuntimeError("Microsoft AI TTS is closed")
        if not text.strip():
            raise ValueError("Microsoft AI TTS requires nonempty text")
        if len(text) > self._max_text_length:
            raise ValueError("Microsoft AI TTS text exceeds max_text_length")
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]", text):
            raise ValueError("Microsoft AI TTS text contains characters invalid in XML")
        positive_timeout(conn_options.timeout, "conn_options.timeout")
        stream = ChunkedStream(tts=self, input_text=text, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        self._closed = True
        await asyncio.gather(*(stream.aclose() for stream in list(self._streams)))
        await self._client.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Validate a complete WAV before exposing audio, so failed attempts emit nothing."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._closed = False

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            async with self._tts._client.session().post(
                self._tts._client.url,
                headers=self._tts._client.headers,
                data=_ssml(self.input_text, self._tts._opts),
                timeout=aiohttp.ClientTimeout(
                    total=self._tts._request_timeout, connect=self._conn_options.timeout
                ),
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise status_error("TTS", response.status)
                if response.content_type not in ("audio/wav", "audio/x-wav", "audio/wave"):
                    raise APIError("Microsoft AI TTS expected a WAV response", retryable=False)
                limit = self._tts._max_audio_bytes
                if response.content_length is not None and response.content_length > limit:
                    raise APIError(
                        "Microsoft AI TTS response exceeds max_audio_bytes", retryable=False
                    )
                data = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    data.extend(chunk)
                    if len(data) > limit:
                        raise APIError(
                            "Microsoft AI TTS response exceeds max_audio_bytes", retryable=False
                        )
                pcm = _decode_wav(bytes(data), self._tts.sample_rate)
        except asyncio.TimeoutError:
            raise APITimeoutError("Microsoft AI TTS request timed out") from None
        except (aiohttp.ClientError, ConnectionError, OSError):
            raise APIConnectionError("Microsoft AI TTS transport failed") from None

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
        )
        output_emitter.push(pcm)

    async def aclose(self) -> None:
        self._closed = True
        await super().aclose()

    async def __anext__(self) -> tts.SynthesizedAudio:
        if self._closed:
            raise StopAsyncIteration
        event = await super().__anext__()
        if self._closed:
            raise StopAsyncIteration
        return event
