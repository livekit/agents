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

"""Speech-to-text for Avaz.

Each recognition uploads the utterance as a 16 kHz mono WAV to ``/stt/transcribe`` and
returns Avaz's transcript as a single final event. Avaz transcribes Turkish only.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

import aiohttp
import numpy as np

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
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .models import DEFAULT_BASE_URL
from .version import __version__

USER_AGENT = f"livekit-plugins-avaz/{__version__}"
TRANSCRIBE_PATH = "/stt/transcribe"
MODEL = "avaz-stt-v1"
# Avaz transcribes 16 kHz mono audio, and the smaller upload also arrives sooner
SAMPLE_RATE = 16000
LANGUAGE = LanguageCode("tr")
# connecting is bounded by conn_options.timeout, this bounds the whole request
REQUEST_TIMEOUT = 30.0

_HTTP_SCHEMES = {"https": "https", "http": "http", "wss": "https", "ws": "http"}


@dataclass
class _STTOptions:
    api_key: str | None
    url: str


class STT(stt.STT):
    """Turkish speech-to-text with Avaz, one request per utterance."""

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of Avaz STT.

        Avaz STT is not streaming: ``AgentSession`` transcribes each utterance once its VAD
        detects the end of speech.

        Args:
            api_key: The Avaz API key. Defaults to the ``AVAZ_API_KEY`` environment
                variable, unless ``base_url`` points to a different endpoint than the hosted
                API. A key is only accepted with an ``https``/``wss`` or loopback ``base_url``.
            base_url: The Avaz API base URL, for example of an on-premises deployment.
                Defaults to the ``AVAZ_BASE_URL`` environment variable, or to the hosted API.
            http_session: The aiohttp session to use. Defaults to the job's shared session.
        """
        resolved_base_url = (
            base_url if is_given(base_url) else os.environ.get("AVAZ_BASE_URL") or DEFAULT_BASE_URL
        )
        url = _to_transcribe_url(resolved_base_url)
        is_hosted = url == _to_transcribe_url(DEFAULT_BASE_URL)
        resolved_api_key: str | None = None
        if is_given(api_key):
            resolved_api_key = api_key
        elif is_hosted or not is_given(base_url):
            # the key from the environment only goes to the hosted API or AVAZ_BASE_URL
            resolved_api_key = os.environ.get("AVAZ_API_KEY")

        if not resolved_api_key and is_hosted:
            raise ValueError(
                "Avaz API key is required, either as argument or set AVAZ_API_KEY"
                " environment variable"
            )
        if resolved_api_key and not _is_secure_or_loopback(url):
            raise ValueError(
                "Avaz API key is only sent over https:// or wss://, use a secure base_url or"
                ' pass api_key="" if the deployment does not require a key'
            )

        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._opts = _STTOptions(api_key=resolved_api_key or None, url=url)
        self._session = http_session

    @property
    def model(self) -> str:
        """The Avaz speech-to-text model, ``"avaz-stt-v1"``."""
        return MODEL

    @property
    def provider(self) -> str:
        """The provider name, ``"Avaz"``."""
        return "Avaz"

    def _ensure_session(self) -> aiohttp.ClientSession:
        return self._session or utils.http_context.http_session()

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        if is_given(language) and LanguageCode(language).language != LANGUAGE:
            raise ValueError(f"Avaz STT only transcribes Turkish, got language={language!r}")

        form = aiohttp.FormData()
        form.add_field("file", _to_wav(buffer), filename="audio.wav", content_type="audio/wav")
        headers = {"User-Agent": USER_AGENT}
        if self._opts.api_key:
            # aiohttp drops Authorization, unlike custom headers, on cross-origin redirects
            headers["Authorization"] = f"Bearer {self._opts.api_key}"

        try:
            async with self._ensure_session().post(
                self._opts.url,
                data=form,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=REQUEST_TIMEOUT, sock_connect=conn_options.timeout
                ),
            ) as resp:
                if resp.status >= 400:
                    # no body: retries log str(e), and Avaz's error text can quote request details
                    raise APIStatusError(
                        "Avaz rejected the transcription request", status_code=resp.status
                    )
                try:
                    payload = await resp.json(content_type=None)
                except ValueError:  # json.JSONDecodeError is a ValueError
                    raise APIError("Avaz sent a response that is not valid JSON") from None
        except asyncio.TimeoutError:
            raise APITimeoutError("Avaz transcription timed out") from None
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Avaz connection error ({type(e).__name__})") from None

        return _parse_transcript(payload)


def _to_wav(buffer: AudioBuffer) -> bytes:
    """Return the buffer as a 16 kHz mono WAV, the format Avaz transcribes."""
    frame = rtc.combine_audio_frames(buffer)
    if frame.num_channels > 1:
        # Avaz transcribes every channel, so stereo speech would come back twice
        samples = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
        mono = samples.mean(axis=1).astype(np.int16)
        frame = rtc.AudioFrame(mono.tobytes(), frame.sample_rate, 1, len(mono))

    if frame.sample_rate != SAMPLE_RATE:
        resampler = rtc.AudioResampler(frame.sample_rate, SAMPLE_RATE)
        frames = [*resampler.push(frame), *resampler.flush()]
        frame = (
            rtc.combine_audio_frames(frames) if frames else rtc.AudioFrame(b"", SAMPLE_RATE, 1, 0)
        )

    return frame.to_wav_bytes()


def _parse_transcript(payload: Any) -> stt.SpeechEvent:
    """Turn an Avaz ``/stt/transcribe`` response into a final transcript event."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict) or payload.get("success") is not True:
        raise APIError("Avaz sent an unexpected transcription response")

    texts = data.get("text")
    segments = data.get("segments") or []
    if not isinstance(texts, list) or not isinstance(segments, list):
        raise APIError("Avaz sent an unexpected transcription response")

    try:
        text = " ".join(t.strip() for t in texts if t.strip())
        times = [(float(s["start_time"]), float(s["end_time"])) for s in segments]
    except (AttributeError, KeyError, TypeError, ValueError):
        raise APIError("Avaz sent an unexpected transcription response") from None

    language = data.get("language_used")
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(
                language=LanguageCode(language) if isinstance(language, str) else LANGUAGE,
                text=text,
                start_time=min((start for start, _ in times), default=0.0),
                end_time=max((end for _, end in times), default=0.0),
            )
        ],
    )


def _to_transcribe_url(base_url: str) -> str:
    parsed = urlparse(base_url.strip())
    scheme = _HTTP_SCHEMES.get(parsed.scheme.lower())
    if scheme is None or not parsed.netloc:
        raise ValueError("Avaz base_url must be an http(s):// or ws(s):// URL")

    path = parsed.path.rstrip("/") + TRANSCRIBE_PATH
    return urlunparse((scheme, parsed.netloc, path, "", parsed.query, ""))


def _is_secure_or_loopback(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return True

    host = parsed.hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
