# Copyright 2025 LiveKit, Inc.
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
import os
import weakref
from dataclasses import dataclass, replace
from typing import Literal, TypedDict, cast

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

TIMEPAY_TTS_BASE_URL = "https://api.tts.timepay.ai/api/v1"

# Supported languages
TimePayTTSLanguages = Literal[
    "en", "hi", "mr", "ta", "te", "gu", "kn", "ml", "bn", "pa", "od", "as"
]

# Supported voice IDs
TimePayVoiceIds = Literal[
    "Ogbs15oBevLzXsUuTtA1",  # Kartik
    "Owbs15oBevLzXsUurdA_",  # Rahul
    "PAbs15oBevLzXsUu4dCi",  # Nisha
    "PQbt15oBevLzXsUuNtD3",  # Tulsi
    "Pgbt15oBevLzXsUubdA6",  # Seema
]


class Voice(TypedDict):
    name: str
    description: str
    languages: list[str]
    gender: Literal["male", "female"]


@dataclass
class _TTSOptions:
    """Options for the TimePay AI TTS service."""

    api_key: str
    voice_id: str
    language: str
    sample_rate: int
    speed: float
    add_wav_header: bool
    base_url: str


class TTS(tts.TTS):
    """TimePay AI Text-to-Speech implementation.

    This class provides text-to-speech functionality using the TimePay AI API.
    TimePay AI specializes in hyper-realistic TTS with support for multiple
    Indian languages and emotion tags.

    Args:
        voice_id: Voice ID to use for synthesis
        language: Language code (en, hi, mr, ta, te, gu, kn, ml, bn, pa, od, as)
        sample_rate: Audio sample rate in Hz (8000, 16000, or 24000)
        speed: Speech speed multiplier (0.5 to 2.0)
        add_wav_header: Whether to add WAV header to output
        api_key: TimePay AI API key (required)
        base_url: API endpoint URL
        http_session: Optional aiohttp session to use
    """

    def __init__(
        self,
        *,
        voice_id: TimePayVoiceIds = "Ogbs15oBevLzXsUuTtA1",  # Kartik
        language: TimePayTTSLanguages = "en",
        sample_rate: int = 24000,
        speed: float = 1.0,
        add_wav_header: bool = True,
        api_key: str | None = None,
        base_url: str = TIMEPAY_TTS_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        # Validate API key
        self._api_key = api_key or os.environ.get("TIMEPAY_API_KEY")
        if not self._api_key:
            raise ValueError(
                "TimePay API key is required. Provide it directly or set TIMEPAY_API_KEY env var."
            )

        # Validate parameters
        if sample_rate not in [8000, 16000, 24000]:
            raise ValueError("Sample rate must be 8000, 16000, or 24000 Hz")

        if not 0.5 <= speed <= 2.0:
            raise ValueError("Speed must be between 0.5 and 2.0")

        self._opts = _TTSOptions(
            api_key=self._api_key,
            voice_id=voice_id,
            language=language,
            sample_rate=sample_rate,
            speed=speed,
            add_wav_header=add_wav_header,
            base_url=base_url,
        )

        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return "timepay-vox"

    @property
    def provider(self) -> str:
        return "TimePay"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        """Synthesize text to speech using TimePay AI TTS API."""
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Create a streaming TTS session."""
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close all active streams and connections."""
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        if self._session:
            await self._session.close()

    async def list_voices(self) -> list[Voice]:
        """List available voices from TimePay API."""
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with self._ensure_session().get(
                f"{self._opts.base_url}/get_voices", headers=headers
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise APIStatusError(
                        message=f"TimePay API Error: {error_text}",
                        status_code=resp.status,
                    )

                data = await resp.json()
                return cast(list[Voice], data)
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Failed to list voices: {e}") from e


class ChunkedStream(tts.ChunkedStream):
    """Chunked synthesis implementation for TimePay AI."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Execute synthesis using TimePay AI API."""
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "text": self._input_text,
            "voice_id": self._opts.voice_id,
            "language": self._opts.language,
            "sample_rate": self._opts.sample_rate,
            "speed": self._opts.speed,
            "add_wav_header": self._opts.add_wav_header,
        }

        try:
            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}/get_speech",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=self._conn_options.timeout,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise APIStatusError(
                        message=f"TimePay TTS API Error: {error_text}",
                        status_code=resp.status,
                    )

                # Initialize the audio emitter
                request_id = resp.headers.get("X-Request-ID", utils.shortuuid())
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/wav",
                )

                # Stream the audio data
                async for data, _ in resp.content.iter_chunks():
                    if data:
                        output_emitter.push(data)

                output_emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError("TimePay TTS API request timed out") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"TimePay TTS API connection error: {e}") from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming synthesis implementation for TimePay AI."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Process streaming text input and synthesize audio."""
        # Collect all text input
        text_parts = []
        async for input_data in self._input_ch:
            if isinstance(input_data, str):
                text_parts.append(input_data)
            elif isinstance(input_data, self._FlushSentinel):
                break

        if not text_parts:
            return

        # Combine all text and synthesize
        full_text = " ".join(text_parts)

        # Create a chunked stream for the combined text
        chunked_stream = ChunkedStream(
            tts=self._tts, input_text=full_text, conn_options=self._conn_options
        )

        # Run the synthesis
        await chunked_stream._run(output_emitter)
