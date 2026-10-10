from __future__ import annotations

import asyncio
import os
from dataclasses import replace

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

GIGGY_URL = "https://giggy.ai/v1/audio/speech"
GIGGY_MODEL = "giggyspeech"
SAMPLE_RATE = 24000
NUM_CHANNELS = 1


class TTS(tts.TTS):
    """Synthesize complete text with Giggy's progressive PCM HTTP endpoint."""

    def __init__(
        self,
        *,
        voice: str | None = None,
        api_key: str | None = None,
        speed: float = 1.0,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Giggy TTS provider.

        Args:
            voice: Giggy voice UUID, or GIGGY_VOICE_ID from the environment.
            api_key: Giggy API key, or GIGGY_API_KEY from the environment.
            speed: Speech speed from 0.25 to 4, inclusive.
            http_session: Optional caller-owned HTTP session.
        """
        resolved_key = api_key or os.getenv("GIGGY_API_KEY")
        resolved_voice = voice or os.getenv("GIGGY_VOICE_ID")
        if not resolved_key:
            raise ValueError("Giggy API key is required.")
        if not resolved_voice:
            raise ValueError("Giggy voice UUID is required.")
        if not 0.25 <= speed <= 4:
            raise ValueError("Giggy speed must be between 0.25 and 4.")

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False, aligned_transcript=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._api_key = resolved_key
        self._voice = resolved_voice
        self._speed = speed
        self._http_session = http_session

    @property
    def provider(self) -> str:
        """The provider name reported in LiveKit metrics."""
        return "Giggy"

    @property
    def model(self) -> str:
        """The Giggy model used for synthesis."""
        return GIGGY_MODEL

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._http_session is not None:
            return self._http_session
        return utils.http_context.http_session()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """Submit complete text once; paid synthesis is never automatically retried."""
        if not text.strip():
            raise ValueError("Giggy synthesis text must not be empty.")
        return ChunkedStream(
            tts_instance=self,
            input_text=text,
            conn_options=replace(conn_options, max_retry=0),
        )

    async def aclose(self) -> None:
        """Leave caller-owned and LiveKit-owned HTTP sessions open."""
        return None


class ChunkedStream(tts.ChunkedStream):
    """Receive progressive Giggy PCM without buffering the entire response."""

    def __init__(
        self,
        *,
        tts_instance: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        """Create a single-request synthesis stream with retries disabled."""
        super().__init__(
            tts=tts_instance,
            input_text=input_text,
            conn_options=replace(conn_options, max_retry=0),
        )
        self._giggy = tts_instance

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload = {
            "model": GIGGY_MODEL,
            "input": self._input_text,
            "voice": self._giggy._voice,
            "response_format": "pcm",
            "sample_rate": SAMPLE_RATE,
            "speed": self._giggy._speed,
        }
        headers = {
            "Authorization": "Bearer " + self._giggy._api_key,
            "Content-Type": "application/json",
            "Accept": "application/octet-stream",
        }
        timeout = aiohttp.ClientTimeout(
            total=None, sock_connect=self._conn_options.timeout, sock_read=90
        )
        try:
            async with self._giggy._ensure_session().post(
                GIGGY_URL,
                json=payload,
                headers=headers,
                timeout=timeout,
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise APIStatusError(
                        "Giggy TTS HTTP request failed",
                        status_code=response.status,
                        request_id=response.headers.get("x-request-id"),
                        body=None,
                        retryable=False,
                    )
                content_type = (
                    response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                )
                if content_type not in {"application/octet-stream", "audio/pcm"}:
                    raise APIConnectionError(
                        "Giggy returned an unexpected audio format.", retryable=False
                    )
                output_emitter.initialize(
                    request_id=response.headers.get("x-request-id") or utils.shortuuid(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )
                total_bytes = 0
                pending = b""
                async for chunk in response.content.iter_chunked(4096):
                    if not chunk:
                        continue
                    data = pending + chunk
                    usable = len(data) & ~1
                    if usable:
                        output_emitter.push(data[:usable])
                        total_bytes += usable
                    pending = data[usable:]
                if pending:
                    raise APIConnectionError(
                        "Giggy returned an incomplete PCM sample.", retryable=False
                    )
                if total_bytes == 0:
                    raise APIConnectionError("Giggy returned no audio.", retryable=False)
        except asyncio.TimeoutError:
            raise APITimeoutError("Giggy TTS request timed out.", retryable=False) from None
        except aiohttp.ClientError:
            # Transport exceptions can contain request headers. Do not expose their cause.
            raise APIConnectionError("Giggy TTS network request failed.", retryable=False) from None
