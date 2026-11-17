from __future__ import annotations

import asyncio
import os

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

DEFAULT_BASE_URL = "https://api.giggy.ai/v1"
GIGGY_MODEL = "giggyspeech"
SAMPLE_RATE = 24000
NUM_CHANNELS = 1


class TTS(tts.TTS):
    """Giggy text-to-speech over the progressive PCM HTTP endpoint."""

    def __init__(
        self,
        *,
        voice: str | None = None,
        api_key: str | None = None,
        speed: float = 1.0,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Giggy TTS instance.

        Args:
            voice: Giggy voice UUID. Defaults to the ``GIGGY_VOICE_ID`` environment variable.
            api_key: Giggy API key. Defaults to the ``GIGGY_API_KEY`` environment variable.
            speed: Speech speed from 0.25 to 4.
            base_url: Giggy API v1 base URL.
            http_session: Optional aiohttp session to use for requests.
        """
        resolved_key = api_key or os.getenv("GIGGY_API_KEY")
        resolved_voice = voice or os.getenv("GIGGY_VOICE_ID")
        if not resolved_key:
            raise ValueError("Giggy API key is required, either as argument or set GIGGY_API_KEY")
        if not resolved_voice:
            raise ValueError("Giggy voice is required, either as argument or set GIGGY_VOICE_ID")
        if not 0.25 <= speed <= 4:
            raise ValueError("Giggy speed must be between 0.25 and 4")

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._api_key = resolved_key
        self._voice = resolved_voice
        self._speed = speed
        self._base_url = base_url.rstrip("/")
        self._session = http_session

    @property
    def provider(self) -> str:
        return "Giggy"

    @property
    def model(self) -> str:
        return GIGGY_MODEL

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    """Synthesize text with a single request and stream back PCM as it arrives."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            async with self._tts._ensure_session().post(
                f"{self._tts._base_url}/audio/speech",
                headers={"Authorization": f"Bearer {self._tts._api_key}"},
                json={
                    "model": GIGGY_MODEL,
                    "input": self._input_text,
                    "voice": self._tts._voice,
                    "response_format": "pcm",
                    "sample_rate": SAMPLE_RATE,
                    "speed": self._tts._speed,
                },
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=self._conn_options.timeout,
                    sock_read=self._conn_options.timeout,
                ),
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=resp.headers.get("x-request-id") or utils.shortuuid(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
