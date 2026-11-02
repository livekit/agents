from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from urllib.parse import urlsplit
from weakref import WeakSet

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

from ._connection import ConnectionWarmer

DEFAULT_BASE_URL = "https://api.giggy.ai/v1"
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
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Giggy TTS provider.

        Args:
            voice: Giggy voice UUID, or GIGGY_VOICE_ID from the environment.
            api_key: Giggy API key, or GIGGY_API_KEY from the environment.
            speed: Speech speed from 0.25 to 4, inclusive.
            base_url: HTTPS Giggy API v1 base URL (HTTP is allowed for explicit loopback development).
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

        try:
            parsed_base = urlsplit(base_url)
            hostname = parsed_base.hostname
            port = parsed_base.port
        except ValueError as error:
            raise ValueError("Giggy base URL must be a valid API v1 URL.") from error
        loopback = hostname in {"localhost", "127.0.0.1", "::1"}
        if (
            parsed_base.scheme not in {"https", "http"}
            or (parsed_base.scheme != "https" and not (parsed_base.scheme == "http" and loopback))
            or not hostname
            or parsed_base.username is not None
            or parsed_base.password is not None
            or "?" in base_url
            or "#" in base_url
            or parsed_base.query
            or parsed_base.fragment
            or parsed_base.netloc.endswith(":")
            or parsed_base.path.rstrip("/") != "/v1"
            or (port is not None and not 1 <= port <= 65535)
        ):
            raise ValueError(
                "Giggy base URL must be HTTPS at the exact /v1 path, except HTTP loopback development URLs."
            )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False, aligned_transcript=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._api_key = resolved_key
        self._voice = resolved_voice
        self._speed = speed
        self._http_session = http_session
        self._speech_url = base_url.rstrip("/") + "/audio/speech"
        self._streams: WeakSet[ChunkedStream] = WeakSet()
        self._closed = False
        self._connection = ConnectionWarmer(self._ensure_session, base_url)

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

    def prewarm(self) -> None:
        """Prewarm once outside a worker; recurring checks belong to the current job."""
        from livekit.agents.job import get_job_context

        if not self._closed:
            self._connection.start(keepalive=get_job_context(required=False) is not None)

    async def connect(self) -> bool:
        """Wait for the initial check and maintain the connection until aclose()."""
        if self._closed:
            raise RuntimeError("Giggy TTS provider is closed.")
        return await self._connection.connect()

    async def __aenter__(self) -> TTS:
        await self.connect()
        return self

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """Submit complete text once; paid synthesis is never automatically retried."""
        if self._closed:
            raise RuntimeError("Giggy TTS provider is closed.")
        if not text.strip():
            raise ValueError("Giggy synthesis text must not be empty.")
        stream = ChunkedStream(
            tts_instance=self,
            input_text=text,
            conn_options=replace(conn_options, max_retry=0),
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close synthesis streams without closing caller-owned or shared HTTP sessions."""
        self._closed = True
        await self._connection.aclose()
        await asyncio.gather(*(stream.aclose() for stream in list(self._streams)))
        self._streams.clear()


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
            total=None,
            connect=self._conn_options.timeout,
            sock_connect=self._conn_options.timeout,
            sock_read=self._conn_options.timeout,
        )
        try:
            async with self._giggy._ensure_session().post(
                self._giggy._speech_url,
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
