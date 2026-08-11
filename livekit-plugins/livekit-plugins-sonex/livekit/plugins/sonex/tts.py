"""
SonexLabs TTS plugin for LiveKit Agents.

Implements the ``livekit.agents.tts.TTS`` interface so SonexLabs' Panini TTS
can be used directly in a LiveKit ``AgentSession``, the same way Cartesia,
Rime, ElevenLabs, and other providers plug in:

    from livekit.plugins import sonex

    session = AgentSession(
        tts=sonex.TTS(
            api_key="vsk_...",
            voice_id="72ly9crx9v",  # from GET /v1/voices
        ),
        # ... llm, stt, vad, turn_handling, etc.
    )

Each call to ``synthesize()`` sends the input text to SonexLabs'
``/v1/speech/stream`` endpoint, which returns audio as chunked HTTP as soon
as each sentence is ready, rather than waiting for the entire utterance to
finish generating. Audio bytes are pushed to the output emitter as they
arrive, so downstream playback can start well before the full response has
been received. SonexLabs' API does not offer a WebSocket/real-time-text
streaming endpoint, so ``TTSCapabilities.streaming`` is ``False`` and
``TTS.stream()`` is not supported — use ``synthesize()`` only.

Connections are kept alive and reused across requests: when no explicit
``http_session`` is supplied, the plugin uses LiveKit Agents' shared,
process-wide ``aiohttp.ClientSession`` (via ``utils.http_context``), which
pools and reuses TCP/TLS connections instead of reconnecting on every
synthesis call — this avoids paying DNS/TCP/TLS setup cost more than once
per process.

The response is requested as WAV. Rather than relying on LiveKit Agents'
built-in ``av``-based audio decoder (which requires the optional ``codecs``
extra and its native binary dependencies), the WAV container is parsed
directly here and raw PCM is pushed with ``mime_type="audio/pcm"``. This
keeps the plugin's only dependency on ``livekit-agents`` itself (no extras),
and avoids depending on a native/compiled decoding library at all.
"""

from __future__ import annotations

import os
from typing import AsyncIterator

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger

NUM_CHANNELS = 1
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_BASE_URL = "https://api.sonexlabs.com"


class TTS(tts.TTS):
    """SonexLabs Panini TTS for LiveKit Agents."""

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice_id: str,
        language: NotGivenOr[str] = NOT_GIVEN,
        speed: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Args:
            api_key: SonexLabs API key (``vsk_...``). Falls back to the
                ``SONEX_API_KEY`` environment variable if not given.
            voice_id: Voice ID from SonexLabs' voice library. Required —
                list available voices with ``GET /v1/voices``.
            language: Optional language override. Leave unset to let
                Panini auto-detect from the input text.
            speed: Speech rate multiplier. ``1.0`` is normal speed.
            sample_rate: Output PCM sample rate in Hz. Defaults to ``24000``.
                Note: this is the rate reported to LiveKit; the actual sample
                rate is read from SonexLabs' response WAV header at synthesis
                time (see ChunkedStream._run below).
            base_url: SonexLabs API base URL. Defaults to
                ``https://api.sonexlabs.com``.
            http_session: Optional existing ``aiohttp.ClientSession`` to reuse.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        self._api_key = api_key if is_given(api_key) else os.environ.get("SONEX_API_KEY")
        if not self._api_key:
            raise ValueError(
                "SonexLabs API key is required, either as the api_key argument "
                "or via the SONEX_API_KEY environment variable"
            )

        self._voice_id = voice_id
        self._language = language if is_given(language) else None
        self._speed = speed
        self._base_url = (base_url if is_given(base_url) else DEFAULT_BASE_URL).rstrip("/")
        self._session = http_session

    @property
    def model(self) -> str:
        return "panini"

    @property
    def provider(self) -> str:
        return "SonexLabs"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Update synthesis options for subsequent ``synthesize()`` calls."""
        if is_given(voice_id):
            self._voice_id = voice_id
        if is_given(language):
            self._language = language
        if is_given(speed):
            self._speed = speed

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


async def _split_wav_header(
    iterator: AsyncIterator[bytes],
) -> tuple[int, bytes, AsyncIterator[bytes]]:
    """Consume *iterator* until the WAV ``data`` sub-chunk is located.

    Walks the actual RIFF/WAVE chunk list (chunk id + chunk size, skipping any
    chunks that aren't "fmt " or "data") rather than assuming a fixed-size
    header, so it stays correct even if SonexLabs' response includes extra
    metadata chunks. Returns ``(sample_rate, leftover_pcm_bytes, iterator)``.
    """
    buf = bytearray()
    sample_rate: int | None = None

    async for chunk in iterator:
        buf.extend(chunk)

        if len(buf) < 12:
            continue
        if buf[0:4] != b"RIFF" or buf[8:12] != b"WAVE":
            logger.warning(
                "SonexLabs TTS: response did not start with a RIFF/WAVE header; "
                "passing bytes through unparsed."
            )
            return DEFAULT_SAMPLE_RATE, bytes(buf), iterator

        pos = 12
        while len(buf) >= pos + 8:
            chunk_id = bytes(buf[pos:pos + 4])
            chunk_size = int.from_bytes(buf[pos + 4:pos + 8], "little")
            body_start = pos + 8

            if chunk_id == b"fmt ":
                if len(buf) < body_start + 16:
                    break
                sample_rate = int.from_bytes(buf[body_start + 4:body_start + 8], "little")

            if chunk_id == b"data":
                if len(buf) < body_start:
                    break
                leftover = bytes(buf[body_start:])
                return sample_rate or DEFAULT_SAMPLE_RATE, leftover, iterator

            pos = body_start + chunk_size + (chunk_size % 2)

    return sample_rate or DEFAULT_SAMPLE_RATE, bytes(buf), iterator


class ChunkedStream(tts.ChunkedStream):
    """Synthesizes a full utterance via a single SonexLabs API request."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload: dict = {
            "text": self._input_text,
            "voice_id": self._tts._voice_id,
            "output_format": "wav",
            "speed": self._tts._speed,
        }
        if self._tts._language:
            payload["language"] = self._tts._language

        try:
            async with self._tts._ensure_session().post(
                f"{self._tts._base_url}/v1/speech/stream",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._tts._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "audio/wav",
                },
                timeout=aiohttp.ClientTimeout(
                    total=30, sock_connect=self._conn_options.timeout
                ),
            ) as resp:
                resp.raise_for_status()

                if not resp.content_type.startswith("audio"):
                    content = await resp.text()
                    logger.error("SonexLabs returned non-audio data: %s", content[:300])
                    return

                async def _iter_chunks():
                    async for chunk in resp.content.iter_chunked(8192):
                        yield chunk

                sample_rate, leftover, remaining = await _split_wav_header(_iter_chunks())

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )

                if leftover:
                    output_emitter.push(leftover)
                async for chunk in remaining:
                    output_emitter.push(chunk)

        except TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
