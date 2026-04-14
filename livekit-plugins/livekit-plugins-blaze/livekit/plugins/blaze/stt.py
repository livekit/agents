"""
Blaze STT Plugin for LiveKit Voice Agent

Speech-to-Text plugin that interfaces with Blaze's transcription service.

API Endpoint: POST /v1/stt/transcribe
Input: WAV audio file, language code
Output: { "transcription": str, "confidence": float, "is_final": bool, "language": str }

Supports both batch recognition and streaming (via StreamAdapter with VAD).
"""

from __future__ import annotations

import io
import time
import uuid

import httpx

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    NotGivenOr,
    stt,
)
from livekit.agents.stt import StreamAdapter
from livekit.agents.utils import AudioBuffer

from ._config import BlazeConfig
from ._utils import apply_normalization_rules, convert_pcm_to_wav
from .log import logger


class STT(stt.STT):
    """
    Blaze Speech-to-Text Plugin.

    Converts speech to text using Blaze's transcription service.
    Supports batch recognition natively. Streaming is available via
    ``with_streaming()`` which wraps this STT with a VAD-based stream adapter.

    Args:
        api_url: Base URL for the STT service.
        language: Language code for transcription (default: "vi").
        auth_token: Bearer token for authentication.
        sample_rate: Audio sample rate in Hz (default: 16000).
        normalization_rules: Dict mapping input strings to replacements.
        timeout: Request timeout in seconds (default: 30.0).
        config: Optional BlazeConfig for centralized configuration.

    Example:
        >>> from livekit.plugins import blaze
        >>>
        >>> # Batch-only (default)
        >>> stt = blaze.STT(language="vi")
        >>>
        >>> # With streaming via VAD adapter
        >>> from livekit.plugins import silero
        >>> streaming_stt = blaze.STT(language="vi").with_streaming(silero.VAD.load())
    """

    def __init__(
        self,
        *,
        api_url: str | None = None,
        language: str = "vi",
        auth_token: str | None = None,
        sample_rate: int = 16000,
        normalization_rules: dict[str, str] | None = None,
        timeout: float | None = None,
        config: BlazeConfig | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )

        self._config = config or BlazeConfig()
        self._api_url = api_url or self._config.api_url
        self._language = language
        self._auth_token = auth_token or self._config.api_token
        self._sample_rate = sample_rate
        self._timeout = timeout if timeout is not None else self._config.stt_timeout
        self._normalization_rules = normalization_rules
        self._transcribe_url = f"{self._api_url}/v1/stt/transcribe"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout, connect=5.0))

        # Frame accumulation: buffer PCM from empty STT segments so short
        # leading fragments (hesitant speech) are prepended to the next segment.
        self._pending_pcm: bytes = b""
        self._pending_sample_rate: int = sample_rate
        self._pending_num_channels: int = 1
        self._pending_empty_count: int = 0
        self._last_recognize_time: float = 0.0

        # Safety limits
        self._max_pending_duration: float = 5.0   # seconds of buffered audio
        self._max_pending_segments: int = 3        # consecutive empty segments
        self._pending_idle_timeout: float = 10.0   # auto-clear after idle gap

        logger.info("BlazeSTT initialized: url=%s, language=%s", self._api_url, self._language)

    @property
    def provider(self) -> str:
        return "Blaze"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def aclose(self) -> None:
        await self._client.aclose()
        await super().aclose()

    def update_options(
        self,
        *,
        language: str | None = None,
        auth_token: str | None = None,
        normalization_rules: dict[str, str] | None = None,
    ) -> None:
        """Update STT options at runtime."""
        if language is not None:
            self._language = language
        if auth_token is not None:
            self._auth_token = auth_token
        if normalization_rules is not None:
            self._normalization_rules = normalization_rules

    def with_streaming(self, vad: object) -> StreamAdapter:
        """Create a streaming STT by wrapping this batch STT with a VAD.

        The returned ``StreamAdapter`` pushes continuous audio frames through
        the VAD, and on each end-of-speech event calls ``recognize()``
        on the accumulated utterance.

        Args:
            vad: A VAD instance (e.g. ``silero.VAD.load()``).

        Returns:
            A ``StreamAdapter`` with ``streaming=True`` capability.

        Example:
            >>> from livekit.plugins import blaze, silero
            >>> stt = blaze.STT().with_streaming(silero.VAD.load())
        """
        from livekit.agents.vad import VAD

        if not isinstance(vad, VAD):
            raise TypeError(f"Expected a VAD instance, got {type(vad).__name__}")
        return StreamAdapter(stt=self, vad=vad)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Recognize speech from audio buffer.

        This method makes a single HTTP request per invocation. Retry logic is
        handled by the base class ``recognize()`` which wraps this method in a
        retry loop and catches ``APIError`` subclasses.

        Empty results are buffered and prepended to the next call so that
        short hesitant speech fragments are not silently dropped.
        """
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        # Merge audio frames from the buffer
        pcm_parts: list[bytes] = []
        sample_rate = self._sample_rate
        num_channels = 1
        frames = [buffer] if not isinstance(buffer, list) else buffer
        for frame in frames:
            pcm_parts.append(bytes(frame.data))
            sample_rate = frame.sample_rate
            num_channels = frame.num_channels
        segment_pcm = b"".join(pcm_parts)

        # Auto-clear stale pending buffer if too much time elapsed
        now = time.monotonic()
        if self._pending_pcm and self._last_recognize_time > 0:
            idle_gap = now - self._last_recognize_time
            if idle_gap > self._pending_idle_timeout:
                logger.debug(
                    "[%s] Clearing stale pending buffer (%.1fs idle)",
                    request_id,
                    idle_gap,
                )
                self._pending_pcm = b""
                self._pending_empty_count = 0
        self._last_recognize_time = now

        # Prepend any buffered PCM from previous empty segments
        if self._pending_pcm:
            logger.info(
                "[%s] Prepending %d bytes pending PCM to %d bytes new segment",
                request_id,
                len(self._pending_pcm),
                len(segment_pcm),
            )
            pcm_data = self._pending_pcm + segment_pcm
        else:
            pcm_data = segment_pcm

        if len(pcm_data) == 0:
            logger.warning("[%s] Empty audio buffer received, skipping", request_id)
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[],
            )

        lang = language if isinstance(language, str) else self._language
        logger.info(
            "[%s] STT request: language=%s, audio=%d bytes",
            request_id,
            lang,
            len(pcm_data),
        )

        wav_data = convert_pcm_to_wav(
            pcm_data,
            sample_rate=sample_rate,
            channels=num_channels,
            bits_per_sample=16,
        )

        params = {
            "language": lang,
            "enable_segments": "false",
            "enable_refinement": "false",
        }
        headers: dict[str, str] = {}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        files = {
            "audio_file": ("audio.wav", io.BytesIO(wav_data), "audio/wav"),
        }

        try:
            response = await self._client.post(
                self._transcribe_url,
                files=files,
                params=params,
                headers=headers,
            )
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"STT request timed out: {e}") from e
        except httpx.NetworkError as e:
            raise APIConnectionError(f"STT network error: {e}") from e
        except Exception as e:
            raise APIConnectionError(f"STT connection error: {e}") from e

        if response.status_code != 200:
            error_text = response.text
            raise APIStatusError(
                f"STT service error {response.status_code}: {error_text}",
                status_code=response.status_code,
                request_id=request_id,
                body=error_text,
            )

        result = response.json()
        raw_text = result.get("transcription", "")
        text = apply_normalization_rules(raw_text, self._normalization_rules)
        confidence = result.get("confidence", 1.0)
        latency = time.monotonic() - start_time

        # --- Frame accumulation logic ---
        # Compute duration of the current segment (not including pending)
        bytes_per_sample = 2 * num_channels  # 16-bit PCM
        segment_duration = (
            len(segment_pcm) / (sample_rate * bytes_per_sample)
            if sample_rate and bytes_per_sample
            else 0.0
        )
        pending_duration = (
            len(self._pending_pcm) / (sample_rate * bytes_per_sample)
            if self._pending_pcm
            else 0.0
        )

        if not text.strip():
            # Empty result — decide whether to buffer or discard
            self._pending_empty_count += 1
            total_pending_duration = pending_duration + segment_duration

            if (
                self._pending_empty_count <= self._max_pending_segments
                and total_pending_duration <= self._max_pending_duration
            ):
                # Buffer this segment's PCM for the next call
                self._pending_pcm = pcm_data  # includes already-prepended pending
                self._pending_sample_rate = sample_rate
                self._pending_num_channels = num_channels
                logger.info(
                    "[%s] STT empty → buffered (count=%d, duration=%.1fs, latency=%.3fs)",
                    request_id,
                    self._pending_empty_count,
                    total_pending_duration,
                    latency,
                )
            else:
                # Safety limit reached — discard buffer
                logger.info(
                    "[%s] STT empty → discarded pending buffer "
                    "(count=%d, duration=%.1fs, latency=%.3fs)",
                    request_id,
                    self._pending_empty_count,
                    total_pending_duration,
                    latency,
                )
                self._pending_pcm = b""
                self._pending_empty_count = 0

            # Return empty so StreamAdapter skips this segment as usual
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text="",
                        language=LanguageCode(lang),
                        confidence=0.0,
                    )
                ],
            )

        # Got real text — clear pending buffer
        had_pending = self._pending_empty_count > 0
        self._pending_pcm = b""
        self._pending_empty_count = 0

        logger.info(
            "[%s] STT completed: text='%s', confidence=%.3f, latency=%.3fs%s",
            request_id,
            text[:80],
            confidence,
            latency,
            f" (included {pending_duration:.1f}s pending audio)" if had_pending else "",
        )

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=text,
                    language=LanguageCode(lang),
                    confidence=confidence,
                )
            ],
        )
