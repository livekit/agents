"""
Blaze STT Plugin for LiveKit Voice Agent

Speech-to-Text plugin that interfaces with Blaze's transcription service.

API Endpoint: POST /v1/stt/transcribe
Input: WAV audio file, language code
Output: { "transcription": str, "confidence": float, "is_final": bool, "language": str }
"""

from __future__ import annotations

import asyncio
import io
import random
import time
import uuid
from typing import Optional, Dict

import httpx
from livekit.agents import stt, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

from .log import logger
from ._config import BlazeConfig
from ._utils import convert_pcm_to_wav, apply_normalization_rules


class STT(stt.STT):
    """
    Blaze Speech-to-Text Plugin.

    Converts speech to text using Blaze's transcription service.
    This plugin supports batch recognition only (no streaming).

    Args:
        api_url: Base URL for the STT service. If not provided,
                 reads from BLAZE_API_URL environment.
        language: Language code for transcription (default: "vi").
        auth_token: Bearer token for authentication. If not provided,
                    reads from BLAZE_AUTH_TOKEN environment.
        sample_rate: Audio sample rate in Hz (default: 16000).
        normalization_rules: Dict mapping input strings to replacements
                            applied to transcription results.
        timeout: Request timeout in seconds (default: 30.0).
        config: Optional BlazeConfig for centralized configuration.

    Example:
        >>> from livekit.plugins import blaze
        >>>
        >>> # Simple usage with environment variables
        >>> stt = blaze.STT()
        >>>
        >>> # Explicit configuration
        >>> stt = blaze.STT(
        ...     api_url="https://api.blaze.vn",
        ...     language="en",
        ...     auth_token="my-token"
        ... )
        >>>
        >>> # Using shared config
        >>> config = blaze.BlazeConfig(api_url="https://api.blaze.vn")
        >>> stt = blaze.STT(config=config)
        >>>
        >>> # With normalization rules
        >>> stt = blaze.STT(
        ...     normalization_rules={"AI": "artificial intelligence"}
        ... )
    """

    def __init__(
        self,
        *,
        api_url: Optional[str] = None,
        language: str = "vi",
        auth_token: Optional[str] = None,
        sample_rate: int = 16000,
        normalization_rules: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        config: Optional[BlazeConfig] = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,  # Blaze STT doesn't support streaming yet
                interim_results=False,
            )
        )

        # Load configuration
        self._config = config or BlazeConfig()

        # Resolve settings with priority: explicit args > config > defaults
        self._api_url = api_url or self._config.api_url
        self._language = language
        self._auth_token = auth_token or self._config.auth_token
        self._sample_rate = sample_rate
        self._timeout = timeout or self._config.stt_timeout
        self._normalization_rules = normalization_rules

        # Build transcribe URL
        self._transcribe_url = f"{self._api_url}/v1/stt/transcribe"

        # Shared HTTP client for connection pooling
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout, connect=5.0)
        )

        logger.info(
            f"BlazeSTT initialized: url={self._api_url}, language={self._language}"
        )

    @property
    def provider(self) -> str:
        """Returns the provider name."""
        return "Blaze"

    @property
    def sample_rate(self) -> int:
        """Returns the configured sample rate."""
        return self._sample_rate

    async def aclose(self) -> None:
        """Close the shared HTTP client and release connections."""
        await self._client.aclose()
        await super().aclose()

    def update_options(
        self,
        *,
        language: Optional[str] = None,
        auth_token: Optional[str] = None,
        normalization_rules: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Update STT options at runtime.

        Args:
            language: New language code
            auth_token: New authentication token
            normalization_rules: New normalization rules
        """
        if language is not None:
            self._language = language
        if auth_token is not None:
            self._auth_token = auth_token
        if normalization_rules is not None:
            self._normalization_rules = normalization_rules

    async def _recognize_impl(
        self,
        buffer: stt.AudioBuffer,
        *,
        language: Optional[str] = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """
        Recognize speech from audio buffer.

        Args:
            buffer: Audio buffer containing PCM data
            language: Optional language override
            conn_options: API connection options (retry, timeout)

        Returns:
            SpeechEvent with recognition results
        """
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        try:
            # Merge audio frames from the buffer
            pcm_parts = []
            sample_rate = self._sample_rate
            num_channels = 1
            for frame in buffer:
                pcm_parts.append(bytes(frame.data))
                sample_rate = frame.sample_rate
                num_channels = frame.num_channels
            pcm_data = b"".join(pcm_parts)

            if len(pcm_data) == 0:
                logger.warning("[%s] Empty audio buffer received, skipping", request_id)
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[],
                )

            lang = language or self._language
            logger.info(
                "[%s] STT request: language=%s, audio=%d bytes",
                request_id, lang, len(pcm_data),
            )

            # Convert PCM to WAV using frame metadata
            wav_data = convert_pcm_to_wav(
                pcm_data,
                sample_rate=sample_rate,
                channels=num_channels,
                bits_per_sample=16,
            )

            # Prepare request
            files = {
                "audio_file": ("audio.wav", io.BytesIO(wav_data), "audio/wav"),
            }
            params = {
                "language": lang,
                "enable_segments": "false",
                "enable_refinement": "false",
            }
            headers: Dict[str, str] = {}
            if self._auth_token:
                headers["Authorization"] = f"Bearer {self._auth_token}"

            # Make request with retry logic (shared client for connection pooling)
            result: Dict = {}

            for attempt in range(conn_options.max_retry + 1):
                try:
                    response = await self._client.post(
                        self._transcribe_url,
                        files=files,
                        params=params,
                        headers=headers,
                    )

                    if response.status_code >= 500:
                        error_text = response.text
                        if attempt < conn_options.max_retry:
                            delay = conn_options.retry_interval * (2 ** attempt)
                            jitter = delay * 0.1 * random.random()
                            logger.warning(
                                f"[{request_id}] STT attempt {attempt + 1}/"
                                f"{conn_options.max_retry + 1} failed "
                                f"({response.status_code}). "
                                f"Retrying in {delay:.1f}s…"
                            )
                            await asyncio.sleep(delay + jitter)
                            continue
                        raise STTError(
                            f"STT service error: {response.status_code}",
                            status_code=response.status_code,
                        )

                    if response.status_code != 200:
                        error_text = response.text
                        logger.error(
                            f"[{request_id}] STT service error: "
                            f"{response.status_code} - {error_text}"
                        )
                        raise STTError(
                            f"STT service error: {response.status_code}",
                            status_code=response.status_code,
                        )

                    result = response.json()
                    break  # Success

                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    if attempt < conn_options.max_retry:
                        delay = conn_options.retry_interval * (2 ** attempt)
                        jitter = delay * 0.1 * random.random()
                        logger.warning(
                            f"[{request_id}] STT network error (attempt "
                            f"{attempt + 1}/{conn_options.max_retry + 1}): "
                            f"{e}. Retrying in {delay:.1f}s…"
                        )
                        await asyncio.sleep(delay + jitter)
                    else:
                        raise STTError(f"STT network error: {e}") from e

            # Extract transcription
            raw_text = result.get("transcription", "")
            text = apply_normalization_rules(raw_text, self._normalization_rules)
            confidence = result.get("confidence", 1.0)
            latency = time.monotonic() - start_time

            logger.info(
                f"[{request_id}] STT completed: text='{text[:80]}', "
                f"confidence={confidence:.3f}, latency={latency:.3f}s"
            )

            # Return speech event
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=text,
                        language=lang,
                        confidence=confidence,
                    )
                ],
            )

        except STTError:
            raise
        except Exception as e:
            latency = time.monotonic() - start_time
            logger.error(
                f"[{request_id}] STT recognition failed after {latency:.3f}s: {e}"
            )
            raise STTError(f"STT recognition failed: {str(e)}") from e

    def stream(
        self,
        *,
        language: Optional[str] = None,
    ) -> "stt.SpeechStream":
        """
        Create a speech recognition stream.

        Note: Blaze STT currently only supports batch recognition.
        This method raises NotImplementedError.

        Raises:
            NotImplementedError: Streaming is not supported by Blaze STT.
        """
        raise NotImplementedError("Streaming STT is not supported by Blaze STT")


class STTError(Exception):
    """Exception raised when STT service encounters an error."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
