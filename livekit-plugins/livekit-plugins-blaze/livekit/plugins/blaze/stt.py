"""
Blaze STT Plugin for LiveKit Voice Agent

Speech-to-Text plugin that interfaces with Blaze's transcription service.

API Endpoint: POST /v1/stt/transcribe
Input: WAV audio file, language code
Output: { "transcription": str, "confidence": float, "is_final": bool, "language": str }
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
from livekit.agents.utils import AudioBuffer

from ._config import BlazeConfig
from ._utils import apply_normalization_rules, convert_pcm_to_wav
from .log import logger


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
                    reads from BLAZE_API_TOKEN environment.
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
                streaming=False,  # Blaze STT doesn't support streaming yet
                interim_results=False,
            )
        )

        # Load configuration
        self._config = config or BlazeConfig()

        # Resolve settings with priority: explicit args > config > defaults
        self._api_url = api_url or self._config.api_url
        self._language = language
        self._auth_token = auth_token or self._config.api_token
        self._sample_rate = sample_rate
        self._timeout = timeout if timeout is not None else self._config.stt_timeout
        self._normalization_rules = normalization_rules

        # Build transcribe URL
        self._transcribe_url = f"{self._api_url}/v1/stt/transcribe"

        # Shared HTTP client for connection pooling
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout, connect=5.0))

        logger.info(f"BlazeSTT initialized: url={self._api_url}, language={self._language}")

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
        language: str | None = None,
        auth_token: str | None = None,
        normalization_rules: dict[str, str] | None = None,
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
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Recognize speech from audio buffer.

        This method makes a single HTTP request per invocation. Retry logic is
        handled by the base class ``recognize()`` which wraps this method in a
        retry loop and catches ``APIError`` subclasses.

        Args:
            buffer: Audio buffer containing PCM data
            language: Optional language override
            conn_options: API connection options (used by base class for retry)

        Returns:
            SpeechEvent with recognition results

        Raises:
            APIStatusError: On non-200 HTTP responses.
            APITimeoutError: On request timeouts.
            APIConnectionError: On network errors.
        """
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        # Merge audio frames from the buffer
        pcm_parts = []
        sample_rate = self._sample_rate
        num_channels = 1
        frames = [buffer] if not isinstance(buffer, list) else buffer
        for frame in frames:
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

        lang = language if isinstance(language, str) else self._language
        logger.info(
            "[%s] STT request: language=%s, audio=%d bytes",
            request_id,
            lang,
            len(pcm_data),
        )

        # Convert PCM to WAV using frame metadata
        wav_data = convert_pcm_to_wav(
            pcm_data,
            sample_rate=sample_rate,
            channels=num_channels,
            bits_per_sample=16,
        )

        # Single-attempt request — base class handles retries.
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

        if response.status_code != 200:
            error_text = response.text
            raise APIStatusError(
                f"STT service error {response.status_code}: {error_text}",
                status_code=response.status_code,
                request_id=request_id,
                body=error_text,
            )

        result = response.json()

        # Extract transcription
        raw_text = result.get("transcription", "")
        text = apply_normalization_rules(raw_text, self._normalization_rules)
        confidence = result.get("confidence", 1.0)
        latency = time.monotonic() - start_time

        logger.info(
            "[%s] STT completed: text='%s', confidence=%.3f, latency=%.3fs",
            request_id,
            text[:80],
            confidence,
            latency,
        )

        # Return speech event
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

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        """Create a speech recognition stream.

        Note: Blaze STT currently only supports batch recognition.
        This method raises NotImplementedError.

        Raises:
            NotImplementedError: Streaming is not supported by Blaze STT.
        """
        raise NotImplementedError("Streaming STT is not supported by Blaze STT")
