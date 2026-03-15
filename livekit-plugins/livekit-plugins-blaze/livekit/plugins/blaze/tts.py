"""
Blaze TTS Plugin for LiveKit Voice Agent

Text-to-Speech plugin that interfaces with Blaze's TTS service.

API Endpoint: POST /v1/tts/realtime
Input: FormData with query, language, audio_format=pcm, speaker_id, normalization, model
Output: Streaming PCM audio chunks
"""

from __future__ import annotations

import time
import uuid

import httpx

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
)

from ._config import BlazeConfig
from ._utils import apply_normalization_rules
from .log import logger


class TTS(tts.TTS):
    """
    Blaze Text-to-Speech Plugin.

    Converts text to speech using Blaze's TTS service with streaming audio.

    Args:
        api_url: Base URL for the TTS service. If not provided,
                 reads from BLAZE_API_URL environment.
        language: Language code for synthesis (default: "vi").
        speaker_id: Speaker voice identifier (default: "default").
        auth_token: Bearer token for authentication. If not provided,
                    reads from BLAZE_API_TOKEN environment.
        model: TTS model to use (default: "v1_5_pro").
        sample_rate: Output audio sample rate (default: 24000).
        normalization_rules: Dict for text preprocessing before synthesis.
        timeout: Request timeout in seconds (default: 60.0).
        config: Optional BlazeConfig for centralized configuration.

    Example:
        >>> from livekit.plugins import blaze
        >>>
        >>> # Simple usage
        >>> tts = blaze.TTS(speaker_id="speaker-1")
        >>>
        >>> # Full configuration
        >>> tts = blaze.TTS(
        ...     api_url="https://api.blaze.vn",
        ...     speaker_id="custom-voice",
        ...     model="v2_pro",
        ...     normalization_rules={"API": "A P I"}
        ... )
        >>>
        >>> # Using shared config
        >>> config = blaze.BlazeConfig(api_url="https://api.blaze.vn")
        >>> tts = blaze.TTS(config=config, speaker_id="voice-1")
    """

    def __init__(
        self,
        *,
        api_url: str | None = None,
        language: str = "vi",
        speaker_id: str = "default",
        auth_token: str | None = None,
        model: str = "v1_5_pro",
        sample_rate: int = 24000,
        normalization_rules: dict[str, str] | None = None,
        timeout: float | None = None,
        config: BlazeConfig | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        # Load configuration
        self._config = config or BlazeConfig()

        # Resolve settings
        self._api_url = api_url or self._config.api_url
        self._language = language
        self._speaker_id = speaker_id
        self._auth_token = auth_token or self._config.api_token
        self._model = model
        self._sample_rate = sample_rate
        self._timeout = timeout if timeout is not None else self._config.tts_timeout
        self._normalization_rules = normalization_rules
        self._chunk_size = 8192

        # Build TTS URL
        self._tts_url = f"{self._api_url}/v1/tts/realtime"

        # Shared HTTP client for connection pooling
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout, connect=5.0))

        logger.info(
            f"BlazeTTS initialized: url={self._api_url}, "
            f"speaker={self._speaker_id}, language={self._language}"
        )

    @property
    def provider(self) -> str:
        """Returns the provider name."""
        return "Blaze"

    @property
    def model(self) -> str:
        """Returns the current TTS model."""
        return self._model

    async def aclose(self) -> None:
        """Close the shared HTTP client and release connections."""
        await self._client.aclose()
        await super().aclose()

    def update_options(
        self,
        *,
        speaker_id: str | None = None,
        model: str | None = None,
        language: str | None = None,
        auth_token: str | None = None,
        normalization_rules: dict[str, str] | None = None,
    ) -> None:
        """
        Update TTS options at runtime.

        Args:
            speaker_id: New speaker voice ID
            model: New TTS model
            language: New language code
            auth_token: New authentication token
            normalization_rules: New normalization rules
        """
        if speaker_id is not None:
            self._speaker_id = speaker_id
        if model is not None:
            self._model = model
        if language is not None:
            self._language = language
        if auth_token is not None:
            self._auth_token = auth_token
        if normalization_rules is not None:
            self._normalization_rules = normalization_rules

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            conn_options: API connection options (retry, timeout)

        Returns:
            ChunkedStream that yields audio frames
        """
        return _TTSStream(self, text, conn_options=conn_options)


class _TTSStream(tts.ChunkedStream):
    """Streaming TTS implementation for Blaze TTS."""

    def __init__(
        self,
        tts_instance: TTS,
        text: str,
        *,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts_instance, input_text=text, conn_options=conn_options)
        self._blaze_tts = tts_instance
        self._text = text

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Execute TTS synthesis and emit audio chunks.

        This method makes a single HTTP request. Retry logic is handled by the
        base class ``_main_task()``, which creates a fresh ``AudioEmitter`` for
        each attempt — preventing duplicate/garbled audio on mid-stream errors.

        Raises:
            APIStatusError: On non-200 HTTP responses.
            APITimeoutError: On request timeouts.
            APIConnectionError: On network errors.
        """
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        # Apply normalization to input text
        normalized_text = apply_normalization_rules(
            self._text, self._blaze_tts._normalization_rules
        )

        if not normalized_text.strip():
            logger.warning("[%s] Empty text after normalization, skipping TTS", request_id)

            # Base class always calls output_emitter.end_input(); initialize first
            # so empty-input flows complete gracefully without runtime errors.
            output_emitter.initialize(
                request_id=request_id,
                sample_rate=self._blaze_tts._sample_rate,
                num_channels=1,
                mime_type="audio/pcm",
            )
            return

        preview = normalized_text[:50] + ("..." if len(normalized_text) > 50 else "")
        logger.info(
            "[%s] TTS request: speaker=%s, text='%s'",
            request_id,
            self._blaze_tts._speaker_id,
            preview,
        )

        # Initialize the audio emitter (stream=False → single segment)
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._blaze_tts._sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        # Prepare multipart form data (httpx: (None, value) = non-file field)
        form_data = {
            "query": (None, normalized_text),
            "language": (None, self._blaze_tts._language),
            "audio_format": (None, "pcm"),
            "speaker_id": (None, self._blaze_tts._speaker_id),
            "normalization": (None, "no"),
            "model": (None, self._blaze_tts._model),
        }

        # Prepare headers
        headers: dict[str, str] = {}
        if self._blaze_tts._auth_token:
            headers["Authorization"] = f"Bearer {self._blaze_tts._auth_token}"

        try:
            async with self._blaze_tts._client.stream(
                "POST",
                self._blaze_tts._tts_url,
                files=form_data,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_text = (await response.aread()).decode(errors="replace")
                    raise APIStatusError(
                        f"TTS service error {response.status_code}: {error_text}",
                        status_code=response.status_code,
                        request_id=request_id,
                        body=error_text,
                    )

                # Stream audio chunks
                async for chunk in response.aiter_bytes(chunk_size=self._blaze_tts._chunk_size):
                    if chunk:
                        output_emitter.push(chunk)

                # Flush buffered audio so finalization follows framework pattern.
                output_emitter.flush()

        except httpx.TimeoutException as e:
            raise APITimeoutError(f"TTS request timed out: {e}") from e
        except httpx.NetworkError as e:
            raise APIConnectionError(f"TTS network error: {e}") from e

        latency = time.monotonic() - start_time
        logger.info(
            "[%s] TTS completed: text='%s', latency=%.3fs",
            request_id,
            preview,
            latency,
        )
