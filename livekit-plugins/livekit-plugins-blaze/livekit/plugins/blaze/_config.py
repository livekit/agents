"""
Blaze Configuration Module

Provides centralized configuration for Blaze services.
Configuration can be provided via environment variables with BLAZE_ prefix
or passed explicitly to the constructor.
"""

from __future__ import annotations

import os

_DEFAULT_API_URL = "https://api.blaze.vn"
_DEFAULT_STT_TIMEOUT = 30.0
_DEFAULT_TTS_TIMEOUT = 60.0
_DEFAULT_TTS_STREAM_TIMEOUT = 300.0
_DEFAULT_LLM_TIMEOUT = 60.0


class BlazeConfig:
    """
    Configuration for Blaze AI services.

    All services (STT, TTS, LLM) route through a single gateway URL.
    Service-specific configuration (language, speaker, etc.) comes from the
    voicebot ID and is passed as constructor arguments to each plugin.

    Environment Variables:
        BLAZE_API_URL: Base URL for Blaze API gateway
        BLAZE_API_TOKEN: Bearer token for API authentication
        BLAZE_STT_TIMEOUT: STT request timeout in seconds
        BLAZE_TTS_TIMEOUT: TTS per-request timeout in seconds
        BLAZE_TTS_STREAM_TIMEOUT: TTS streaming session timeout in seconds
        BLAZE_LLM_TIMEOUT: LLM request timeout in seconds

    Example:
        >>> from livekit.plugins.blaze import BlazeConfig
        >>>
        >>> # Load from environment variables
        >>> config = BlazeConfig()
        >>>
        >>> # Or provide explicit values
        >>> config = BlazeConfig(
        ...     api_url="https://api.blaze.vn",
        ...     api_token="my-token",
        ... )
    """

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_token: str | None = None,
        stt_timeout: float | None = None,
        tts_timeout: float | None = None,
        tts_stream_timeout: float | None = None,
        llm_timeout: float | None = None,
    ) -> None:
        self.api_url: str = api_url or os.environ.get("BLAZE_API_URL", _DEFAULT_API_URL)
        self.api_token: str = api_token or os.environ.get("BLAZE_API_TOKEN", "")
        self.stt_timeout: float = (
            stt_timeout
            if stt_timeout is not None
            else float(os.environ.get("BLAZE_STT_TIMEOUT", _DEFAULT_STT_TIMEOUT))
        )
        self.tts_timeout: float = (
            tts_timeout
            if tts_timeout is not None
            else float(os.environ.get("BLAZE_TTS_TIMEOUT", _DEFAULT_TTS_TIMEOUT))
        )
        # Separate timeout for the full TTS streaming session (WebSocket + all batches).
        # Streaming turns can span many text batches and require a longer timeout than
        # individual per-request timeouts.
        self.tts_stream_timeout: float = (
            tts_stream_timeout
            if tts_stream_timeout is not None
            else float(
                os.environ.get("BLAZE_TTS_STREAM_TIMEOUT", _DEFAULT_TTS_STREAM_TIMEOUT)
            )
        )
        self.llm_timeout: float = (
            llm_timeout
            if llm_timeout is not None
            else float(os.environ.get("BLAZE_LLM_TIMEOUT", _DEFAULT_LLM_TIMEOUT))
        )

