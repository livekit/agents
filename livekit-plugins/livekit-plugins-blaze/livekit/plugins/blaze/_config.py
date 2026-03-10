"""
Blaze Configuration Module

Provides centralized configuration for Blaze services using Pydantic settings.
Configuration can be provided via environment variables with BLAZE_ prefix.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class BlazeConfig(BaseSettings):
    """
    Configuration for Blaze AI services.

    All services (STT, TTS, LLM) route through a single gateway URL.
    Service-specific configuration (language, speaker, etc.) comes from the
    voicebot ID and is passed as constructor arguments to each plugin.

    Environment Variables:
        BLAZE_API_URL: Base URL for Blaze API gateway
        BLAZE_API_TOKEN: Bearer token for API authentication
        BLAZE_STT_TIMEOUT: STT request timeout in seconds
        BLAZE_TTS_TIMEOUT: TTS request timeout in seconds
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
        ...     auth_token="my-token",
        ... )
    """

    # Service URL
    api_url: str = Field(
        default="https://api.blaze.vn",
        description="Base URL for Blaze API gateway",
    )

    # Authentication
    auth_token: str = Field(
        default="",
        description="Bearer token for API authentication",
    )

    # Timeouts
    stt_timeout: float = Field(
        default=30.0,
        description="STT request timeout in seconds",
    )
    tts_timeout: float = Field(
        default=60.0,
        description="TTS request timeout in seconds",
    )
    llm_timeout: float = Field(
        default=60.0,
        description="LLM request timeout in seconds",
    )

    model_config = {
        "env_prefix": "BLAZE_",
        "env_file": ".env",
        "extra": "ignore",
    }
