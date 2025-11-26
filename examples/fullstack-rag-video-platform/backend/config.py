"""
Configuration for RAG Video Platform
Centralized configuration management with environment variable support.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration settings for the RAG video platform."""

    # LiveKit Configuration
    livekit_url: str = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    livekit_api_key: str = os.getenv("LIVEKIT_API_KEY", "devkey")
    livekit_api_secret: str = os.getenv("LIVEKIT_API_SECRET", "secret")

    # LLM Configuration
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4-turbo")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # STT Configuration
    stt_provider: str = os.getenv("STT_PROVIDER", "deepgram")
    stt_language: str = os.getenv("STT_LANGUAGE", "en")

    # TTS Configuration
    tts_provider: str = os.getenv("TTS_PROVIDER", "elevenlabs")
    tts_voice: str = os.getenv("TTS_VOICE", "Rachel")

    # RAG Configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k: int = int(os.getenv("TOP_K", "5"))

    # Vector Database
    vector_db_url: Optional[str] = os.getenv("VECTOR_DB_URL", None)
    vector_db_type: str = os.getenv("VECTOR_DB_TYPE", "local")  # local, qdrant, pinecone

    # Memory Configuration
    memory_db_path: str = os.getenv("MEMORY_DB_PATH", "./data/memory.db")
    memory_window: int = int(os.getenv("MEMORY_WINDOW", "10"))
    long_term_memory: bool = os.getenv("LONG_TERM_MEMORY", "true").lower() == "true"

    # Video Configuration
    enable_video: bool = os.getenv("ENABLE_VIDEO", "true").lower() == "true"
    avatar_provider: str = os.getenv("AVATAR_PROVIDER", "simli")
    video_fps: int = int(os.getenv("VIDEO_FPS", "30"))
    video_quality: str = os.getenv("VIDEO_QUALITY", "high")  # low, medium, high

    # Performance Configuration
    max_concurrent_sessions: int = int(os.getenv("MAX_CONCURRENT_SESSIONS", "100"))
    enable_caching: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))  # seconds

    # Monitoring Configuration
    enable_metrics: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    enable_telemetry: bool = os.getenv("ENABLE_TELEMETRY", "true").lower() == "true"
    metrics_port: int = int(os.getenv("METRICS_PORT", "9090"))

    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_cors_origins: str = os.getenv("API_CORS_ORIGINS", "*")

    # Storage Configuration
    storage_path: str = os.getenv("STORAGE_PATH", "./storage")
    upload_max_size: int = int(os.getenv("UPLOAD_MAX_SIZE", "52428800"))  # 50MB in bytes
    allowed_file_types: str = os.getenv(
        "ALLOWED_FILE_TYPES", ".pdf,.txt,.md,.docx,.pptx"
    )

    # Security Configuration
    enable_auth: bool = os.getenv("ENABLE_AUTH", "false").lower() == "true"
    jwt_secret: str = os.getenv("JWT_SECRET", "change-this-secret")
    jwt_expiry: int = int(os.getenv("JWT_EXPIRY", "86400"))  # 24 hours

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration values."""
        # Validate LLM provider
        valid_llm_providers = ["openai", "anthropic", "google", "groq"]
        if self.llm_provider not in valid_llm_providers:
            raise ValueError(
                f"Invalid LLM provider: {self.llm_provider}. "
                f"Must be one of {valid_llm_providers}"
            )

        # Validate STT provider
        valid_stt_providers = ["deepgram", "assemblyai", "google", "azure"]
        if self.stt_provider not in valid_stt_providers:
            raise ValueError(
                f"Invalid STT provider: {self.stt_provider}. "
                f"Must be one of {valid_stt_providers}"
            )

        # Validate TTS provider
        valid_tts_providers = ["elevenlabs", "openai", "cartesia", "azure"]
        if self.tts_provider not in valid_tts_providers:
            raise ValueError(
                f"Invalid TTS provider: {self.tts_provider}. "
                f"Must be one of {valid_tts_providers}"
            )

        # Validate chunk sizes
        if self.chunk_size < 100 or self.chunk_size > 2000:
            raise ValueError("Chunk size must be between 100 and 2000")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

        # Validate video settings
        if self.video_fps < 1 or self.video_fps > 60:
            raise ValueError("Video FPS must be between 1 and 60")

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "stt_provider": self.stt_provider,
            "tts_provider": self.tts_provider,
            "avatar_provider": self.avatar_provider,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "memory_window": self.memory_window,
            "enable_video": self.enable_video,
            "video_fps": self.video_fps,
        }


# Global config instance
config = Config()
