"""
Scraper Configuration
Centralized config with Pydantic validation.
"""

import os
from typing import Optional, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ConcurrencyConfig(BaseModel):
    """Concurrency settings"""
    max_workers: int = Field(default=10, ge=1, le=100)
    max_connections: int = Field(default=100, ge=1, le=1000)
    timeout: int = Field(default=30, ge=1, le=300)


class RateLimitConfig(BaseModel):
    """Rate limiting settings"""
    requests_per_second: float = Field(default=5.0, ge=0.1, le=100.0)
    delay_between_requests: float = Field(default=0.2, ge=0, le=10.0)
    respect_robots_txt: bool = True


class CredentialsConfig(BaseModel):
    """Credentials management"""
    vault_path: str = "./vault.enc"
    encryption_key_env: str = "VAULT_KEY"
    session_dir: str = "./sessions"


class LearningConfig(BaseModel):
    """Self-improvement settings"""
    enabled: bool = True
    min_samples: int = Field(default=10, ge=1)
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    pattern_db: str = "./patterns.db"


class LLMConfig(BaseModel):
    """LLM configuration (open-source)"""
    provider: str = "ollama"  # ollama, llamacpp, or local
    model: str = "llama3.2"
    base_url: str = "http://localhost:11434"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=32000)
    streaming: bool = True


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration"""
    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    device: str = "cpu"  # cpu or cuda


class VoiceConfig(BaseModel):
    """Voice settings (open-source)"""
    stt: str = "whisper"  # Speech-to-text
    stt_model: str = "base"  # tiny, base, small, medium, large
    tts: str = "coqui"  # Text-to-speech
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    livekit_url: str = "ws://localhost:7880"


class ProxyConfig(BaseModel):
    """Proxy settings"""
    enabled: bool = False
    proxies: List[str] = []
    rotation_strategy: str = "round-robin"  # round-robin, random, geo-based


class CacheConfig(BaseModel):
    """Caching settings"""
    enabled: bool = True
    ttl: int = 3600  # seconds
    max_size: int = 1000  # items
    backend: str = "memory"  # memory, redis, filesystem


class ScraperConfig(BaseSettings):
    """Main scraper configuration"""

    # Component configs
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    rate_limiting: RateLimitConfig = RateLimitConfig()
    credentials: CredentialsConfig = CredentialsConfig()
    learning: LearningConfig = LearningConfig()
    llm: LLMConfig = LLMConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    voice: VoiceConfig = VoiceConfig()
    proxy: ProxyConfig = ProxyConfig()
    cache: CacheConfig = CacheConfig()

    # Engine preferences
    primary_engine: str = "playwright"
    fallback_engine: str = "httpx"
    parser_engine: str = "selectolax"  # selectolax, beautifulsoup

    # Behavior
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    headless: bool = True
    stealth_mode: bool = False
    screenshot_on_error: bool = False

    # Storage
    storage_path: str = "./storage"
    log_level: str = "INFO"

    # MCP
    mcp_enabled: bool = True
    mcp_servers: List[str] = ["browser", "filesystem", "database"]

    class Config:
        env_prefix = "SCRAPER_"
        case_sensitive = False


# Global config instance
config = ScraperConfig()
