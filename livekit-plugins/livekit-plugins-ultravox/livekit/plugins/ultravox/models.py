# String aliases to prevent type drift
Models = str
Voices = str

# Curated known sets for warnings (not authoritative)
AVAILABLE_MODELS: list[str] = [
    "fixie-ai/ultravox",
    "fixie-ai/ultravox-gemma3-27b-preview",
    "fixie-ai/ultravox-llama3.3-70b",
    "fixie-ai/ultravox-qwen3-32b-preview",
]

# Small curated voice set for common voices (server is source of truth)
AVAILABLE_VOICES: list[str] = [
    "Mark",
    "Jessica",
]

# Safe defaults that should always work
DEFAULT_MODEL: str = "fixie-ai/ultravox"
"""Default Ultravox model - guaranteed to be available."""

DEFAULT_VOICE: str = "Mark"
"""Default Ultravox voice - guaranteed to be available."""
