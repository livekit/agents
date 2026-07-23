from typing import Literal

# Qwen3 TTS Model
TTSModel = Literal["qwen3-tts-flash-realtime"]

# Available voices for Qwen3 TTS
# See: https://www.alibabacloud.com/help/en/model-studio/qwen-tts
TTSVoice = Literal[
    # Female voices
    "Kiki",
    "Cherry",
    "Jennifer",
    # Male voices
    "Rocky",
    "Ethan",
    "Ryan",
    # Regional variants
    "Sichuan-Sunny",
    "Shanghai-Jada",
    "Beijing-Yunxi",
    # Cantonese voices
    "Cantonese_ProfessionalHost",
]

# Supported languages (lowercase as required by API)
TTSLanguage = Literal[
    "auto",
    "chinese",
    "english",
    "german",
    "italian",
    "portuguese",
    "spanish",
    "japanese",
    "korean",
    "french",
    "russian",
]

# Session modes
TTSMode = Literal[
    "server_commit",  # Auto-triggers synthesis after pause (recommended for streaming)
    "commit",  # Manual trigger via commit_text_buffer
]

# Audio format (only PCM supported for realtime)
TTSAudioFormat = Literal["pcm"]

# Sample rate
TTSSampleRate = Literal[24000]

# Default values
DEFAULT_MODEL: TTSModel = "qwen3-tts-flash-realtime"
DEFAULT_VOICE: TTSVoice = "Kiki"
DEFAULT_LANGUAGE: TTSLanguage = "auto"
DEFAULT_MODE: TTSMode = "server_commit"
DEFAULT_SAMPLE_RATE: TTSSampleRate = 24000

# Base URLs
DEFAULT_BASE_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
INTL_BASE_URL = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
