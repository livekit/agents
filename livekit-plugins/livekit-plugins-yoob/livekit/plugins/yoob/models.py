from typing import Literal

TTSModels = Literal["yoob-tts-1"]
STTModels = Literal["yoob-asr-1", "yoob-asr-2"]
TTSQuality = Literal["best", "fast"]

DEFAULT_BASE_URL = "https://tts-api.yoob.com/v1"
DEFAULT_VOICE = "w1243"
TTS_SAMPLE_RATE = 48000
STT_SAMPLE_RATE = 16000
