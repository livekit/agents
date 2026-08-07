from typing import Literal

TTSEncoding = Literal["pcm_s16le", "pcm_f32le", "pcm_mulaw"]

TTSModels = Literal["async_flash_v1.0"]
TTSLanguages = Literal[
    "en", "de", "es", "fr", "it", "pt", "ar", "ru", "ro", "ja", "he", "hy", "tr", "hi", "zh"
]
TTSDefaultVoiceId = "e0f39dc4-f691-4e78-bba5-5c636692cc04"
