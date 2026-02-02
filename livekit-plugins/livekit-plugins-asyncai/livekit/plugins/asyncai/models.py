from typing import Literal

TTSEncoding = Literal["pcm_s16le", "pcm_f32le", "pcm_mulaw"]

TTSModels = Literal["asyncflow_multilingual_v1.0", "asyncflow_v2.0"]
TTSLanguages = Literal["en", "de", "es", "fr", "it"]
TTSDefaultVoiceId = "e0f39dc4-f691-4e78-bba5-5c636692cc04"
