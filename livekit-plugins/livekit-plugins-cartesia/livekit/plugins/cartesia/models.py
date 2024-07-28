from typing import Literal

TTSEncoding = Literal[
    "pcm_s16le",
    # Not yet supported
    # "pcm_f32le",
    # "pcm_mulaw",
    # "pcm_alaw",
]


TTSModels = Literal["sonic-english", "sonic-multilingual"]
TTSLanguages = Literal["en", "es", "fr", "de", "pt", "zh", "ja"]
TTSDefaultVoiceId = "248be419-c632-4f23-adf1-5324ed7dbf1d"
