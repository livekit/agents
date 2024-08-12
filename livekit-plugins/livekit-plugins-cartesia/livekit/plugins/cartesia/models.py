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
TTSDefaultVoiceId = "b7d50908-b17c-442d-ad8d-810c63997ed9"
