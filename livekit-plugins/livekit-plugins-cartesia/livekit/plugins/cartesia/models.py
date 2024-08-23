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
TTSDefaultVoiceId = "c2ac25f9-ecc4-4f56-9095-651354df60c0"
