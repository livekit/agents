from typing import Literal

TTSEncoding = Literal[
    "pcm_s16le",
    # Not yet supported
    # "pcm_f32le",
    # "pcm_mulaw",
    # "pcm_alaw",
]


TTSModels = Literal["sonic-english", "sonic-multilingual"]

# Customer Support Lady
TTSDefaultVoiceID: str = "829ccd10-f8b3-43cd-b8a0-4aeaa81f3b30"
