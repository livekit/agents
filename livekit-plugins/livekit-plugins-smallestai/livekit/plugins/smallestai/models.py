from typing import Literal

TTSModels = Literal[
    "lightning",
    "lightning-large",
    "lightning-v2",
    "lightning-v3.1",
]

TTSEncoding = Literal[
    "pcm",
    "mp3",
    "wav",
    "mulaw",
]
