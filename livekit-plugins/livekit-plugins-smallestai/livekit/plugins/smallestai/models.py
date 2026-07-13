from typing import Literal

TTSModels = Literal[
    "lightning_v3.1",
    "lightning_v3.1_pro",
]

TTSEncoding = Literal[
    "pcm",
    "mp3",
    "wav",
    "ulaw",
    "alaw",
]

STTModels = Literal["pulse"]

STTEncoding = Literal[
    "linear16",
    "linear32",
    "alaw",
    "mulaw",
    "opus",
    "ogg_opus",
]
