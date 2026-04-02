from typing import Literal

TTSModels = Literal[
    "lightning-v2",
    "lightning-v3.1",
]

TTSEncoding = Literal[
    "pcm",
    "mp3",
    "wav",
    "mulaw",
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
