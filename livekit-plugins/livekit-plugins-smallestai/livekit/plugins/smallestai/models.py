from typing import Literal

TTSModels = Literal[
    "lightning",
    "lightning-large",
    "lightning-v2",
]

TTSEncoding = Literal[
    "pcm",
    "mp3",
    "wav",
    "mulaw",
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

STTSampleRates = Literal[
    8000,
    16000,
    22050,
    24000,
    44100,
    48000,
]
