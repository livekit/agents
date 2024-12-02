from typing import Literal

TTSEngines = Literal[
    "PlayHT2.0",
    "PlayHT1.0",
    "PlayHT2.0-turbo",
    "Play3.0-mini",
]

TTSEncoding = Literal[
    "mp3_22050_32",
    "mp3_44100_32",
    "mp3_44100_64",
    "mp3_44100_96",
    "mp3_44100_128",
    "mp3_44100_192",
    "pcm_16000",
    "pcm_22050",
    "pcm_44100",
]
