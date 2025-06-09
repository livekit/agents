from typing import Literal

TTSModels = Literal[
    "eleven_monolingual_v1",
    "eleven_multilingual_v1",
    "eleven_multilingual_v2",
    "eleven_turbo_v2",
    "eleven_turbo_v2_5",
    "eleven_flash_v2_5",
    "eleven_flash_v2",
    "eleven_v3",
]

TTSEncoding = Literal[
    "mp3_22050_32",
    "mp3_44100",
    "mp3_44100_32",
    "mp3_44100_64",
    "mp3_44100_96",
    "mp3_44100_128",
    "mp3_44100_192",
]
