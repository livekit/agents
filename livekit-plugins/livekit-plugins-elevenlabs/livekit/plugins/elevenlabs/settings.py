from typing import Literal

TTSModels = Literal[
    "eleven_monolingual_v1",
    "eleven_multilingual_v1",
    "eleven_multilingual_v2",
    "eleven_turbo_v2",
]

OutputFormat = Literal[
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

Encoding = Literal[
    "mp3",
    "pcm",
]


def sample_rate_from_format(output_format: OutputFormat) -> int:
    split = output_format.split("_")
    return int(split[1])


def encoding_from_format(output_format: OutputFormat) -> Encoding:
    if output_format.startswith("mp3"):
        return "mp3"
    elif output_format.startswith("pcm"):
        return "pcm"

    raise ValueError(f"Unknown format: {output_format}")
