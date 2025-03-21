from enum import Enum


class OutputFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"


class Precision(str, Enum):
    PCM_16 = "PCM_16"
