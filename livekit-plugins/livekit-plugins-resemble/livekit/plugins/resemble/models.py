from enum import Enum


class OutputFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"


class Precision(str, Enum):
    PCM_32 = "PCM_32"
    PCM_24 = "PCM_24"
    PCM_16 = "PCM_16"
    MULAW = "MULAW" 