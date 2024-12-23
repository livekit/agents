from typing import Literal

from pyht.client import Format

TTSModel = Literal["Play3.0-mini-ws", "PlayDialog-ws", "Play3.0-mini", "PlayDialog"]
FORMAT = Literal["raw", "mp3", "wav", "ogg", "flac", "mulaw", "pcm"]
format_mapping = {
    "raw": Format.FORMAT_RAW,
    "mp3": Format.FORMAT_MP3,
    "wav": Format.FORMAT_WAV,
    "ogg": Format.FORMAT_OGG,
    "flac": Format.FORMAT_FLAC,
    "mulaw": Format.FORMAT_MULAW,
    "pcm": Format.FORMAT_PCM,
}
