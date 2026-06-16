from __future__ import annotations

import base64
from pathlib import Path


def encode_audio_file(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    mime = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }.get(ext, "audio/wav")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def normalize_ref_audio(ref_audio: str | Path) -> str:
    if isinstance(ref_audio, Path):
        return encode_audio_file(ref_audio)
    value = str(ref_audio)
    if value.startswith("data:") or value.startswith("http://") or value.startswith("https://"):
        return value
    return encode_audio_file(Path(value))
