from typing import Literal

STTModels = Literal["transcribe-1"]

TTSModels = Literal["s1", "s2-pro"]

OutputFormat = Literal["wav", "pcm", "mp3", "opus"]

LatencyMode = Literal["normal", "balanced", "low"]
