from typing import Literal

TTSBackends = Literal["speech-1.5", "speech-1.6", "agent-x0", "s1", "s1-mini"]

OutputFormat = Literal["wav", "pcm", "mp3"]

LatencyMode = Literal["normal", "balanced"]
