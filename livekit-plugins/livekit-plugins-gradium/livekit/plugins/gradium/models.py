from typing import Literal

TTSEncoding = Literal["pcm_s16le",]
STTEncoding = Literal["pcm_s16le",]
# Lowest level of the server's log messages a session asks for.
LogLevel = Literal["debug", "info", "warning", "error"]
