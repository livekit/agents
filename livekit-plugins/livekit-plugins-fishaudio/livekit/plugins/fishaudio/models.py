from typing import Literal

TTSModels = Literal["s1", "s2-pro", "s2.1-pro", "s2.1-pro-free"]

OutputFormat = Literal["wav", "pcm", "mp3", "opus"]

LatencyMode = Literal["normal", "balanced", "low"]

MP3Bitrate = Literal[64, 128, 192]
"""MP3 bitrate in kbps."""

OpusBitrate = Literal[-1000, 24000, 32000, 48000, 64000]
"""Opus bitrate in bps. ``-1000`` selects Fish Audio's automatic bitrate."""
