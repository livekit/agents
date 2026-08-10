"""Gandr TTS plugin for LiveKit Agents."""

from .tts import TTS, ChunkedStream
from .version import __version__

GandrTTS = TTS  # the name the standalone file used

__all__ = ["TTS", "GandrTTS", "ChunkedStream", "__version__"]
