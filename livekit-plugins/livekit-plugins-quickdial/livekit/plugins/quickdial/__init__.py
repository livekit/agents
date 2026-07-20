# Copyright 2026 Samay AI (Quickdial)
# Licensed under the Apache License, Version 2.0
"""Quickdial plugin for LiveKit Agents — cheap, fast, real-time TTS & STT.

    from livekit.plugins import quickdial

    session = AgentSession(
        tts=quickdial.TTS(voice="alba"),
        stt=quickdial.STT(language="en"),
        vad=silero.VAD.load(),
        llm=...,
    )
"""
from livekit.agents import Plugin

from .log import logger
from .stt import STT, SpeechStream
from .tts import TTS, ChunkedStream, SynthesizeStream
from .version import __version__

__all__ = [
    "TTS",
    "STT",
    "ChunkedStream",
    "SynthesizeStream",
    "SpeechStream",
    "__version__",
]


class QuickdialPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(QuickdialPlugin())

# hide internal submodules from `dir()` per LiveKit plugin convention
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}
for _n in NOT_IN_ALL:
    __pdoc__[_n] = False
