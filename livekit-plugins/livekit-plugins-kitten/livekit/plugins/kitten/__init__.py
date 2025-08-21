"""Kitten plugin for LiveKit Agents"""

from .tts import TTS, ChunkedStream
from .version import __version__

__all__ = ["TTS", "ChunkedStream", "KittenPlugin", "__version__"]

import os

from livekit.agents import Plugin

from .model import (
    HG_MODEL,
    ONNX_FILENAME,
    VOICES_FILENAME,
)


class KittenPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__)

    def download_files(self) -> None:
        from huggingface_hub import hf_hub_download

        repo_id = os.getenv("KITTENTTS_REPO_ID", HG_MODEL)

        hf_hub_download(repo_id=repo_id, filename=ONNX_FILENAME)
        hf_hub_download(repo_id=repo_id, filename=VOICES_FILENAME)


Plugin.register_plugin(KittenPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
