# Copyright 2026 Oruk AI
# SPDX-License-Identifier: Apache-2.0

"""Optional local Orukeet speech recognition for LiveKit Agents."""

import logging

from livekit.agents import Plugin

from ._model import download_model
from .stt import STT
from .version import __version__

__all__ = ["STT", "__version__"]


class OrukeetPlugin(Plugin):
    """Register the plugin and its normal CLI model-download hook."""

    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logging.getLogger(__name__))

    def download_files(self) -> None:
        """Download and verify weights for the agent's download-files command."""
        download_model()


Plugin.register_plugin(OrukeetPlugin())
