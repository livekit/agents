# SPDX-FileCopyrightText: 2026 OpenSpline contributors
# SPDX-License-Identifier: Apache-2.0
"""OpenSpline avatar plugin for LiveKit Agents."""

import logging

from livekit.agents import Plugin

from .avatar import AvatarSession
from .version import __version__

__all__ = ["AvatarSession", "__version__"]


class OpensplinePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logging.getLogger(__name__))


Plugin.register_plugin(OpensplinePlugin())
