"""SpatialReal avatar plugin for LiveKit Agents.

This plugin provides integration with SpatialReal's avatar service for
lip-synced avatar rendering in LiveKit voice agents.

See https://docs.spatialreal.ai for more information.

Usage:
    from livekit.plugins.spatialreal import AvatarSession

    avatar = AvatarSession()
    await avatar.start(agent_session, room=ctx.room)
"""

from .avatar import AvatarSession, SpatialRealException
from .version import __version__

__all__ = [
    "AvatarSession",
    "SpatialRealException",
    "__version__",
]

# Try to register plugin if Plugin class is available (livekit-agents >= 1.3)
try:
    from livekit.agents import Plugin

    from .log import logger

    class SpatialRealPlugin(Plugin):
        def __init__(self) -> None:
            super().__init__(__name__, __version__, __package__, logger)

    Plugin.register_plugin(SpatialRealPlugin())
except (ImportError, AttributeError):
    # Plugin registration not available in older versions
    pass
