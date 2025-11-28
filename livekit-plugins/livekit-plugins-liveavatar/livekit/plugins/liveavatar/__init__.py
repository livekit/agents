"""LiveAvatar avatar plugin for LiveKit Agents

Provides LiveAvatar interactive avatar integration similar to Tavus.
"""

from .api import LiveAvatarException
from .avatar import AvatarSession

__all__ = [
    "LiveAvatarException",
    "AvatarSession",
]
