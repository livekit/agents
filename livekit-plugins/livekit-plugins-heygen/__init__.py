"""HeyGen avatar plugin for LiveKit Agents

Provides HeyGen interactive avatar integration similar to Tavus.
"""

from .api import HeyGenException
from .avatar import AvatarSession

__all__ = [
    "HeyGenException",
    "AvatarSession",
]
