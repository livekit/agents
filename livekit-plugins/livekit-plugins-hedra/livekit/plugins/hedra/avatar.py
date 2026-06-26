from __future__ import annotations


class HedraException(Exception):
    """Exception for Hedra errors"""


class AvatarSession:
    """Hedra realtime avatar service has been disabled."""

    def __init__(self, **kwargs: object) -> None:
        raise HedraException(
            "The Hedra realtime avatar service has been disabled. This plugin no longer functions. "
            "Please browse our other avatar integrations instead at https://docs.livekit.io/agents/models/avatar/."
        )
