from __future__ import annotations


class HedraException(Exception):
    """Exception for Hedra errors"""


class AvatarSession:
    """Hedra realtime avatar service has been disabled."""

    def __init__(self, **kwargs: object) -> None:
        raise HedraException(
            "The Hedra realtime avatar service has been disabled. "
            "Please use another avatar provider, such as Anam (livekit-plugins-anam)."
        )
