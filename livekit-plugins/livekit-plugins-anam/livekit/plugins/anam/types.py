from dataclasses import dataclass


@dataclass
class PersonaConfig:
    """Configuration for Anam avatar persona"""

    name: str
    avatarId: str
    avatarModel: str | None = None


@dataclass
class SessionOptions:
    """Per-session output options forwarded to Anam's session-token API.

    Mirrors the ``sessionOptions`` field of the Anam session-token request.

    Attributes:
        video_width: Output video frame width in pixels. Provide together with
            ``video_height`` (both or neither). Omit to use the avatar model's
            default output size. Supported pairs are model-dependent and are
            validated by Anam; an unsupported pair is rejected with an HTTP 400
            rather than silently downgraded.
        video_height: Output video frame height in pixels. See ``video_width``.
    """

    video_width: int | None = None
    video_height: int | None = None
