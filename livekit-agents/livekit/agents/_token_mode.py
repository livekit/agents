"""Token mode: the process holds no API secret, only a worker token minted for it.

Token mode is on when ``LIVEKIT_TOKEN`` is set and neither ``LIVEKIT_API_KEY`` nor
``LIVEKIT_API_SECRET`` is. The worker uses the token to register and exports it to
``LIVEKIT_TOKEN`` for the jobs it runs. Code in a job reads it only through
:func:`token_mode_token`, so a refreshed token can later be put in this one place.
"""

from __future__ import annotations

import os


def token_mode_token() -> str | None:
    """The worker token when this process runs in token mode, else None."""
    if os.environ.get("LIVEKIT_API_KEY") or os.environ.get("LIVEKIT_API_SECRET"):
        return None

    return os.environ.get("LIVEKIT_TOKEN") or None


def require_api_secret(feature: str) -> None:
    """Raises when ``feature``, which signs tokens with the API secret, runs in token mode."""
    if token_mode_token() is not None:
        raise RuntimeError(
            f"{feature} signs tokens with LIVEKIT_API_KEY and LIVEKIT_API_SECRET, and is not "
            "supported in token mode (LIVEKIT_TOKEN without an API key and secret)"
        )
