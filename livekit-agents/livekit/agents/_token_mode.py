"""Token mode: the process holds no API secret, only a worker token minted for it.

Token mode is on when ``LIVEKIT_TOKEN`` is set and neither ``LIVEKIT_API_KEY`` nor
``LIVEKIT_API_SECRET`` is. The worker uses the token to register, and only for that. A
job gets the tokens it needs from the server (CreateJobTokenRequest): the ones for
inference and telemetry are kept current in the background while the job runs.
"""

from __future__ import annotations

import json
import os

# the claims of the tokens a job keeps current
INFERENCE_CLAIMS = json.dumps({"identity": "agent", "inference": {"perform": True}})
OBSERVABILITY_CLAIMS = json.dumps({"observability": {"write": True}})


def token_mode_token() -> str | None:
    """The worker token when this process runs in token mode, else None."""
    if os.environ.get("LIVEKIT_API_KEY") or os.environ.get("LIVEKIT_API_SECRET"):
        return None

    return os.environ.get("LIVEKIT_TOKEN") or None
