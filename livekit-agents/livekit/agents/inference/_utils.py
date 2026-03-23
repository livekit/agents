from __future__ import annotations

import datetime
import os

from livekit import api

DEFAULT_INFERENCE_URL = "https://agent-gateway.livekit.cloud/v1"
STAGING_INFERENCE_URL = "https://agent-gateway.staging.livekit.cloud/v1"


def get_default_inference_url() -> str:
    """Get the default inference URL based on the environment.

    Priority:
    1. LIVEKIT_INFERENCE_URL if set
    2. If LIVEKIT_URL contains '.staging.livekit.cloud', use staging gateway
    3. Otherwise, use production gateway
    """
    inference_url = os.environ.get("LIVEKIT_INFERENCE_URL")
    if inference_url:
        return inference_url

    livekit_url = os.environ.get("LIVEKIT_URL", "")
    if ".staging.livekit.cloud" in livekit_url:
        return STAGING_INFERENCE_URL

    return DEFAULT_INFERENCE_URL


def create_access_token(api_key: str | None, api_secret: str | None, ttl: float = 600) -> str:
    grant = api.access_token.InferenceGrants(perform=True)
    return (
        api.AccessToken(api_key, api_secret)
        .with_identity("agent")
        .with_inference_grants(grant)
        .with_ttl(datetime.timedelta(seconds=ttl))
        .to_jwt()
    )
