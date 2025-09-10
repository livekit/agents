from __future__ import annotations

import datetime

from livekit import api


def create_access_token(api_key: str | None, api_secret: str | None, ttl: float = 600) -> str:
    grant = api.access_token.InferenceGrants(perform=True)
    return (
        api.AccessToken(api_key, api_secret)
        .with_identity("agent")
        .with_inference_grants(grant)
        .with_ttl(datetime.timedelta(seconds=ttl))
        .to_jwt()
    )
