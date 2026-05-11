from __future__ import annotations

import datetime
import os
import platform

from livekit import api

from ..version import __version__

DEFAULT_INFERENCE_URL = "https://agent-gateway.livekit.cloud/v1"
STAGING_INFERENCE_URL = "https://agent-gateway.staging.livekit.cloud/v1"

HEADER_USER_AGENT = "User-Agent"
HEADER_ROOM_ID = "X-LiveKit-Room-ID"
HEADER_JOB_ID = "X-LiveKit-Job-ID"
HEADER_INFERENCE_PROVIDER = "X-LiveKit-Inference-Provider"
HEADER_INFERENCE_PRIORITY = "X-LiveKit-Inference-Priority"


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


def get_inference_headers() -> dict[str, str]:
    """Build identification headers for inference requests.

    Always includes User-Agent with SDK version and Python version.
    Includes X-LiveKit-Room-ID and X-LiveKit-Job-ID when running
    inside a job context (omitted in console mode or tests).
    """
    headers: dict[str, str] = {
        HEADER_USER_AGENT: (f"LiveKit Agents/{__version__} (python {platform.python_version()})"),
    }
    try:
        from ..job import get_job_context

        ctx = get_job_context()
        if ctx.job.room.sid:
            headers[HEADER_ROOM_ID] = ctx.job.room.sid
        if ctx.job.id:
            headers[HEADER_JOB_ID] = ctx.job.id
    except RuntimeError:
        pass
    return headers


def create_access_token(api_key: str | None, api_secret: str | None, ttl: float = 600) -> str:
    grant = api.access_token.InferenceGrants(perform=True)
    return (
        api.AccessToken(api_key, api_secret)
        .with_identity("agent")
        .with_inference_grants(grant)
        .with_ttl(datetime.timedelta(seconds=ttl))
        .to_jwt()
    )
