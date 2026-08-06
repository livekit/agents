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
HEADER_AGENT_ID = "X-LiveKit-Agent-ID"
HEADER_WORKER_TOKEN = "X-LiveKit-Worker-Token"
HEADER_INFERENCE_PROVIDER = "X-LiveKit-Inference-Provider"
HEADER_INFERENCE_PRIORITY = "X-LiveKit-Inference-Priority"

# Inference class forced on every LiveKit Inference request issued from a text
# simulation. A simulation run dispatches many jobs at once and nobody is waiting
# on the answers, so it is batch load: it must never compete with live traffic for
# gateway capacity. Audio simulations are excluded: they run in real time against
# the audio pipeline, so their latency has to stay representative.
SIMULATION_INFERENCE_CLASS = "low"


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


def get_inference_headers(*, inference_class: str | None = None) -> dict[str, str]:
    """Build identification headers for inference requests.

    Always includes User-Agent with SDK version and Python version.
    Includes X-LiveKit-Room-ID, X-LiveKit-Job-ID, and X-LiveKit-Agent-ID
    when running inside a job context (omitted in console mode or tests).
    Includes X-LiveKit-Worker-Token when LIVEKIT_WORKER_TOKEN is set (hosted agents).

    ``inference_class`` is the class the caller configured, if any; it lands in
    X-LiveKit-Inference-Priority. This resolves it for every LiveKit Inference model,
    so it is the one place that decides which class a request goes out with.
    """
    headers: dict[str, str] = {
        HEADER_USER_AGENT: (f"LiveKit Agents/{__version__} (python {platform.python_version()})"),
    }
    try:
        from ..job import get_job_context

        ctx = get_job_context()
        if isinstance(room_sid := ctx.job.room.sid, str) and room_sid:
            headers[HEADER_ROOM_ID] = room_sid
        if isinstance(job_id := ctx.job.id, str) and job_id:
            headers[HEADER_JOB_ID] = job_id
        # for hosted agents where job context is always present
        if worker_token := os.getenv("LIVEKIT_WORKER_TOKEN"):
            headers[HEADER_WORKER_TOKEN] = worker_token
        # ctx.agent resolves to room.local_participant, which raises until the room
        # is connected (STT/TTS may open their websockets before ctx.connect()).
        # isconnected() is the codebase-standard readiness guard (see
        # utils/participant.py); local_participant is set once on connect and never
        # cleared, so the access below won't raise once isconnected() is True.
        if ctx.room.isconnected() and isinstance(agent_sid := ctx.agent.sid, str) and agent_sid:
            headers[HEADER_AGENT_ID] = agent_sid
    except RuntimeError:
        pass

    # A text simulation overrides the configured class rather than falling back to it:
    # a simulation must never be able to ask for priority capacity.
    from ..job import current_simulation
    from ..simulation import SimulationMode

    sim = current_simulation()
    if sim is not None and sim.simulation_mode == SimulationMode.SIMULATION_MODE_TEXT:
        inference_class = SIMULATION_INFERENCE_CLASS

    if inference_class:
        headers[HEADER_INFERENCE_PRIORITY] = inference_class

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
