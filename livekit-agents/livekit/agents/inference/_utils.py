from __future__ import annotations

import datetime
import os
import platform
from collections.abc import Mapping
from typing import Literal

import aiohttp

from livekit import api

from .._exceptions import APIStatusError, create_api_error_from_http
from .._token_mode import INFERENCE_CLAIMS, token_mode_token
from ..types import NOT_GIVEN, NotGivenOr
from ..utils.deprecation import deprecate_params
from ..utils.misc import is_given, shortuuid
from ..version import __version__

InferenceClass = Literal["priority", "standard", "low"]
"""Scheduling class for a request. ``low`` yields to voice traffic, so it is only
appropriate for work no caller is waiting on."""

DEFAULT_INFERENCE_URL = "https://agent-gateway.livekit.cloud/v1"
STAGING_INFERENCE_URL = "https://agent-gateway.staging.livekit.cloud/v1"

HEADER_USER_AGENT = "User-Agent"
HEADER_ROOM_ID = "X-LiveKit-Room-ID"
HEADER_JOB_ID = "X-LiveKit-Job-ID"
HEADER_AGENT_ID = "X-LiveKit-Agent-ID"
HEADER_SESSION_ID = "X-LiveKit-Session-ID"
HEADER_WORKER_TOKEN = "X-LiveKit-Worker-Token"
HEADER_INFERENCE_PROVIDER = "X-LiveKit-Inference-Provider"
HEADER_INFERENCE_PRIORITY = "X-LiveKit-Inference-Priority"
HEADER_REQUEST_ID = "X-LiveKit-Request-Id"


@deprecate_params(
    {"provider": "Omit this argument to let LiveKit route requests and use fallback providers."},
    target_date="March 1, 2027",
)
def _resolve_provider(*, provider: NotGivenOr[str | None] = NOT_GIVEN) -> str | None:
    return provider if is_given(provider) else None


def _terminal_denial(headers: Mapping[str, str] | None) -> bool:
    # quota denials a retry cannot clear (out of credits, project suspended);
    # httpx and aiohttp header maps are both case-insensitive
    return (headers or {}).get("X-Should-Retry", "").strip().lower() == "false"


def create_api_error_from_response(error: aiohttp.ClientResponseError) -> APIStatusError:
    """Preserve quota details when a failed WebSocket upgrade hides its body."""
    # UPDATE_ALL_INSTANCES(INFERENCE_QUOTA_ERROR_HEADERS): agent-gateway/pkg/http/header.go.
    headers: Mapping[str, str] = error.headers or {}
    message = headers.get("X-LiveKit-Inference-Quota-Message") or error.message
    category = headers.get("X-LiveKit-Inference-Quota-Category")
    result = create_api_error_from_http(
        message,
        status=error.status,
        request_id=headers.get(HEADER_REQUEST_ID),
        body={"category": category} if category else None,
    )
    if _terminal_denial(headers):
        result.retryable = False
    return result


# Quota telemetry headers the inference gateway stamps on LLM completions
# responses (including 429 rejections). RPM/TPM are per-minute rate limits;
# Credits is a cumulative token-credit balance. The gateway omits a dimension's
# headers when it has no data for it, so absence means "not enforced / unknown",
# never zero.
QUOTA_HEADER_FIELDS: dict[str, str] = {
    "X-LiveKit-Inference-RPM-Limit": "rpm_limit",
    "X-LiveKit-Inference-RPM-Used": "rpm_used",
    "X-LiveKit-Inference-TPM-Limit": "tpm_limit",
    "X-LiveKit-Inference-TPM-Used": "tpm_used",
    "X-LiveKit-Inference-Credits-Limit": "credits_limit",
    "X-LiveKit-Inference-Credits-Used": "credits_used",
}


def extract_quota_usage(headers: Mapping[str, str]) -> dict[str, str]:
    """Extract the gateway's quota headers into log-friendly fields.

    Only dimensions the gateway actually stamped are returned; an empty dict
    means the response carried no quota telemetry.
    """
    return {
        field: value
        for header, field in QUOTA_HEADER_FIELDS.items()
        if (value := headers.get(header))
    }


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


def resolve_credentials(
    api_key: str | None,
    api_secret: str | None,
) -> tuple[str, str]:
    resolved_api_key = api_key or os.getenv(
        "LIVEKIT_INFERENCE_API_KEY", os.getenv("LIVEKIT_API_KEY", "")
    )
    if not resolved_api_key and token_mode_token() is None:
        raise ValueError(
            "api_key is required, either as argument or set LIVEKIT_API_KEY environmental variable"
        )

    resolved_api_secret = api_secret or os.getenv(
        "LIVEKIT_INFERENCE_API_SECRET", os.getenv("LIVEKIT_API_SECRET", "")
    )
    if not resolved_api_secret and token_mode_token() is None:
        raise ValueError(
            "api_secret is required, either as argument or set LIVEKIT_API_SECRET environmental variable"
        )

    # both are empty in token mode, where create_access_token returns the job's token
    return resolved_api_key or "", resolved_api_secret or ""


def get_inference_headers(*, inference_class: str | None = None) -> dict[str, str]:
    """Build identification headers for inference requests.

    Always includes User-Agent with SDK version and Python version.
    Includes X-LiveKit-Room-ID, X-LiveKit-Job-ID, X-LiveKit-Agent-ID, and a
    client-generated X-LiveKit-Session-ID when running inside a job context
    (omitted in console mode or tests).
    Includes X-LiveKit-Worker-Token when LIVEKIT_WORKER_TOKEN is set (hosted agents).

    ``inference_class`` is the class the caller configured, if any; it lands in
    X-LiveKit-Inference-Priority. A job can override it through
    :attr:`JobContext.inference_headers`. The SDK-owned session ID remains reserved.
    """
    headers: dict[str, str] = {
        HEADER_USER_AGENT: (f"LiveKit Agents/{__version__} (python {platform.python_version()})"),
    }
    if inference_class:
        headers[HEADER_INFERENCE_PRIORITY] = inference_class

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
        # merged last: what the job asserts about itself outranks what an individual
        # model was configured with.
        headers.update(ctx.inference_headers)
        # The session ID identifies this outbound HTTP request or WebSocket
        # connection. It is SDK-owned so custom inference headers cannot replace it.
        headers[HEADER_SESSION_ID] = shortuuid("inference_")
    except RuntimeError:
        pass

    return headers


def create_inference_request_id(
    session_id: str | None,
    request_kind: Literal["tts", "eot"],
    *,
    fallback_prefix: str = "",
) -> str:
    """Create a logical request ID linked to its inference connection."""
    prefix = f"{session_id}_{request_kind}_" if session_id else fallback_prefix
    return shortuuid(prefix)


_NO_JOB_TOKEN = "no-job-token"


def create_client_access_token(api_key: str | None, api_secret: str | None) -> str:
    """The token an inference client is constructed with. A client can be made before its
    job runs (process setup), where token mode has no token yet: there it is a placeholder,
    and the client gets the job's token before each request."""
    if (not api_key or not api_secret) and token_mode_token() is not None:
        return _NO_JOB_TOKEN  # the client refuses an empty key
    return create_access_token(api_key, api_secret)


def _job_inference_token() -> str | None:
    from ..job import get_job_context

    try:
        return get_job_context()._job_token(INFERENCE_CLAIMS)
    except RuntimeError:
        return None


def access_token_available(api_key: str | None, api_secret: str | None) -> bool:
    """False in token mode outside a running job, where ``create_access_token`` raises."""
    if (not api_key or not api_secret) and token_mode_token() is not None:
        return _job_inference_token() is not None
    return True


def create_access_token(api_key: str | None, api_secret: str | None, ttl: float = 600) -> str:
    if (not api_key or not api_secret) and token_mode_token() is not None:
        # token mode: no secret to sign with, the job keeps a token from the server current
        token = _job_inference_token()
        if token is None:
            raise RuntimeError(
                "LiveKit Inference in token mode (LIVEKIT_TOKEN without an API key and secret) "
                "needs a running job: the job gets its inference token from the LiveKit server"
            )
        return token

    grant = api.access_token.InferenceGrants(perform=True)
    return (
        api.AccessToken(api_key, api_secret)
        .with_identity("agent")
        .with_inference_grants(grant)
        .with_ttl(datetime.timedelta(seconds=ttl))
        .to_jwt()
    )
