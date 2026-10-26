from __future__ import annotations

import datetime
import os
import platform
from collections.abc import Mapping
from dataclasses import dataclass

from livekit import api

from .. import _core
from .._generated.inference_llm import InferenceClass
from ..types import NotGivenOr
from ..utils import is_given
from ..utils.misc import shortuuid
from ..version import __version__

__all__ = [
    "InferenceClass",
    "Credentials",
    "create_access_token",
    "extract_quota_usage",
    "get_default_inference_url",
    "get_inference_headers",
    "resolve_credentials",
]

HEADER_SESSION_ID = "X-LiveKit-Session-ID"
HEADER_INFERENCE_PROVIDER = "X-LiveKit-Inference-Provider"
HEADER_INFERENCE_PRIORITY = "X-LiveKit-Inference-Priority"


def get_default_inference_url() -> str:
    """``LIVEKIT_INFERENCE_URL``, else the staging or production gateway by ``LIVEKIT_URL``."""
    return _core.inference_default_url()  # type: ignore[no-any-return]


def extract_quota_usage(headers: Mapping[str, str]) -> dict[str, str]:
    """The gateway's quota headers as log fields; empty when it stamped none."""
    return _core.inference_quota_usage(list(headers.items()))  # type: ignore[no-any-return]


def resolve_credentials(api_key: str | None, api_secret: str | None) -> tuple[str, str]:
    """The inference key and secret: explicit, else ``LIVEKIT_INFERENCE_*``, else ``LIVEKIT_*``."""
    key, secret = (
        _core.inference_resolve_api_key(api_key),
        _core.inference_resolve_api_secret(api_secret),
    )
    if not key:
        raise ValueError(
            "api_key is required, either as argument or set LIVEKIT_API_KEY environmental variable"
        )
    if not secret:
        raise ValueError(
            "api_secret is required, either as argument or set LIVEKIT_API_SECRET environmental "
            "variable"
        )
    return key, secret


@dataclass(frozen=True)
class Credentials:
    """Where a gateway client connects, and with what."""

    base_url: str
    api_key: str
    api_secret: str

    @classmethod
    def resolve(
        cls,
        base_url: NotGivenOr[str] | None,
        api_key: NotGivenOr[str] | None,
        api_secret: NotGivenOr[str] | None,
    ) -> Credentials:
        key, secret = resolve_credentials(
            api_key if is_given(api_key) else None, api_secret if is_given(api_secret) else None
        )
        url = base_url if is_given(base_url) and base_url else get_default_inference_url()
        return cls(base_url=url, api_key=key, api_secret=secret)


def get_inference_headers(*, inference_class: str | None = None) -> dict[str, str]:
    """The identification headers of an inference request: the SDK, and in a job its room, job,
    agent, worker token, the job's ``inference_headers`` and a fresh session id."""
    headers = {"User-Agent": f"LiveKit Agents/{__version__} (python {platform.python_version()})"}
    if inference_class:
        headers[HEADER_INFERENCE_PRIORITY] = inference_class
    try:
        from ..job import get_job_context

        ctx = get_job_context()
    except RuntimeError:
        return headers
    if isinstance(room_sid := ctx.job.room.sid, str) and room_sid:
        headers["X-LiveKit-Room-ID"] = room_sid
    if isinstance(job_id := ctx.job.id, str) and job_id:
        headers["X-LiveKit-Job-ID"] = job_id
    if worker_token := os.getenv("LIVEKIT_WORKER_TOKEN"):
        headers["X-LiveKit-Worker-Token"] = worker_token
    # the local participant exists only once the room is connected
    if ctx.room.isconnected() and isinstance(agent_sid := ctx.agent.sid, str) and agent_sid:
        headers["X-LiveKit-Agent-ID"] = agent_sid
    headers.update(ctx.inference_headers)
    headers[HEADER_SESSION_ID] = shortuuid("inference_")
    return headers


def create_access_token(api_key: str | None, api_secret: str | None, ttl: float = 600) -> str:
    """A gateway token, for the clients the core does not run (realtime, avatar)."""
    return (
        api.AccessToken(api_key, api_secret)
        .with_identity("agent")
        .with_inference_grants(api.access_token.InferenceGrants(perform=True))
        .with_ttl(datetime.timedelta(seconds=ttl))
        .to_jwt()
    )
