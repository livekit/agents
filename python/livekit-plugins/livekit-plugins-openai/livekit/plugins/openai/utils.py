from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

import httpx2 as httpx

from livekit.agents import APIConnectOptions
from livekit.agents.types import ErrorKind

AsyncAzureADTokenProvider = Callable[[], str | Awaitable[str]]


def get_base_url(base_url: str | None) -> str:
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return base_url


__all__ = ["get_base_url", "AsyncAzureADTokenProvider"]


def http_timeout(options: APIConnectOptions) -> httpx.Timeout:
    return httpx.Timeout(
        connect=options.connect_timeout,
        read=options.read_timeout,
        write=options.write_timeout,
        pool=options.pool_timeout,
    )


# OpenAI error codes and types by what failed; any other refused the request
_ERROR_KINDS: dict[str, ErrorKind] = {
    "invalid_api_key": "auth",
    "authentication_error": "auth",
    "permission_denied": "auth",
    "insufficient_quota": "quota",
    "billing_hard_limit_reached": "quota",
    "rate_limit_exceeded": "rate_limit",
    "rate_limit_error": "rate_limit",
    "server_error": "server",
    "server_is_overloaded": "server",
}


def error_kind(code: str | None) -> ErrorKind:
    """The kind of an OpenAI error event's ``code`` or ``type``."""
    return _ERROR_KINDS.get(code or "", "invalid_request")
