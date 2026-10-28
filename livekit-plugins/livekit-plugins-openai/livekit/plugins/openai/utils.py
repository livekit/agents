from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

import httpx2 as httpx

from livekit.agents import APIConnectOptions, APIError

AsyncAzureADTokenProvider = Callable[[], str | Awaitable[str]]


def get_base_url(base_url: str | None) -> str:
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return base_url


__all__ = ["get_base_url", "AsyncAzureADTokenProvider"]


def http_timeout(options: APIConnectOptions) -> httpx.Timeout:
    if options.total_timeout is not None:
        raise APIError(
            "OpenAI SDK cannot enforce total_timeout across framework attempts", retryable=False
        )
    return httpx.Timeout(
        connect=options.connect_timeout,
        read=options.read_timeout,
        write=options.write_timeout,
        pool=options.pool_timeout,
    )
