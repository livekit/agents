from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

import httpx2

import openai

AsyncAzureADTokenProvider = Callable[[], str | Awaitable[str]]


def create_http_client(timeout: httpx2.Timeout | None = None) -> httpx2.AsyncClient:
    return openai.DefaultAsyncHttpx2Client(
        timeout=timeout or httpx2.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
        follow_redirects=True,
        limits=httpx2.Limits(
            max_connections=50,
            max_keepalive_connections=50,
            keepalive_expiry=120,
        ),
    )


def get_base_url(base_url: str | None) -> str:
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return base_url


__all__ = ["get_base_url", "create_http_client", "AsyncAzureADTokenProvider"]
