from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TypeAlias

import httpx
import httpx2

HTTPXTimeout: TypeAlias = httpx2.Timeout | httpx.Timeout
HTTPXLimits: TypeAlias = httpx2.Limits | httpx.Limits

LegacyTimeoutException = httpx.TimeoutException

_DEPRECATION_MESSAGE = (
    "httpx.Timeout inputs are deprecated and will no longer be supported in LiveKit Agents 2.0. "
    "Use httpx2.Timeout instead."
)


def warn_on_legacy_timeout(timeout: HTTPXTimeout | None) -> None:
    if isinstance(timeout, httpx.Timeout):
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)


def to_httpx2_timeout(timeout: HTTPXTimeout | None) -> httpx2.Timeout | None:
    if timeout is None or isinstance(timeout, httpx2.Timeout):
        return timeout

    return httpx2.Timeout(
        connect=timeout.connect,
        read=timeout.read,
        write=timeout.write,
        pool=timeout.pool,
    )


def to_legacy_timeout(timeout: HTTPXTimeout) -> httpx.Timeout:
    if isinstance(timeout, httpx.Timeout):
        return timeout

    return httpx.Timeout(
        connect=timeout.connect,
        read=timeout.read,
        write=timeout.write,
        pool=timeout.pool,
    )


def legacy_async_client(
    *,
    timeout: HTTPXTimeout,
    limits: HTTPXLimits,
    headers: Mapping[str, str] | None = None,
    follow_redirects: bool = False,
) -> httpx.AsyncClient:
    if isinstance(limits, httpx2.Limits):
        resolved_limits = httpx.Limits(
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
        )
    else:
        resolved_limits = limits

    return httpx.AsyncClient(
        timeout=to_legacy_timeout(timeout),
        limits=resolved_limits,
        headers=headers,
        follow_redirects=follow_redirects,
    )
