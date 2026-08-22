"""Signed webhook delivery for post-call reports.

Beta: not covered by semver stability guarantees.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

import aiohttp
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from ...utils import http_context
from .models import PostCallReport

_RESERVED_HEADERS = {"content-type", "x-livekit-signature"}
_ALLOWED_SCHEMES = {"http", "https"}


class WebhookConfig(BaseModel):
    """Configuration for :class:`PostCallWebhookDispatcher`.

    Signature format: ``X-LiveKit-Signature: v1=<hex_digest>``, an HMAC-SHA256 digest of
    the exact JSON request body bytes, computed with ``secret``::

        hmac.new(secret.encode(), body_bytes, hashlib.sha256).hexdigest()

    The caller controls ``url`` — this library places no restriction beyond requiring an
    ``http``/``https`` scheme and does not attempt SSRF protection beyond that; treat the
    destination as trusted, and never place a secret in the URL itself.
    """

    url: str
    secret: str
    headers: dict[str, str] = Field(default_factory=dict)
    """Extra headers to send. ``Content-Type`` and ``X-LiveKit-Signature`` are reserved
    and cannot be overridden here."""
    timeout: float = 10.0
    max_retries: int = 2
    """Retries after the initial attempt (default: 2, i.e. 3 attempts total)."""
    retry_backoff: tuple[float, ...] = (0.5, 1.0)
    """Delay, in seconds, before each retry attempt. The last value is reused if there
    are more retries than entries."""

    @model_validator(mode="after")
    def _validate(self) -> Self:
        parsed = urlsplit(self.url)
        if parsed.scheme not in _ALLOWED_SCHEMES or not parsed.netloc:
            raise ValueError(f"WebhookConfig.url must be an http(s) URL, got: {self.url!r}")
        for name in self.headers:
            if name.lower() in _RESERVED_HEADERS:
                raise ValueError(f"WebhookConfig.headers cannot set the reserved header {name!r}")
        return self


class WebhookDeliveryError(Exception):
    """Raised when webhook delivery fails after exhausting retries, or immediately on a
    non-retryable (4xx) response. Never includes the secret or the full report body."""

    def __init__(
        self,
        *,
        attempts: int,
        status: int | None,
        response_summary: str | None,
        cause: Exception | None,
    ) -> None:
        self.attempts = attempts
        self.status = status
        self.response_summary = response_summary
        message = f"webhook delivery failed after {attempts} attempt(s)"
        if status is not None:
            message += f" (status={status})"
        if cause is not None:
            message += f": {cause}"
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


class PostCallWebhookDispatcher:
    """Delivers a :class:`PostCallReport` to a caller-configured webhook endpoint.

    POSTs the report as ``application/json``, signed with HMAC-SHA256 over the exact
    request body (see :class:`WebhookConfig`). Retries connection errors, timeouts, and
    5xx responses using a fixed backoff schedule (default: 0.5s then 1.0s — 3 attempts
    total). 4xx responses are never retried. Deliberately does not reuse
    ``types.APIConnectOptions`` — its "immediate retry, then flat interval" schedule, and
    lack of a 4xx/5xx distinction, don't match the schedule this dispatcher implements.

    HTTP session resolution, in order: an injected ``http_session``; otherwise the
    shared session from ``livekit.agents.utils.http_context`` when bound (the normal
    case inside a job process); otherwise a private session is created and owned by
    this dispatcher, closed by :meth:`aclose`. This lets the dispatcher work with or
    without a ``JobContext`` — see ``utils.http_context.open()`` to bind the shared
    session in scripts/tests run outside a job worker.
    """

    def __init__(
        self,
        config: WebhookConfig,
        *,
        http_session: aiohttp.ClientSession | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._config = config
        self._injected_session = http_session
        self._owned_session: aiohttp.ClientSession | None = None
        self._sleep = sleep

    async def send(self, report: PostCallReport) -> None:
        """POST ``report`` to the configured webhook.

        Raises :class:`WebhookDeliveryError` immediately on a 4xx response, or once
        retries are exhausted for connection errors/timeouts/5xx responses.
        """
        body = report.model_dump_json().encode("utf-8")
        signature = hmac.new(self._config.secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        headers = {
            **self._config.headers,
            "Content-Type": "application/json",
            "X-LiveKit-Signature": f"v1={signature}",
        }

        session = self._resolve_session()
        last_error: WebhookDeliveryError | None = None
        max_attempts = self._config.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            try:
                async with session.post(
                    self._config.url,
                    data=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self._config.timeout),
                ) as resp:
                    if resp.status < 400:
                        return
                    error = WebhookDeliveryError(
                        attempts=attempt,
                        status=resp.status,
                        response_summary=await _safe_response_text(resp),
                        cause=None,
                    )
                    if resp.status < 500:
                        raise error
                    last_error = error
            except WebhookDeliveryError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = WebhookDeliveryError(
                    attempts=attempt, status=None, response_summary=None, cause=exc
                )

            if attempt < max_attempts:
                delay_idx = min(attempt - 1, len(self._config.retry_backoff) - 1)
                await self._sleep(self._config.retry_backoff[delay_idx])

        assert last_error is not None  # loop always sets it before falling through
        raise last_error

    async def aclose(self) -> None:
        """Close the internally-owned HTTP session, if one was created.

        A no-op when an ``http_session`` was injected, or when the shared
        ``utils.http_context`` session was used (that session's lifecycle belongs to its
        own context, not to this dispatcher).
        """
        if self._owned_session is not None and not self._owned_session.closed:
            await self._owned_session.close()
        self._owned_session = None

    def _resolve_session(self) -> aiohttp.ClientSession:
        if self._injected_session is not None:
            return self._injected_session
        try:
            return http_context.http_session()
        except RuntimeError:
            pass
        if self._owned_session is None or self._owned_session.closed:
            self._owned_session = aiohttp.ClientSession()
        return self._owned_session


async def _safe_response_text(resp: aiohttp.ClientResponse) -> str | None:
    try:
        text = await resp.text()
    except Exception:  # noqa: BLE001 - best-effort diagnostic only
        return None
    return text[:500]
