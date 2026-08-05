"""Resilient webhook dispatcher for post-call reports.

Signs the exact request body bytes with HMAC-SHA256 and retries transient
failures with exponential backoff. Failures are swallowed and logged — a
telemetry failure must never propagate into the agent session.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac

import aiohttp

from ...log import logger
from ...utils import http_context
from .models import PostCallReport


class WebhookDispatcher:
    """POST a :class:`PostCallReport` to a webhook URL with signing and retries.

    Signature scheme: when ``webhook_secret`` is set, the request carries an
    ``X-LiveKit-Signature`` header containing the HMAC-SHA256 hex digest of the
    exact JSON body bytes. Note this is a convention specific to this
    dispatcher — livekit-server's own webhooks use a JWT ``Authorization``
    header instead — so receivers must implement this verification:

    .. code-block:: python

        import hashlib, hmac

        def verify(body: bytes, signature: str, timestamp: str, secret: str) -> bool:
            payload = f"{timestamp}.".encode() + body
            expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
            return hmac.compare_digest(expected, signature)

    Retry policy: ``max_retries`` total attempts; 5xx responses and network
    errors are retried with exponential backoff (``base_delay * 2**attempt``,
    slept only *between* attempts); 4xx responses are not retried. Worst-case
    wall time with defaults is ~31.5s (3 x 10s request timeouts + 0.5s + 1s
    backoff) — callers awaiting the dispatch (e.g. ``aflush()``) must budget
    for it.

    The HTTP session is resolved lazily inside :meth:`dispatch` via
    ``utils.http_context.http_session()`` unless one is injected through the
    ``http_session`` kwarg, so constructing a dispatcher outside a job context
    never raises.
    """

    def __init__(
        self,
        url: str,
        *,
        webhook_secret: str | None = None,
        max_retries: int = 3,
        base_delay: float = 0.5,
        timeout: float = 10.0,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        if url.lower().startswith("http://"):
            logger.warning(
                "Webhook URL uses http:// scheme. Sensitive conversation data will be transmitted in cleartext."
            )
        self._url = url
        self._webhook_secret = webhook_secret
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._timeout = timeout
        self._http_session = http_session

    async def dispatch(self, report: PostCallReport) -> bool:
        """Send the report to the webhook URL. Returns True on 2xx, never raises."""
        try:
            return await self._dispatch_impl(report)
        except Exception:
            logger.exception("unexpected error dispatching post-call report webhook")
            return False

    async def _dispatch_impl(self, report: PostCallReport) -> bool:
        import time

        body = report.model_dump_json().encode()
        headers = {"Content-Type": "application/json"}
        if self._webhook_secret is not None:
            timestamp = str(int(time.time()))
            payload_to_sign = f"{timestamp}.".encode() + body
            headers["X-LiveKit-Signature"] = hmac.new(
                self._webhook_secret.encode(), payload_to_sign, hashlib.sha256
            ).hexdigest()
            headers["X-LiveKit-Timestamp"] = timestamp

        # resolved lazily here (not in __init__) so the dispatcher can be
        # constructed outside a job context; see utils/http_context.py
        session = self._http_session or http_context.http_session()

        for attempt in range(self._max_retries):
            try:
                async with session.post(
                    self._url,
                    data=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                    allow_redirects=False,
                ) as resp:
                    if 200 <= resp.status < 300:
                        return True

                    if 300 <= resp.status < 400:
                        logger.warning(
                            "post-call report webhook returned redirect %d, not retrying",
                            resp.status,
                        )
                        return False

                    if 400 <= resp.status < 500:
                        logger.warning(
                            "post-call report webhook rejected with status %d, not retrying",
                            resp.status,
                        )
                        return False

                    logger.warning(
                        "post-call report webhook returned status %d (attempt %d/%d)",
                        resp.status,
                        attempt + 1,
                        self._max_retries,
                    )
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(
                    "post-call report webhook request failed (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries,
                    e,
                )

            if attempt < self._max_retries - 1:
                await asyncio.sleep(self._base_delay * 2**attempt)

        logger.warning(
            "post-call report webhook dispatch failed after %d attempts", self._max_retries
        )
        return False
