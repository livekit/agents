from __future__ import annotations

import hashlib
import hmac

import aiohttp
import pytest

from livekit.agents.beta.gtm_telemetry.models import PostCallReport
from livekit.agents.beta.gtm_telemetry.webhook import (
    PostCallWebhookDispatcher,
    WebhookConfig,
    WebhookDeliveryError,
)
from livekit.agents.utils import http_context

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


def _report() -> PostCallReport:
    return PostCallReport(
        report_id="r1", report_created_at=0.0, metadata={"secret_leak_check": "x"}
    )


class _FakePostCM:
    def __init__(self, result: object) -> None:
        self._result = result

    async def __aenter__(self):
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeResponse:
    def __init__(self, status: int, text: str = "") -> None:
        self.status = status
        self._text = text

    async def text(self) -> str:
        return self._text


class _FakeSession:
    """A scripted aiohttp.ClientSession-shaped fake: each .post() call consumes the
    next queued result (a _FakeResponse or an exception to raise) and records the call."""

    def __init__(self, results: list[object]) -> None:
        self._results = list(results)
        self.calls: list[dict] = []
        self.closed = False

    def post(self, url: str, *, data: bytes, headers: dict, timeout: object) -> _FakePostCM:
        self.calls.append({"url": url, "data": data, "headers": headers, "timeout": timeout})
        return _FakePostCM(self._results.pop(0))

    async def close(self) -> None:
        self.closed = True


async def _fake_sleep(delays: list[float]):
    async def _sleep(seconds: float) -> None:
        delays.append(seconds)

    return _sleep


# --- WebhookConfig validation --------------------------------------------------------


def test_webhook_url_scheme_http_https_accepted():
    WebhookConfig(url="http://example.com/hook", secret="s")
    WebhookConfig(url="https://example.com/hook", secret="s")


def test_webhook_url_invalid_scheme_raises_at_construction():
    with pytest.raises(ValueError):
        WebhookConfig(url="ftp://example.com/hook", secret="s")


def test_webhook_url_malformed_raises_at_construction():
    with pytest.raises(ValueError):
        WebhookConfig(url="not-a-url", secret="s")


def test_webhook_reserved_header_override_raises_at_construction():
    with pytest.raises(ValueError):
        WebhookConfig(url="https://example.com", secret="s", headers={"Content-Type": "text/plain"})
    with pytest.raises(ValueError):
        WebhookConfig(
            url="https://example.com", secret="s", headers={"X-LiveKit-Signature": "v1=x"}
        )


# --- signing --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_webhook_signature_format_is_v1_prefixed_hex_hmac_sha256():
    session = _FakeSession([_FakeResponse(200)])
    config = WebhookConfig(url="https://example.com/hook", secret="topsecret")
    dispatcher = PostCallWebhookDispatcher(config, http_session=session)
    report = _report()

    await dispatcher.send(report)

    sig_header = session.calls[0]["headers"]["X-LiveKit-Signature"]
    assert sig_header.startswith("v1=")
    digest = sig_header.removeprefix("v1=")
    assert len(digest) == 64  # hex sha256
    int(digest, 16)  # must be valid hex


@pytest.mark.asyncio
async def test_webhook_signature_computed_over_exact_sent_bytes():
    session = _FakeSession([_FakeResponse(200)])
    config = WebhookConfig(url="https://example.com/hook", secret="topsecret")
    dispatcher = PostCallWebhookDispatcher(config, http_session=session)
    report = _report()

    await dispatcher.send(report)

    sent_body = session.calls[0]["data"]
    sig_header = session.calls[0]["headers"]["X-LiveKit-Signature"]
    expected = hmac.new(b"topsecret", sent_body, hashlib.sha256).hexdigest()
    assert sig_header == f"v1={expected}"


@pytest.mark.asyncio
async def test_webhook_content_type_header_set():
    session = _FakeSession([_FakeResponse(200)])
    dispatcher = PostCallWebhookDispatcher(
        WebhookConfig(url="https://example.com/hook", secret="s"), http_session=session
    )
    await dispatcher.send(_report())
    assert session.calls[0]["headers"]["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_webhook_custom_headers_included():
    session = _FakeSession([_FakeResponse(200)])
    config = WebhookConfig(url="https://example.com/hook", secret="s", headers={"X-Api-Key": "abc"})
    dispatcher = PostCallWebhookDispatcher(config, http_session=session)
    await dispatcher.send(_report())
    assert session.calls[0]["headers"]["X-Api-Key"] == "abc"


# --- success / retry behavior ---------------------------------------------------------


@pytest.mark.asyncio
async def test_webhook_success_first_attempt_no_retry():
    session = _FakeSession([_FakeResponse(200)])
    delays: list[float] = []
    dispatcher = PostCallWebhookDispatcher(
        WebhookConfig(url="https://example.com/hook", secret="s"),
        http_session=session,
        sleep=await _fake_sleep(delays),
    )
    await dispatcher.send(_report())
    assert len(session.calls) == 1
    assert delays == []


@pytest.mark.asyncio
async def test_webhook_5xx_retries_with_configured_backoff_then_succeeds():
    session = _FakeSession([_FakeResponse(500), _FakeResponse(503), _FakeResponse(200)])
    delays: list[float] = []
    dispatcher = PostCallWebhookDispatcher(
        WebhookConfig(url="https://example.com/hook", secret="s", max_retries=2),
        http_session=session,
        sleep=await _fake_sleep(delays),
    )
    await dispatcher.send(_report())
    assert len(session.calls) == 3
    assert delays == [0.5, 1.0]


@pytest.mark.asyncio
async def test_webhook_connection_error_retries():
    session = _FakeSession([aiohttp.ClientConnectionError("boom"), _FakeResponse(200)])
    delays: list[float] = []
    dispatcher = PostCallWebhookDispatcher(
        WebhookConfig(url="https://example.com/hook", secret="s"),
        http_session=session,
        sleep=await _fake_sleep(delays),
    )
    await dispatcher.send(_report())
    assert len(session.calls) == 2
    assert delays == [0.5]


@pytest.mark.asyncio
async def test_webhook_timeout_error_retries():
    session = _FakeSession([TimeoutError("timed out"), _FakeResponse(200)])
    dispatcher = PostCallWebhookDispatcher(
        WebhookConfig(url="https://example.com/hook", secret="s"),
        http_session=session,
        sleep=await _fake_sleep([]),
    )
    await dispatcher.send(_report())
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_webhook_4xx_never_retries():
    session = _FakeSession([_FakeResponse(404, text="not found")])
    dispatcher = PostCallWebhookDispatcher(
        WebhookConfig(url="https://example.com/hook", secret="s"),
        http_session=session,
        sleep=await _fake_sleep([]),
    )
    with pytest.raises(WebhookDeliveryError) as exc_info:
        await dispatcher.send(_report())
    assert len(session.calls) == 1
    assert exc_info.value.attempts == 1
    assert exc_info.value.status == 404


@pytest.mark.asyncio
async def test_webhook_401_never_retries():
    session = _FakeSession([_FakeResponse(401)])
    dispatcher = PostCallWebhookDispatcher(
        WebhookConfig(url="https://example.com/hook", secret="s"),
        http_session=session,
        sleep=await _fake_sleep([]),
    )
    with pytest.raises(WebhookDeliveryError):
        await dispatcher.send(_report())
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_webhook_exhausts_retries_raises_typed_error_with_attempt_count():
    session = _FakeSession([_FakeResponse(500), _FakeResponse(500), _FakeResponse(500)])
    dispatcher = PostCallWebhookDispatcher(
        WebhookConfig(url="https://example.com/hook", secret="s", max_retries=2),
        http_session=session,
        sleep=await _fake_sleep([]),
    )
    with pytest.raises(WebhookDeliveryError) as exc_info:
        await dispatcher.send(_report())
    assert len(session.calls) == 3
    assert exc_info.value.attempts == 3
    assert exc_info.value.status == 500


@pytest.mark.asyncio
async def test_webhook_max_retries_zero_means_single_attempt():
    session = _FakeSession([_FakeResponse(500)])
    dispatcher = PostCallWebhookDispatcher(
        WebhookConfig(url="https://example.com/hook", secret="s", max_retries=0),
        http_session=session,
        sleep=await _fake_sleep([]),
    )
    with pytest.raises(WebhookDeliveryError) as exc_info:
        await dispatcher.send(_report())
    assert len(session.calls) == 1
    assert exc_info.value.attempts == 1


@pytest.mark.asyncio
async def test_webhook_delivery_error_never_contains_secret_or_full_report():
    session = _FakeSession([_FakeResponse(500)])
    dispatcher = PostCallWebhookDispatcher(
        WebhookConfig(url="https://example.com/hook", secret="super-secret-value", max_retries=0),
        http_session=session,
        sleep=await _fake_sleep([]),
    )
    with pytest.raises(WebhookDeliveryError) as exc_info:
        await dispatcher.send(_report())
    message = str(exc_info.value)
    assert "super-secret-value" not in message
    assert "secret_leak_check" not in message


# --- session resolution -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_webhook_uses_injected_http_session_when_provided():
    session = _FakeSession([_FakeResponse(200)])
    dispatcher = PostCallWebhookDispatcher(
        WebhookConfig(url="https://example.com/hook", secret="s"), http_session=session
    )
    await dispatcher.send(_report())
    assert len(session.calls) == 1
    await dispatcher.aclose()
    assert session.closed is False  # injected session is not owned, so not closed


@pytest.mark.asyncio
async def test_webhook_falls_back_to_http_context_session_when_bound(monkeypatch):
    session = _FakeSession([_FakeResponse(200)])
    monkeypatch.setattr(http_context, "http_session", lambda: session)

    dispatcher = PostCallWebhookDispatcher(
        WebhookConfig(url="https://example.com/hook", secret="s")
    )
    await dispatcher.send(_report())

    assert len(session.calls) == 1
    # the http_context-bound session's lifecycle isn't owned by the dispatcher
    await dispatcher.aclose()
    assert session.closed is False


@pytest.mark.asyncio
async def test_webhook_creates_and_owns_private_session_when_neither_available(monkeypatch):
    session = _FakeSession([_FakeResponse(200)])
    monkeypatch.setattr(
        "livekit.agents.beta.gtm_telemetry.webhook.aiohttp.ClientSession", lambda: session
    )
    dispatcher = PostCallWebhookDispatcher(
        WebhookConfig(url="https://example.com/hook", secret="s")
    )
    await dispatcher.send(_report())
    assert len(session.calls) == 1

    await dispatcher.aclose()
    assert session.closed is True
