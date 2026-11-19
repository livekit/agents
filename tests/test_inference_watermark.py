"""Unit tests for livekit.agents.inference.WatermarkDetector.

Covers the multipart request (parts, fixed file name, minted token), the inputs
``check`` accepts and refuses, the three answer shapes, pass-through of unknown
values, the request id header, error mapping for JSON, plain-text and non-UTF-8
bodies, the attempt timeout, and retries: which failures retry and how long each
one waits.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import wave
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import jwt
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectionError, APIError, APIStatusError, APITimeoutError
from livekit.agents.inference import (
    WatermarkDetector,
    WatermarkResult,
    watermark as watermark_module,
)
from livekit.agents.inference._utils import HEADER_USER_AGENT
from livekit.agents.inference.watermark import _CHECK_PATH, _parse_retry_after
from livekit.agents.types import APIConnectOptions

pytestmark = pytest.mark.unit

ReplyFactory = Callable[[], web.Response]

VERDICT = {
    "provider": "cartesia",
    "results": [{"type": "watermark", "outcome": "detected"}],
}
SCORE = {
    "provider": "inworld",
    "results": [{"type": "classifier", "probability": 0.97}],
}
UNSCORED = {
    "provider": "inworld",
    "results": [{"type": "classifier"}],
}
CAPACITY = {"type": "provider_capacity", "error": "provider is at capacity"}
BUSY = {"type": "gateway_busy", "error": "the gateway is busy"}
PROVIDER_ERROR = {"type": "provider_error", "error": "the provider failed"}


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\x00\x00" * 160)
    return buffer.getvalue()


@dataclass
class _Part:
    name: str | None
    filename: str | None
    headers: dict[str, str]
    data: bytes


@dataclass
class _Request:
    path: str
    headers: dict[str, str]
    parts: list[_Part]

    def part(self, name: str) -> _Part:
        return next(p for p in self.parts if p.name == name)


@dataclass
class _Gateway:
    """Answers each request with the next reply, repeating the last one."""

    replies: list[ReplyFactory]
    delay: float = 0.0
    requests: list[_Request] = field(default_factory=list)

    async def handle(self, request: web.Request) -> web.Response:
        parts: list[_Part] = []
        async for part in await request.multipart():
            assert isinstance(part, aiohttp.BodyPartReader)
            parts.append(
                _Part(
                    name=part.name,
                    filename=part.filename,
                    headers=dict(part.headers),
                    data=bytes(await part.read()),
                )
            )
        self.requests.append(
            _Request(path=request.path, headers=dict(request.headers), parts=parts)
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        index = min(len(self.requests), len(self.replies)) - 1
        return self.replies[index]()


class _RecordingSession:
    """Delegates to a real session and records each attempt's timeout."""

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session
        self.timeouts: list[aiohttp.ClientTimeout] = []

    def post(self, url: str, **kwargs: Any) -> Any:
        self.timeouts.append(kwargs["timeout"])
        return self._session.post(url, **kwargs)


def _json(
    body: dict[str, Any],
    *,
    status: int = 200,
    request_id: str | None = "req-1",
    retry_after: str | None = None,
    headers: dict[str, str] | None = None,
) -> ReplyFactory:
    all_headers: dict[str, str] = dict(headers or {})
    if request_id is not None:
        all_headers["X-LiveKit-Request-Id"] = request_id
    if retry_after is not None:
        all_headers["Retry-After"] = retry_after
    return lambda: web.json_response(body, status=status, headers=all_headers)


def _raw(
    body: bytes,
    *,
    status: int = 200,
    content_type: str = "application/json",
    request_id: str | None = "req-1",
) -> ReplyFactory:
    headers = {"X-LiveKit-Request-Id": request_id} if request_id is not None else {}
    return lambda: web.Response(
        body=body, status=status, content_type=content_type, headers=headers
    )


def _draining() -> web.Response:
    return web.Response(
        status=503,
        text="gateway is draining",
        headers={"Retry-After": "1", "Connection": "close"},
    )


@contextlib.asynccontextmanager
async def _serve(
    gateway: _Gateway, *, recording: bool = False, **kwargs: Any
) -> AsyncIterator[WatermarkDetector]:
    app = web.Application()
    app.router.add_post(f"/v1{_CHECK_PATH}", gateway.handle)
    server = TestServer(app)
    await server.start_server()
    session = aiohttp.ClientSession()
    options: dict[str, Any] = {
        "api_key": "test-key",
        "api_secret": "test-secret",
        "base_url": str(server.make_url("/v1")),
        "http_session": _RecordingSession(session) if recording else session,
    }
    options.update(kwargs)
    try:
        yield WatermarkDetector(**options)
    finally:
        await session.close()
        await server.close()


def _record_waits(detector: WatermarkDetector) -> list[float]:
    """Records this instance's retry waits instead of sleeping through them."""
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    detector._sleep = fake_sleep  # type: ignore[method-assign]
    return recorded


async def test_check_returns_verdict() -> None:
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway) as detector:
        check = await detector.check(wav_bytes(), provider="cartesia")

    assert check.provider == "cartesia"
    assert check.results == [WatermarkResult(type="watermark", outcome="detected")]
    assert check.results[0].probability is None


async def test_check_returns_score() -> None:
    gateway = _Gateway([_json(SCORE)])
    async with _serve(gateway) as detector:
        check = await detector.check(wav_bytes(), provider="inworld")

    assert check.provider == "inworld"
    assert check.results == [WatermarkResult(type="classifier", probability=0.97)]
    assert check.results[0].outcome is None


async def test_inworld_result_without_probability() -> None:
    gateway = _Gateway([_json(UNSCORED)])
    async with _serve(gateway) as detector:
        check = await detector.check(wav_bytes(), provider="inworld")

    assert check.provider == "inworld"
    result = check.results[0]
    assert result.type == "classifier"
    assert result.outcome is None
    assert result.probability is None


async def test_check_posts_to_the_gateway_route() -> None:
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway) as detector:
        await detector.check(wav_bytes(), provider="cartesia")

    assert gateway.requests[0].path == f"/v1{_CHECK_PATH}"


async def test_check_accepts_str_path(tmp_path: Path) -> None:
    audio = wav_bytes()
    path = tmp_path / "clip.wav"
    path.write_bytes(audio)
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway) as detector:
        await detector.check(str(path), provider="cartesia")

    assert gateway.requests[0].part("file").data == audio


async def test_check_accepts_pathlike(tmp_path: Path) -> None:
    audio = wav_bytes()
    path = tmp_path / "clip.wav"
    path.write_bytes(audio)
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway) as detector:
        await detector.check(path, provider="cartesia")

    assert gateway.requests[0].part("file").data == audio


async def test_check_accepts_bytes() -> None:
    audio = wav_bytes()
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway) as detector:
        await detector.check(audio, provider="cartesia")

    assert gateway.requests[0].part("file").data == audio


async def test_check_accepts_bytes_io() -> None:
    audio = wav_bytes()
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway) as detector:
        await detector.check(io.BytesIO(audio), provider="cartesia")

    assert gateway.requests[0].part("file").data == audio


async def test_check_accepts_open_file_and_leaves_it_open(tmp_path: Path) -> None:
    audio = wav_bytes()
    path = tmp_path / "clip.wav"
    path.write_bytes(audio)
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway) as detector:
        with path.open("rb") as opened:
            await detector.check(opened, provider="cartesia")
            assert not opened.closed

    assert gateway.requests[0].part("file").data == audio


async def test_path_file_name_is_not_sent(tmp_path: Path) -> None:
    path = tmp_path / "jane-doe-voicemail.wav"
    path.write_bytes(wav_bytes())
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway) as detector:
        await detector.check(path, provider="cartesia")

    sent = gateway.requests[0].part("file")
    assert sent.filename == "audio"
    assert "jane-doe" not in str(sent.headers)
    assert "jane-doe" not in str(gateway.requests[0].headers)


async def test_file_object_name_is_not_sent(tmp_path: Path) -> None:
    path = tmp_path / "jane-doe-voicemail.wav"
    path.write_bytes(wav_bytes())
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway) as detector:
        with path.open("rb") as opened:
            await detector.check(opened, provider="cartesia")

    sent = gateway.requests[0].part("file")
    assert sent.filename == "audio"
    assert "jane-doe" not in str(sent.headers)
    assert "jane-doe" not in str(gateway.requests[0].headers)


async def test_check_sends_exactly_provider_and_file_parts() -> None:
    audio = wav_bytes()
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway) as detector:
        await detector.check(audio, provider="cartesia")

    request = gateway.requests[0]
    assert sorted(str(part.name) for part in request.parts) == ["file", "provider"]
    assert request.part("provider").data == b"cartesia"
    assert request.part("provider").filename is None
    assert request.part("file").data == audio
    assert request.headers["Content-Type"].startswith("multipart/form-data")


async def test_request_carries_minted_token() -> None:
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway) as detector:
        await detector.check(wav_bytes(), provider="cartesia")

    headers = gateway.requests[0].headers
    scheme, token = headers["Authorization"].split(" ", 1)
    assert scheme == "Bearer"
    claims = jwt.decode(token, "test-secret", algorithms=["HS256"])
    assert claims["iss"] == "test-key"
    assert claims["inference"] == {"perform": True}
    assert headers[HEADER_USER_AGENT].startswith("LiveKit Agents/")


async def test_attempt_timeout_covers_the_gateway_worst_case() -> None:
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway, recording=True) as detector:
        await detector.check(wav_bytes(), provider="cartesia")
        session = detector._session

    assert isinstance(session, _RecordingSession)
    assert len(session.timeouts) == 1
    assert session.timeouts[0].total is not None
    assert session.timeouts[0].total >= 190.0


async def test_attempt_timeout_raises_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watermark_module, "_ATTEMPT_TIMEOUT", 0.05)
    gateway = _Gateway([_json(VERDICT)], delay=1.0)
    async with _serve(gateway) as detector:
        waits = _record_waits(detector)
        with pytest.raises(APITimeoutError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert not error.value.retryable
    assert len(gateway.requests) == 1
    assert waits == []


async def test_failed_connect_is_retried() -> None:
    server = TestServer(web.Application())
    await server.start_server()
    base_url = str(server.make_url("/v1"))
    await server.close()
    options = APIConnectOptions(max_retry=1, retry_interval=0)
    async with aiohttp.ClientSession() as session:
        detector = WatermarkDetector(
            api_key="test-key",
            api_secret="test-secret",
            base_url=base_url,
            http_session=session,
            conn_options=options,
        )
        waits = _record_waits(detector)
        with pytest.raises(APIConnectionError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.retryable
    assert waits == [0.1]


async def test_unknown_outcome_is_passed_through() -> None:
    body = {**VERDICT, "results": [{"type": "watermark", "outcome": "inconclusive"}]}
    gateway = _Gateway([_json(body)])
    async with _serve(gateway) as detector:
        check = await detector.check(wav_bytes(), provider="cartesia")

    assert check.results[0].outcome == "inconclusive"


async def test_unknown_type_is_passed_through() -> None:
    body = {**VERDICT, "results": [{"type": "fingerprint", "outcome": "detected"}]}
    gateway = _Gateway([_json(body)])
    async with _serve(gateway) as detector:
        check = await detector.check(wav_bytes(), provider="cartesia")

    assert check.results[0].type == "fingerprint"
    assert check.results[0].outcome == "detected"


async def test_request_id_comes_from_response_header() -> None:
    gateway = _Gateway([_json(VERDICT, request_id="req-42")])
    async with _serve(gateway) as detector:
        check = await detector.check(wav_bytes(), provider="cartesia")

    assert check.request_id == "req-42"


async def test_request_id_is_none_without_header() -> None:
    gateway = _Gateway([_json(VERDICT, request_id=None)])
    async with _serve(gateway) as detector:
        check = await detector.check(wav_bytes(), provider="cartesia")

    assert check.request_id is None


async def test_not_enabled_raises_non_retryable() -> None:
    body = {"type": "not_enabled", "error": "watermark checks are not enabled for this project"}
    gateway = _Gateway([_json(body, status=403)])
    async with _serve(gateway) as detector:
        waits = _record_waits(detector)
        with pytest.raises(APIStatusError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.status_code == 403
    assert not error.value.retryable
    assert error.value.body == {**body, "hint": ""}
    assert error.value.request_id == "req-1"
    assert len(gateway.requests) == 1
    assert waits == []


async def test_eu_pinned_cartesia_region_restricted_is_not_retried() -> None:
    body = {
        "type": "inference_region_restricted",
        "error": "cartesia has no endpoint in the project's region",
    }
    gateway = _Gateway([_json(body, status=403)])
    async with _serve(gateway) as detector:
        waits = _record_waits(detector)
        with pytest.raises(APIStatusError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.status_code == 403
    assert not error.value.retryable
    assert error.value.body == {**body, "hint": ""}
    assert len(gateway.requests) == 1
    assert waits == []


async def test_not_available_hint_lists_supported_tags() -> None:
    body = {
        "type": "not_available",
        "error": "cartesia cannot serve the requested data residency",
        "hint": "supported tags: global",
    }
    gateway = _Gateway([_json(body, status=400)])
    async with _serve(gateway) as detector:
        with pytest.raises(APIStatusError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.status_code == 400
    assert not error.value.retryable
    assert error.value.body == body
    assert len(gateway.requests) == 1


async def test_validation_error_carries_type_and_hint() -> None:
    body = {
        "type": "validation_error",
        "error": "file is not a supported audio format",
        "hint": "send WAV, MP3, OGG, FLAC, M4A or WebM",
    }
    gateway = _Gateway([_json(body, status=400)])
    async with _serve(gateway) as detector:
        with pytest.raises(APIStatusError) as error:
            await detector.check(b"\x00\x01\x02\x03", provider="cartesia")

    assert error.value.status_code == 400
    assert not error.value.retryable
    assert error.value.body == body
    assert "not a supported audio format" in error.value.message
    assert len(gateway.requests) == 1


async def test_unauthorized_plain_text_body_raises_status_error() -> None:
    def unauthorized() -> web.Response:
        return web.Response(status=401, text="invalid token")

    gateway = _Gateway([unauthorized])
    async with _serve(gateway) as detector:
        with pytest.raises(APIStatusError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.status_code == 401
    assert not error.value.retryable
    assert error.value.body == "invalid token"
    assert "invalid token" in error.value.message
    assert error.value.request_id is None
    assert len(gateway.requests) == 1


async def test_suspended_plain_text_403_raises_status_error() -> None:
    def suspended() -> web.Response:
        return web.Response(status=403, text="project suspended")

    gateway = _Gateway([suspended])
    async with _serve(gateway) as detector:
        with pytest.raises(APIStatusError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.status_code == 403
    assert not error.value.retryable
    assert error.value.body == "project suspended"
    assert len(gateway.requests) == 1


async def test_draining_plain_text_503_is_retried_after_retry_after() -> None:
    gateway = _Gateway([_draining, _json(VERDICT)])
    async with _serve(gateway) as detector:
        waits = _record_waits(detector)
        check = await detector.check(wav_bytes(), provider="cartesia")

    assert check.results[0].outcome == "detected"
    assert len(gateway.requests) == 2
    assert waits == [1.0]


async def test_draining_plain_text_503_surfaces_its_text() -> None:
    gateway = _Gateway([_draining])
    options = APIConnectOptions(max_retry=0)
    async with _serve(gateway, conn_options=options) as detector:
        with pytest.raises(APIStatusError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.status_code == 503
    assert error.value.retryable
    assert error.value.body == "gateway is draining"
    assert "gateway is draining" in error.value.message
    assert len(gateway.requests) == 1


async def test_capacity_503_is_retried_after_retry_after() -> None:
    gateway = _Gateway([_json(CAPACITY, status=503, retry_after="5"), _json(VERDICT)])
    async with _serve(gateway) as detector:
        waits = _record_waits(detector)
        check = await detector.check(wav_bytes(), provider="cartesia")

    assert check.results[0].outcome == "detected"
    assert len(gateway.requests) == 2
    assert gateway.requests[1].part("file").data == wav_bytes()
    assert waits == [5.0]


async def test_busy_503_is_retried_after_retry_after() -> None:
    gateway = _Gateway([_json(BUSY, status=503, retry_after="3"), _json(VERDICT)])
    async with _serve(gateway) as detector:
        waits = _record_waits(detector)
        check = await detector.check(wav_bytes(), provider="cartesia")

    assert check.results[0].outcome == "detected"
    assert len(gateway.requests) == 2
    assert waits == [3.0]


async def test_503_without_retry_after_uses_conn_options() -> None:
    gateway = _Gateway([_json(BUSY, status=503), _json(BUSY, status=503), _json(VERDICT)])
    options = APIConnectOptions(max_retry=2, retry_interval=1.5)
    async with _serve(gateway, conn_options=options) as detector:
        waits = _record_waits(detector)
        check = await detector.check(wav_bytes(), provider="cartesia")

    assert check.results[0].outcome == "detected"
    assert len(gateway.requests) == 3
    assert waits == [0.1, 1.5]


async def test_rate_limited_is_retried_after_retry_after() -> None:
    limited = {
        "type": "quota_exceeded",
        "error": "rate limit exceeded",
        "hint": "rate limit: 10 per minute per project",
    }
    gateway = _Gateway([_json(limited, status=429, retry_after="2"), _json(VERDICT)])
    async with _serve(gateway) as detector:
        waits = _record_waits(detector)
        check = await detector.check(wav_bytes(), provider="cartesia")

    assert check.results[0].outcome == "detected"
    assert len(gateway.requests) == 2
    assert waits == [2.0]


async def test_rate_limited_without_retry_after_is_not_retried() -> None:
    limited = {"type": "quota_exceeded", "error": "rate limit exceeded"}
    gateway = _Gateway([_json(limited, status=429)])
    async with _serve(gateway) as detector:
        waits = _record_waits(detector)
        with pytest.raises(APIStatusError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.status_code == 429
    assert not error.value.retryable
    assert len(gateway.requests) == 1
    assert waits == []


async def test_terminal_quota_denial_is_not_retried() -> None:
    limited = {"type": "quota_exceeded", "error": "out of credits"}
    gateway = _Gateway(
        [_json(limited, status=429, retry_after="2", headers={"X-Should-Retry": "false"})]
    )
    async with _serve(gateway) as detector:
        waits = _record_waits(detector)
        with pytest.raises(APIStatusError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.status_code == 429
    assert not error.value.retryable
    assert len(gateway.requests) == 1
    assert waits == []


async def test_long_retry_after_is_raised_instead_of_waited() -> None:
    limited = {"type": "quota_exceeded", "error": "daily allowance spent"}
    gateway = _Gateway([_json(limited, status=429, retry_after="3600")])
    async with _serve(gateway) as detector:
        waits = _record_waits(detector)
        with pytest.raises(APIStatusError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.status_code == 429
    assert len(gateway.requests) == 1
    assert waits == []


async def test_retries_stop_at_max_retry() -> None:
    gateway = _Gateway([_json(BUSY, status=503, retry_after="2")])
    options = APIConnectOptions(max_retry=1, retry_interval=0)
    async with _serve(gateway, conn_options=options) as detector:
        waits = _record_waits(detector)
        with pytest.raises(APIStatusError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.status_code == 503
    assert error.value.retryable
    assert error.value.body == {**BUSY, "hint": ""}
    assert len(gateway.requests) == 2
    assert waits == [2.0]


async def test_provider_error_502_is_not_retried() -> None:
    gateway = _Gateway([_json(PROVIDER_ERROR, status=502), _json(VERDICT)])
    async with _serve(gateway) as detector:
        waits = _record_waits(detector)
        with pytest.raises(APIStatusError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.status_code == 502
    assert not error.value.retryable
    assert error.value.body == {**PROVIDER_ERROR, "hint": ""}
    assert len(gateway.requests) == 1
    assert waits == []


async def test_provider_timeout_504_is_not_retried() -> None:
    body = {"type": "provider_timeout", "error": "the provider did not answer in time"}
    gateway = _Gateway([_json(body, status=504), _json(VERDICT)])
    async with _serve(gateway) as detector:
        waits = _record_waits(detector)
        with pytest.raises(APIStatusError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.status_code == 504
    assert not error.value.retryable
    assert len(gateway.requests) == 1
    assert waits == []


async def test_internal_server_error_is_not_retried() -> None:
    body = {"type": "internal_error", "error": "something broke"}
    gateway = _Gateway([_json(body, status=500)])
    async with _serve(gateway) as detector:
        waits = _record_waits(detector)
        with pytest.raises(APIStatusError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.status_code == 500
    assert not error.value.retryable
    assert len(gateway.requests) == 1
    assert waits == []


async def test_request_timeout_status_is_not_retried() -> None:
    body = {"type": "validation_error", "error": "upload not received in time"}
    gateway = _Gateway([_json(body, status=408)])
    async with _serve(gateway) as detector:
        waits = _record_waits(detector)
        with pytest.raises(APIStatusError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.status_code == 408
    assert not error.value.retryable
    assert len(gateway.requests) == 1
    assert waits == []


async def test_zero_retry_after_uses_the_retry_interval() -> None:
    gateway = _Gateway(
        [
            _json(BUSY, status=503, retry_after="0"),
            _json(BUSY, status=503, retry_after="0"),
            _json(VERDICT),
        ]
    )
    options = APIConnectOptions(max_retry=2, retry_interval=1.5)
    async with _serve(gateway, conn_options=options) as detector:
        waits = _record_waits(detector)
        check = await detector.check(wav_bytes(), provider="cartesia")

    assert check.results[0].outcome == "detected"
    assert len(gateway.requests) == 3
    assert waits == [0.1, 1.5]


def test_whole_seconds_retry_after_is_used() -> None:
    assert _parse_retry_after("7") == 7.0


def test_negative_retry_after_is_ignored() -> None:
    assert _parse_retry_after("-5") is None


def test_nan_retry_after_is_ignored() -> None:
    assert _parse_retry_after("nan") is None


def test_infinite_retry_after_is_ignored() -> None:
    assert _parse_retry_after("inf") is None


def test_fractional_retry_after_is_ignored() -> None:
    assert _parse_retry_after("1.5") is None


def test_date_retry_after_is_ignored() -> None:
    assert _parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None


async def test_non_utf8_error_body_raises_status_error() -> None:
    latin1 = "café upstream failed".encode("latin-1")
    gateway = _Gateway([_raw(latin1, status=502, content_type="text/plain")])
    async with _serve(gateway) as detector:
        with pytest.raises(APIStatusError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.status_code == 502
    assert not error.value.retryable
    assert error.value.body == "caf� upstream failed"
    assert len(gateway.requests) == 1


async def test_non_utf8_success_body_raises_api_error() -> None:
    latin1 = (
        '{"provider": "café", "results": [{"type": "watermark", "outcome": "detected"}]}'
    ).encode("latin-1")
    gateway = _Gateway([_raw(latin1)])
    async with _serve(gateway) as detector:
        with pytest.raises(APIError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert not isinstance(error.value, UnicodeError)
    assert not error.value.retryable
    assert len(gateway.requests) == 1


async def test_malformed_success_body_carries_request_id() -> None:
    gateway = _Gateway([_raw(b"<html>", content_type="text/html", request_id="req-9")])
    async with _serve(gateway) as detector:
        with pytest.raises(APIError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert isinstance(error.value, APIStatusError)
    assert error.value.request_id == "req-9"
    assert not error.value.retryable
    assert len(gateway.requests) == 1


async def test_empty_results_raises_api_error() -> None:
    gateway = _Gateway([_json({**VERDICT, "results": []})])
    async with _serve(gateway) as detector:
        with pytest.raises(APIError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert not error.value.retryable
    assert len(gateway.requests) == 1


async def test_error_body_without_type_or_error_keeps_raw_text() -> None:
    text = '{"message": "upstream said no"}'
    gateway = _Gateway([_raw(text.encode(), status=400)])
    async with _serve(gateway) as detector:
        with pytest.raises(APIStatusError) as error:
            await detector.check(wav_bytes(), provider="cartesia")

    assert error.value.status_code == 400
    assert error.value.body == text


async def test_empty_path_raises_before_any_request(tmp_path: Path) -> None:
    path = tmp_path / "jane-doe-voicemail.wav"
    path.write_bytes(b"")
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway) as detector:
        with pytest.raises(ValueError) as error:
            await detector.check(path, provider="cartesia")

    assert gateway.requests == []
    assert str(error.value) == "file is empty"


async def test_empty_bytes_raise_before_any_request() -> None:
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway) as detector:
        with pytest.raises(ValueError) as error:
            await detector.check(b"", provider="cartesia")

    assert gateway.requests == []
    assert str(error.value) == "file is empty"


async def test_empty_file_object_raises_before_any_request() -> None:
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway) as detector:
        with pytest.raises(ValueError) as error:
            await detector.check(io.BytesIO(b""), provider="cartesia")

    assert gateway.requests == []
    assert str(error.value) == "file is empty"


async def test_text_mode_file_object_raises_type_error(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    path.write_bytes(b"RIFF\xff\xfe\xfd not text")
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway) as detector:
        with path.open("r", encoding="utf-8") as opened:
            with pytest.raises(TypeError):
                await detector.check(opened, provider="cartesia")  # type: ignore[arg-type]
            assert opened.tell() == 0

    assert gateway.requests == []


async def test_unsupported_input_raises_type_error() -> None:
    gateway = _Gateway([_json(VERDICT)])
    async with _serve(gateway) as detector:
        with pytest.raises(TypeError):
            await detector.check(12345, provider="cartesia")  # type: ignore[arg-type]

    assert gateway.requests == []


async def test_file_object_is_resent_in_full_on_retry() -> None:
    audio = wav_bytes()
    stream = io.BytesIO(audio)
    gateway = _Gateway([_json(CAPACITY, status=503, retry_after="1"), _json(VERDICT)])
    async with _serve(gateway) as detector:
        waits = _record_waits(detector)
        check = await detector.check(stream, provider="cartesia")

    assert check.results[0].outcome == "detected"
    assert len(gateway.requests) == 2
    assert gateway.requests[0].part("file").data == audio
    assert gateway.requests[1].part("file").data == audio
    assert gateway.requests[1].part("file").filename == "audio"
    assert not stream.closed
    assert waits == [1.0]


async def test_retry_warning_logs_status_and_request_id(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "jane-doe-voicemail.wav"
    path.write_bytes(wav_bytes())
    gateway = _Gateway(
        [_json(CAPACITY, status=503, retry_after="1", request_id="req-7"), _json(VERDICT)]
    )
    async with _serve(gateway) as detector:
        _record_waits(detector)
        with caplog.at_level(logging.WARNING, logger="livekit.agents"):
            await detector.check(path, provider="cartesia")

    retries = [r for r in caplog.records if r.getMessage() == "watermark check failed, retrying"]
    assert len(retries) == 1
    assert retries[0].__dict__["status_code"] == 503
    assert retries[0].__dict__["request_id"] == "req-7"
    assert retries[0].__dict__["delay"] == 1.0
    assert "jane-doe" not in str(retries[0].__dict__)
