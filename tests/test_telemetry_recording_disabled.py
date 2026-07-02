"""Unit tests for the "recording disabled by owner" upload latch in
``livekit.agents.telemetry.traces``.

When a LiveKit Cloud project has data recording disabled, the OTLP exporters (and the
one-shot recording upload) get a 401/403 whose body says recording is disabled. The SDK
detects that signal, warns once per session, and short-circuits further uploads by handing
the exporter a synthetic 200 so it stops logging errors.
"""

from __future__ import annotations

import logging

import pytest

from livekit.agents.telemetry import traces

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_latch():
    traces._upload_gate.reset()
    yield
    traces._upload_gate.reset()


def _status_proto(message: str) -> bytes:
    from google.rpc import status_pb2  # type: ignore[import-untyped]

    status = status_pb2.Status(code=7, message=message)
    return status.SerializeToString()


# the exact message LiveKit Cloud returns (cloud-observability gin.go), always inside a
# google.rpc.Status protobuf body
_DISABLED_MSG = "project data recording is disabled by owner"


@pytest.mark.parametrize(
    "status_code, body, expected",
    [
        # the real wire shape: 401 with a protobuf google.rpc.Status body
        (401, _status_proto(_DISABLED_MSG), True),
        # plain-text body (defensive: if the gateway ever returns text)
        (401, _DISABLED_MSG.encode(), True),
        (403, b"data recording is disabled", True),
        # wrong status code
        (200, _status_proto(_DISABLED_MSG), False),
        (500, _status_proto(_DISABLED_MSG), False),
        # sibling 401s that share the same status/grpc code -> must NOT latch
        (401, _status_proto("missing project id"), False),
        (401, _status_proto("operation requires observability write grant"), False),
        (401, b"", False),
    ],
)
def test_is_disabled_response(status_code: int, body: bytes, expected: bool):
    assert traces._UploadGate.is_disabled_response(status_code, body) is expected


def test_make_ok_response_is_ok():
    resp = traces._AuthRefreshingSession._make_ok_response()
    assert resp.status_code == 200
    assert resp.ok  # OTLP exporters treat this as a successful export


def test_disable_uploads_warns_once_per_session(caplog):
    assert not traces._upload_gate.disabled

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        traces._upload_gate.disable()
        traces._upload_gate.disable()

    assert traces._upload_gate.disabled
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1

    # a new session re-arms the latch and warns again
    traces._upload_gate.reset()
    assert not traces._upload_gate.disabled
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        traces._upload_gate.disable()
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", "ignore")


def test_session_latches_and_warns_then_short_circuits(monkeypatch, caplog):
    calls = {"super": 0}

    def fake_request(self, *args, **kwargs):
        calls["super"] += 1
        return _FakeResponse(401, _status_proto(_DISABLED_MSG))

    # patch the parent's request so no real network/credentials are needed
    monkeypatch.setattr("requests.Session.request", fake_request, raising=True)

    session = traces._AuthRefreshingSession(lambda: {"Authorization": "Bearer x"})

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        first = session.request("POST", "https://example/observability/metrics/otlp/v0")
        # subsequent exports must not hit the network at all
        second = session.request("POST", "https://example/observability/metrics/otlp/v0")
        third = session.request("POST", "https://example/observability/metrics/otlp/v0")

    # OTel sees success every time -> it never logs the 401
    assert first.ok and second.ok and third.ok
    # only the detecting request reached the parent; later ones short-circuited
    assert calls["super"] == 1
    assert traces._upload_gate.disabled
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_session_passes_through_success_and_unrelated_errors(monkeypatch):
    response = _FakeResponse(200, b"ok")

    def fake_request(self, *args, **kwargs):
        return response

    monkeypatch.setattr("requests.Session.request", fake_request, raising=True)

    session = traces._AuthRefreshingSession(lambda: {"Authorization": "Bearer x"})
    assert session.request("POST", "https://example") is response
    assert not traces._upload_gate.disabled

    # an unrelated 401 (e.g. bad token) is returned as-is and does NOT latch
    bad = _FakeResponse(401, b"invalid token")
    monkeypatch.setattr("requests.Session.request", lambda self, *a, **k: bad, raising=True)
    assert session.request("POST", "https://example") is bad
    assert not traces._upload_gate.disabled
