"""Tests for Sarvam keyterm prompting (``saaras:v4`` only).

Keyterms are sent as one JSON-encoded array — a query parameter on the
WebSocket URLs and a form field on the REST endpoint — and Sarvam limits them
to 50 distinct terms of 64 characters each. They are only applied for
``saaras:v4``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import pytest

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins.sarvam import stt as sarvam_stt
from livekit.plugins.sarvam.stt import (
    MAX_KEYTERM_LENGTH,
    MAX_KEYTERMS,
    STT,
    SarvamSTTOptions,
    SpeechStream,
    _build_websocket_url,
)
from livekit.plugins.sarvam.stt_streaming import (
    RealtimeSpeechStream,
    RealtimeSTTOptions,
    STTRealtime,
    _build_realtime_ws_url,
)

pytestmark = pytest.mark.unit

SARVAM_WS_URL = "wss://api.sarvam.ai/speech-to-text/ws"
SARVAM_REALTIME_WS_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"


def _query(url: str) -> dict[str, list[str]]:
    return parse_qs(urlparse(url).query)


def _legacy_opts(**kwargs: Any) -> SarvamSTTOptions:
    return SarvamSTTOptions(language="en-IN", api_key="test-key", **kwargs)


def _realtime_opts(**kwargs: Any) -> RealtimeSTTOptions:
    return RealtimeSTTOptions(language="en-IN", api_key="test-key", **kwargs)


# ---------------------------------------------------------------------------
# WebSocket URL — legacy streaming endpoint
# ---------------------------------------------------------------------------


def test_ws_url_includes_keyterms_for_saaras_v4() -> None:
    """The keyterms query parameter is one JSON-encoded array, as Sarvam expects."""
    opts = _legacy_opts(model="saaras:v4", keyterms=["Sarvam", "New Delhi"])
    query = _query(_build_websocket_url(SARVAM_WS_URL, opts))
    assert json.loads(query["keyterms"][0]) == ["Sarvam", "New Delhi"]


def test_ws_url_omits_keyterms_for_default_model() -> None:
    """Keyterm prompting is saaras:v4 only, so other models must not send it."""
    opts = _legacy_opts(model="saaras:v3", keyterms=["Sarvam"])
    assert "keyterms" not in _query(_build_websocket_url(SARVAM_WS_URL, opts))


def test_ws_url_omits_keyterms_when_not_configured() -> None:
    opts = _legacy_opts(model="saaras:v4")
    assert "keyterms" not in _query(_build_websocket_url(SARVAM_WS_URL, opts))


# ---------------------------------------------------------------------------
# WebSocket URL — realtime endpoint
# ---------------------------------------------------------------------------


def test_realtime_ws_url_includes_keyterms_for_saaras_v4() -> None:
    opts = _realtime_opts(model="saaras:v4", keyterms=["PhonePe"])
    query = _query(_build_realtime_ws_url(SARVAM_REALTIME_WS_URL, opts))
    assert json.loads(query["keyterms"][0]) == ["PhonePe"]


def test_realtime_ws_url_omits_keyterms_for_default_model() -> None:
    opts = _realtime_opts(keyterms=["PhonePe"])
    assert "keyterms" not in _query(_build_realtime_ws_url(SARVAM_REALTIME_WS_URL, opts))


def test_realtime_accepts_saaras_v4_and_rejects_unknown_models() -> None:
    opts = _realtime_opts(model="saaras:v4")
    assert opts.model == "saaras:v4"
    with pytest.raises(ValueError, match="model must be one of"):
        _realtime_opts(model="saaras:v9")


# ---------------------------------------------------------------------------
# Validation against Sarvam's limits
# ---------------------------------------------------------------------------


def test_keyterms_are_deduplicated_preserving_order() -> None:
    opts = _legacy_opts(keyterms=["Sarvam", "New Delhi", "Sarvam"])
    assert opts.keyterms == ["Sarvam", "New Delhi"]


def test_too_many_keyterms_rejected() -> None:
    keyterms = [f"term-{i}" for i in range(MAX_KEYTERMS + 1)]
    with pytest.raises(ValueError, match=str(MAX_KEYTERMS)):
        _legacy_opts(keyterms=keyterms)


def test_overlong_keyterm_rejected() -> None:
    with pytest.raises(ValueError, match=str(MAX_KEYTERM_LENGTH)):
        _legacy_opts(keyterms=["x" * (MAX_KEYTERM_LENGTH + 1)])


def test_max_length_keyterm_accepted() -> None:
    opts = _legacy_opts(keyterms=["x" * MAX_KEYTERM_LENGTH])
    assert opts.keyterms == ["x" * MAX_KEYTERM_LENGTH]


def test_realtime_rejects_too_many_keyterms() -> None:
    keyterms = [f"term-{i}" for i in range(MAX_KEYTERMS + 1)]
    with pytest.raises(ValueError, match=str(MAX_KEYTERMS)):
        _realtime_opts(keyterms=keyterms)


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_keyterm_capability_matches_model() -> None:
    assert STT(api_key="test-key", model="saaras:v4").capabilities.keyterms is True
    assert STT(api_key="test-key", model="saaras:v3").capabilities.keyterms is False
    assert STTRealtime(api_key="test-key", model="saaras:v4").capabilities.keyterms is True
    assert STTRealtime(api_key="test-key").capabilities.keyterms is False


# ---------------------------------------------------------------------------
# REST endpoint
# ---------------------------------------------------------------------------


class _MockResponse:
    status = 200

    async def json(self) -> dict[str, Any]:
        return {"transcript": "namaste", "request_id": "req-test"}

    async def text(self) -> str:
        return ""


class _MockPostContext:
    async def __aenter__(self) -> _MockResponse:
        return _MockResponse()

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _RecordingFormData:
    """Records the fields the plugin adds before aiohttp serializes them."""

    def __init__(self) -> None:
        self.fields: dict[str, Any] = {}

    def add_field(self, name: str, value: Any, **kwargs: Any) -> None:
        self.fields[name] = value


class _MockSession:
    """Captures the form data of the last REST request."""

    def __init__(self) -> None:
        self.form_data: _RecordingFormData | None = None

    def post(self, **kwargs: Any) -> _MockPostContext:
        self.form_data = kwargs["data"]
        return _MockPostContext()


async def _recognize_with_mock(
    monkeypatch: pytest.MonkeyPatch, instance: STT, session: _MockSession
) -> dict[str, Any]:
    monkeypatch.setattr(sarvam_stt.aiohttp, "FormData", _RecordingFormData)
    instance._session = session  # type: ignore[assignment]
    frame = rtc.AudioFrame.create(sample_rate=16000, num_channels=1, samples_per_channel=160)
    await instance._recognize_impl([frame])
    assert session.form_data is not None
    return session.form_data.fields


async def test_rest_sends_keyterms_as_one_json_field(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = STT(api_key="test-key", model="saaras:v4", keyterms=["Sarvam", "New Delhi"])

    fields = await _recognize_with_mock(monkeypatch, instance, _MockSession())

    assert fields["keyterms"] == json.dumps(["Sarvam", "New Delhi"])


async def test_rest_omits_keyterms_for_unsupported_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = STT(api_key="test-key", model="saaras:v3", keyterms=["Sarvam"])

    fields = await _recognize_with_mock(monkeypatch, instance, _MockSession())

    assert "keyterms" not in fields


# ---------------------------------------------------------------------------
# Framework-managed session keyterms
# ---------------------------------------------------------------------------


class _FakeStream:
    def __init__(self) -> None:
        self.updated: list[list[str]] = []

    def _update_keyterms(self, keyterms: list[str]) -> None:
        self.updated.append(list(keyterms))


def test_update_session_keyterms_merges_with_user_terms() -> None:
    instance = STT(api_key="test-key", model="saaras:v4", keyterms=["Sarvam"])
    stream = _FakeStream()
    instance._streams = {stream}  # type: ignore[assignment]

    instance._update_session_keyterms(["New Delhi", "Sarvam"])

    assert instance._opts.keyterms == ["Sarvam", "New Delhi"]
    assert stream.updated == [["Sarvam", "New Delhi"]]


def test_update_session_keyterms_caps_overlong_sets() -> None:
    user_terms = [f"term-{i}" for i in range(MAX_KEYTERMS)]
    instance = STT(api_key="test-key", model="saaras:v4", keyterms=user_terms)
    instance._streams = set()  # type: ignore[assignment]

    # The framework cannot know the provider limit, so the extra term is
    # dropped instead of failing the session.
    instance._update_session_keyterms(["one-too-many"])

    assert instance._opts.keyterms == user_terms


def test_update_session_keyterms_does_not_interrupt_a_live_stream() -> None:
    """Framework keyterms are recorded for the next stream; the live one keeps flowing."""
    instance = STT(api_key="test-key", model="saaras:v4")
    stream = SpeechStream.__new__(SpeechStream)
    stream._opts = SimpleNamespace(keyterms=None)  # type: ignore[attr-defined]
    stream._logger = MagicMock()  # type: ignore[attr-defined]
    stream._build_log_context = lambda: {}  # type: ignore[attr-defined]
    instance._streams = {stream}  # type: ignore[assignment]

    instance._update_session_keyterms(["Sarvam"])

    # Reconnecting would end this single-attempt stream mid-call; a bare
    # instance has no `_reconnect_event`, so touching it would raise.
    assert stream._opts.keyterms == ["Sarvam"]
    assert stream._logger.debug.called


def test_update_session_keyterms_ignored_without_capability() -> None:
    instance = STT(api_key="test-key", model="saaras:v3")

    instance._update_session_keyterms(["Sarvam"])

    assert instance._opts.keyterms is None


def test_realtime_update_options_sets_keyterms_for_new_streams() -> None:
    instance = STTRealtime(api_key="test-key", model="saaras:v4")

    instance.update_options(keyterms=["Sarvam"])

    assert instance._opts.keyterms == ["Sarvam"]


def test_realtime_explicit_keyterms_survive_session_updates() -> None:
    """A session keyterm change must merge with an explicit update, not revert it."""
    instance = STTRealtime(api_key="test-key", model="saaras:v4")

    instance.update_options(keyterms=["Sarvam"])
    instance._update_session_keyterms(["New Delhi"])

    assert instance._opts.keyterms == ["Sarvam", "New Delhi"]


def test_realtime_explicit_keyterms_merge_with_active_session_terms() -> None:
    instance = STTRealtime(api_key="test-key", model="saaras:v4")

    instance._update_session_keyterms(["New Delhi"])
    instance.update_options(keyterms=["Sarvam"])

    assert instance._opts.keyterms == ["Sarvam", "New Delhi"]


async def test_realtime_live_stream_retains_keyterms() -> None:
    """Keyterms are fixed when the connection opens, so a live stream keeps its set."""
    instance = STTRealtime(api_key="test-key", model="saaras:v4")
    stream = RealtimeSpeechStream(
        stt=instance,
        opts=instance._opts,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        http_session=MagicMock(),
    )

    stream.update_options(keyterms=["Sarvam"])

    assert stream._opts.keyterms is None
