"""Tests for the adaptive-interruption threshold negotiation contract.

The feature is server-driven: the SDK only sends ``threshold`` in ``session.create`` when the user
explicitly overrode it, and otherwise omits the field so the server applies its fetched default.
These tests lock that serialization contract (which breaks silently if someone "cleans up" the
``if is_given(...) else None`` guard or the ``exclude_none=True`` dump) plus the parsing of the
server's ``default_threshold`` off ``session.created``.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import aiohttp
import pytest

from livekit.agents.inference.interruption import (
    AdaptiveInterruptionDetector,
    InterruptionWebSocketStream,
    InterruptionWSSessionCreatedMessage,
    InterruptionWSSessionCreateSettings,
)
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import aio

pytestmark = pytest.mark.unit

CONN_OPTIONS = APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=1.0)


def _make_detector(*, threshold: NotGivenOr[float] = NOT_GIVEN) -> AdaptiveInterruptionDetector:
    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    return AdaptiveInterruptionDetector(
        threshold=threshold,
        base_url="http://localhost:9999",
        api_key="test-key",
        api_secret="test-secret",
        http_session=mock_session,
    )


async def _make_idle_stream(
    detector: AdaptiveInterruptionDetector,
) -> InterruptionWebSocketStream:
    """Create a stream and cancel its live receive loop so we can drive methods directly.

    The stream's __init__ spawns a background task that runs the full WS loop; cancel it before it
    races against the assertions (it would otherwise call _connect_ws itself against the mock).
    """
    stream = detector.stream(conn_options=CONN_OPTIONS)
    await aio.cancel_and_wait(stream._task)
    return stream


async def _capture_session_create(detector: AdaptiveInterruptionDetector) -> dict:
    """Run the real _connect_ws path and return the parsed session.create payload it sent."""
    mock_ws = AsyncMock(spec=aiohttp.ClientWebSocketResponse)
    mock_ws.closed = False
    detector._session.ws_connect = AsyncMock(return_value=mock_ws)  # type: ignore[union-attr]

    stream = await _make_idle_stream(detector)
    try:
        await stream._connect_ws()
    finally:
        await stream.aclose()

    mock_ws.send_str.assert_called_once()
    return json.loads(mock_ws.send_str.call_args.args[0])


class TestSessionCreateThreshold:
    @pytest.mark.asyncio
    async def test_omits_threshold_when_not_given(self) -> None:
        payload = await _capture_session_create(_make_detector())
        assert "threshold" not in payload["settings"], payload

    @pytest.mark.asyncio
    async def test_includes_threshold_when_overridden(self) -> None:
        payload = await _capture_session_create(_make_detector(threshold=0.7))
        assert payload["settings"]["threshold"] == 0.7, payload


class TestSessionCreateSettingsSerialization:
    """The contract at the model level: exclude_none must drop an unset threshold."""

    def test_exclude_none_drops_unset_threshold(self) -> None:
        settings = InterruptionWSSessionCreateSettings(
            sample_rate=16000, num_channels=1, threshold=None, min_frames=2, encoding="s16le"
        )
        dumped = json.loads(settings.model_dump_json(exclude_none=True))
        assert "threshold" not in dumped, dumped

    def test_explicit_threshold_survives_exclude_none(self) -> None:
        settings = InterruptionWSSessionCreateSettings(
            sample_rate=16000, num_channels=1, threshold=0.7, min_frames=2, encoding="s16le"
        )
        dumped = json.loads(settings.model_dump_json(exclude_none=True))
        assert dumped["threshold"] == 0.7, dumped


class TestSessionCreatedDefaultThreshold:
    def test_parses_default_threshold(self) -> None:
        msg = InterruptionWSSessionCreatedMessage.model_validate(
            {"type": "session.created", "default_threshold": 0.42}
        )
        assert msg.default_threshold == 0.42

    def test_default_threshold_optional(self) -> None:
        msg = InterruptionWSSessionCreatedMessage.model_validate({"type": "session.created"})
        assert msg.default_threshold is None


class TestResolveEffectiveThreshold:
    """Observability-only resolution: user override > server default > None."""

    @pytest.mark.asyncio
    async def test_user_override_wins(self) -> None:
        stream = await _make_idle_stream(_make_detector(threshold=0.7))
        try:
            assert stream._resolve_effective_threshold(0.3) == 0.7
        finally:
            await stream.aclose()

    @pytest.mark.asyncio
    async def test_falls_back_to_server_default(self) -> None:
        stream = await _make_idle_stream(_make_detector())
        try:
            assert stream._resolve_effective_threshold(0.3) == 0.3
        finally:
            await stream.aclose()

    @pytest.mark.asyncio
    async def test_returns_none_when_server_silent(self) -> None:
        stream = await _make_idle_stream(_make_detector())
        try:
            assert stream._resolve_effective_threshold(None) is None
        finally:
            await stream.aclose()
