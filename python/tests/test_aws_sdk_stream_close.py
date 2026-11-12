from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from livekit.plugins.aws.experimental.realtime import RealtimeModel, RealtimeSession

pytestmark = pytest.mark.unit


async def test_nova_close_uses_the_sdk_stream_close_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RealtimeSession, "initialize_streams", AsyncMock())
    model = RealtimeModel()
    session = model.session()
    input_stream = SimpleNamespace(send=AsyncMock(), close=AsyncMock())
    output_stream = SimpleNamespace(close=AsyncMock())
    session._stream_response = SimpleNamespace(
        input_stream=input_stream, output_stream=output_stream
    )
    session._is_sess_active.set()
    try:
        await asyncio.sleep(0)
        await session.aclose()
        input_stream.close.assert_awaited_once()
        output_stream.close.assert_awaited_once()
    finally:
        await model.aclose()
