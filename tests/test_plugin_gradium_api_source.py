"""The Gradium STT websocket identifies the plugin to the API through `x-api-source`."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS

pytestmark = pytest.mark.unit

# Pinned on purpose rather than imported: the value is what the API records, so a
# rename in the plugin must fail here.
API_SOURCE = "livekit-plugin"


async def test_stt_sends_api_source_header():
    from livekit.plugins.gradium import STT
    from livekit.plugins.gradium.stt import SpeechStream, STTOptions

    recorded: dict[str, Any] = {}

    class FakeWebSocket:
        async def send_str(self, message: str) -> None:
            pass

    class FakeSession:
        async def ws_connect(self, url: str, **kwargs: Any) -> FakeWebSocket:
            recorded["headers"] = kwargs["headers"]
            return FakeWebSocket()

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    stt = STT(api_key="test-key")
    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = SpeechStream(
            stt=stt,
            opts=STTOptions(),
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            api_key="test-key",
            model_endpoint="wss://api.gradium.ai/api/speech/asr",
            model_name="default",
            http_session=FakeSession(),  # type: ignore[arg-type]
        )
    await stream._connect_ws()

    assert recorded["headers"] == {"x-api-key": "test-key", "x-api-source": API_SOURCE}
