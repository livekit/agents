from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from livekit.plugins.mistralai import tts as mistral_tts
from livekit.plugins.spitch import tts as spitch_tts

pytestmark = pytest.mark.unit


async def test_spitch_close_releases_its_owned_sdk_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock(close=AsyncMock())
    monkeypatch.setattr(spitch_tts, "AsyncSpitch", lambda: client)
    model = spitch_tts.TTS()
    await model.aclose()
    client.close.assert_awaited_once()


async def test_mistral_close_releases_both_owned_sdk_transports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock(__aexit__=AsyncMock(), __exit__=Mock())
    monkeypatch.setattr(mistral_tts, "Mistral", lambda **_: client)
    model = mistral_tts.TTS(api_key="test")
    await model.aclose()
    client.__aexit__.assert_awaited_once_with(None, None, None)
    client.__exit__.assert_called_once_with(None, None, None)


async def test_mistral_close_preserves_an_application_owned_client() -> None:
    client = Mock(__aexit__=AsyncMock(), __exit__=Mock())
    model = mistral_tts.TTS(client=client)
    await model.aclose()
    client.__aexit__.assert_not_awaited()
    client.__exit__.assert_not_called()
