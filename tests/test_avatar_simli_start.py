from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest

from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.plugins.simli import AvatarSession, SimliConfig

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("failed_request", [0, 1])
async def test_start_propagates_provider_failure(monkeypatch, failed_request):
    base_start = AsyncMock()
    monkeypatch.setattr(BaseAvatarSession, "start", base_start)
    monkeypatch.setattr(
        "livekit.plugins.simli.avatar.get_job_context",
        lambda: SimpleNamespace(local_participant_identity="agent"),
    )
    failure = aiohttp.ClientConnectionError("provider disconnected")
    first = SimpleNamespace(text=AsyncMock(return_value='{"session_token": "token"}'))
    first.raise_for_status = Mock()
    http = SimpleNamespace(post=AsyncMock(side_effect=[first, first]))
    http.post.side_effect = [failure] if failed_request == 0 else [first, failure]
    monkeypatch.setattr(AvatarSession, "_ensure_http_session", lambda _: http)
    avatar = AvatarSession(simli_config=SimliConfig(api_key="key", face_id="face"))
    agent = SimpleNamespace(output=SimpleNamespace(replace_audio_tail=Mock()))

    with pytest.raises(aiohttp.ClientConnectionError, match="provider disconnected"):
        await avatar.start(
            agent,
            SimpleNamespace(name="room"),
            livekit_url="wss://example.test",
            livekit_api_key="key",
            livekit_api_secret="a" * 32,
        )

    agent.output.replace_audio_tail.assert_not_called()
    base_start.assert_not_awaited()
