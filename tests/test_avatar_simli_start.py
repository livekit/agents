from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from livekit.agents import APIConnectionError
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
    failure = APIConnectionError("provider disconnected")
    first = SimpleNamespace(
        text=AsyncMock(return_value='{"session_token": "token"}'),
        raise_for_status=AsyncMock(),
    )
    responses = [failure] if failed_request == 0 else [first, failure]

    @asynccontextmanager
    async def post(*_args, **_kwargs):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        yield response

    http = SimpleNamespace(post=post)
    monkeypatch.setattr(AvatarSession, "_ensure_http_session", lambda _: http)
    avatar = AvatarSession(simli_config=SimliConfig(api_key="key", face_id="face"))
    agent = SimpleNamespace(output=SimpleNamespace(replace_audio_tail=Mock()))

    with pytest.raises(APIConnectionError, match="provider disconnected"):
        await avatar.start(
            agent,
            SimpleNamespace(name="room"),
            livekit_url="wss://example.test",
            livekit_api_key="key",
            livekit_api_secret="a" * 32,
        )

    agent.output.replace_audio_tail.assert_not_called()
    base_start.assert_not_awaited()
