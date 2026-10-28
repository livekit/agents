from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.plugins.avatartalk import AvatarSession
from livekit.plugins.avatartalk.api import AvatarTalkException

pytestmark = pytest.mark.unit


async def test_start_propagates_provider_failure(monkeypatch):
    base_start = AsyncMock()
    monkeypatch.setattr(BaseAvatarSession, "start", base_start)
    monkeypatch.setattr(
        "livekit.plugins.avatartalk.avatar.get_job_context",
        lambda: SimpleNamespace(local_participant_identity="agent", add_shutdown_callback=Mock()),
    )
    avatar = AvatarSession(api_secret="key")
    avatar._avatartalk_api.start_session = AsyncMock(side_effect=AvatarTalkException("rejected"))
    agent = SimpleNamespace(output=SimpleNamespace(replace_audio_tail=Mock()))

    with pytest.raises(AvatarTalkException, match="rejected"):
        await avatar.start(
            agent,
            SimpleNamespace(name="room"),
            livekit_url="wss://example.test",
            livekit_api_key="key",
            livekit_api_secret="a" * 32,
        )

    agent.output.replace_audio_tail.assert_not_called()
    base_start.assert_not_awaited()


async def test_close_terminates_created_session_without_job_shutdown(monkeypatch):
    monkeypatch.setattr(BaseAvatarSession, "aclose", AsyncMock())
    avatar = AvatarSession(api_secret="key")
    avatar.conversation_id = "billable-session"
    avatar._avatartalk_api.stop_session = AsyncMock(return_value={})

    await avatar.aclose()
    await avatar.aclose()

    avatar._avatartalk_api.stop_session.assert_awaited_once_with("billable-session")
    assert avatar.conversation_id is None
