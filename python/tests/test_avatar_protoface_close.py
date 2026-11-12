from unittest.mock import AsyncMock

import pytest

from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.plugins.protoface import AvatarSession

pytestmark = pytest.mark.unit


async def test_close_retains_session_until_termination_succeeds(monkeypatch):
    monkeypatch.setattr(BaseAvatarSession, "aclose", AsyncMock())
    avatar = AvatarSession(api_key="key")
    avatar._session_id = "billable-session"
    avatar._api.end_session = AsyncMock(side_effect=[OSError("provider unavailable"), None])

    await avatar.aclose()
    assert avatar.session_id == "billable-session"
    await avatar.aclose()
    assert avatar.session_id is None
    assert avatar._api.end_session.await_args_list[0].args == ("billable-session",)
    assert avatar._api.end_session.await_args_list[1].args == ("billable-session",)
