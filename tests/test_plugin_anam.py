"""Tests for the Anam plugin's AvatarSession.

Covers what ``start()`` hands to the agent and to the Anam API: the audio output that
replaces the session's audio tail, and the LiveKit token the avatar joins with. The
Anam API and the base ``AvatarSession.start`` are faked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest

from livekit import api, rtc
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.plugins.anam import AnamException, PersonaConfig, avatar

pytestmark = pytest.mark.unit

LIVEKIT_API_KEY = "test-key"
LIVEKIT_API_SECRET = "test-secret-long-enough-for-hs256-signing"
PERSONA = PersonaConfig(name="test", avatarId="avatar-id")


class _FakeAnamAPI:
    """Records the start_session call and answers with a session id."""

    calls: list[dict[str, Any]] = []

    def __init__(self, **_: Any) -> None:
        pass

    async def __aenter__(self) -> _FakeAnamAPI:
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    async def start_session(self, **kwargs: Any) -> dict[str, Any]:
        _FakeAnamAPI.calls.append(kwargs)
        return {"sessionId": "session-1"}


@pytest.fixture
def output_cls(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    _FakeAnamAPI.calls = []
    output_cls = MagicMock()
    monkeypatch.setattr(BaseAvatarSession, "start", AsyncMock())
    monkeypatch.setattr(avatar, "AnamAPI", _FakeAnamAPI)
    monkeypatch.setattr(avatar, "DataStreamAudioOutput", output_cls)
    monkeypatch.setattr(
        avatar, "get_job_context", lambda: SimpleNamespace(local_participant_identity="agent")
    )
    return output_cls


async def _start(session: avatar.AvatarSession) -> MagicMock:
    agent_session = MagicMock()
    room = MagicMock()
    room.name = "room"
    await session.start(
        agent_session,
        room,
        livekit_url="wss://example.livekit.cloud",
        livekit_api_key=LIVEKIT_API_KEY,
        livekit_api_secret=LIVEKIT_API_SECRET,
    )
    return agent_session


async def test_start_waits_for_playback_started(output_cls: MagicMock) -> None:
    session = avatar.AvatarSession(persona_config=PERSONA, api_key="anam-key")
    agent_session = await _start(session)

    output_cls.assert_called_once()
    kwargs = output_cls.call_args.kwargs
    # the avatar reports lk.playback_started, so the agent waits for it
    assert kwargs["wait_playback_start"] is True
    assert kwargs["destination_identity"] == "anam-avatar-agent"
    assert kwargs["sample_rate"] == avatar.SAMPLE_RATE
    assert kwargs["wait_remote_track"] == rtc.TrackKind.KIND_VIDEO
    # the agent's default clear_buffer_timeout stays the backstop for a lost finish
    assert "clear_buffer_timeout" not in kwargs
    agent_session.output.replace_audio_tail.assert_called_once_with(output_cls.return_value)
    assert session.session_id == "session-1"


async def test_start_sends_the_avatar_to_a_custom_identity(output_cls: MagicMock) -> None:
    session = avatar.AvatarSession(
        persona_config=PERSONA, api_key="anam-key", avatar_participant_identity="my-avatar"
    )
    await _start(session)

    assert output_cls.call_args.kwargs["destination_identity"] == "my-avatar"
    assert session.avatar_identity == "my-avatar"


async def test_start_mints_a_publish_on_behalf_token(output_cls: MagicMock) -> None:
    session = avatar.AvatarSession(persona_config=PERSONA, api_key="anam-key")
    await _start(session)

    (call,) = _FakeAnamAPI.calls
    assert call["persona_config"] is PERSONA
    assert call["livekit_url"] == "wss://example.livekit.cloud"
    token = call["livekit_token"]
    claims = api.TokenVerifier(LIVEKIT_API_KEY, LIVEKIT_API_SECRET).verify(token)
    assert claims.identity == "anam-avatar-agent"
    # TokenVerifier does not parse the participant kind back, so read the raw payload
    payload = jwt.decode(token, LIVEKIT_API_SECRET, algorithms=["HS256"])
    assert payload["kind"] == "agent"
    assert claims.video is not None
    assert claims.video.room_join is True
    assert claims.video.room == "room"
    assert claims.attributes == {ATTRIBUTE_PUBLISH_ON_BEHALF: "agent"}


def test_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANAM_API_KEY", raising=False)
    with pytest.raises(AnamException):
        avatar.AvatarSession(persona_config=PERSONA)


async def test_start_requires_livekit_credentials(
    output_cls: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    session = avatar.AvatarSession(persona_config=PERSONA, api_key="anam-key")
    with pytest.raises(AnamException):
        await session.start(MagicMock(), MagicMock())
    output_cls.assert_not_called()
