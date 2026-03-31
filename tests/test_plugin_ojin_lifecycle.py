"""Tests for AvatarSession start/aclose lifecycle."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.plugins.ojin.avatar import AvatarSession


def _mock_agent_session():
    """Create a mock AgentSession."""
    session = MagicMock()
    session.output = MagicMock()
    session.output.audio = None
    return session


def _mock_room():
    """Create a mock Room."""
    room = MagicMock()
    room.isconnected = MagicMock(return_value=True)
    room.name = "test-room"
    return room


def _mock_ojin_client():
    """Create a mock OjinClient that returns sessionReady on first receive."""
    client = MagicMock()
    client.connect = AsyncMock()
    client.close = AsyncMock()
    client.send_message = AsyncMock()
    client.start_interaction = MagicMock()

    # First receive_message returns sessionReady
    from ojin.ojin_client_messages import OjinSessionReadyMessage

    session_ready_msg = MagicMock(spec=OjinSessionReadyMessage)
    session_ready_msg.parameters = None
    client.receive_message = AsyncMock(return_value=session_ready_msg)
    return client, session_ready_msg


@pytest.mark.asyncio
async def test_start_sets_queue_audio_output():
    """After start(), agent_session.output.audio should be a QueueAudioOutput."""
    from livekit.agents.voice.avatar import QueueAudioOutput

    avatar = AvatarSession(api_key="test-key", config_id="test-config")
    agent_session = _mock_agent_session()
    room = _mock_room()

    client, session_ready_msg = _mock_ojin_client()

    with (
        patch("ojin.ojin_client.OjinClient", return_value=client),
        patch("livekit.plugins.ojin.avatar.AvatarRunner") as MockRunner,
    ):
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.start = AsyncMock()

        await avatar.start(agent_session, room)

        # agent_session.output.audio should now be a QueueAudioOutput
        assert isinstance(agent_session.output.audio, QueueAudioOutput)


@pytest.mark.asyncio
async def test_aclose_is_idempotent():
    """Double-close should not raise."""
    avatar = AvatarSession(api_key="test-key", config_id="test-config")

    # aclose on a session that was never started
    await avatar.aclose()
    # second aclose should also be safe
    await avatar.aclose()


@pytest.mark.asyncio
async def test_aclose_after_start():
    """aclose() after start() should clean up runner, generator, and client."""
    avatar = AvatarSession(api_key="test-key", config_id="test-config")
    agent_session = _mock_agent_session()
    room = _mock_room()

    client, session_ready_msg = _mock_ojin_client()

    with (
        patch("ojin.ojin_client.OjinClient", return_value=client),
        patch("livekit.plugins.ojin.avatar.AvatarRunner") as MockRunner,
    ):
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.start = AsyncMock()
        mock_runner_instance.aclose = AsyncMock()

        await avatar.start(agent_session, room)
        await avatar.aclose()

        mock_runner_instance.aclose.assert_awaited_once()
        client.close.assert_awaited_once()

        assert avatar._avatar_runner is None
        assert avatar._generator is None
        assert avatar._client is None
