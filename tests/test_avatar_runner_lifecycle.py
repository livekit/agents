"""Tests for AvatarRunner lifecycle edge cases."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents.voice.avatar import AvatarOptions, AvatarRunner


def _mock_room() -> MagicMock:
    """Create a mock Room for runner lifecycle tests."""
    room = MagicMock()
    room.isconnected = MagicMock(return_value=False)
    room.on = MagicMock()
    room.off = MagicMock()
    room.local_participant = MagicMock()
    return room


def _avatar_options() -> AvatarOptions:
    """Create minimal valid avatar options."""
    return AvatarOptions(
        video_width=640,
        video_height=480,
        video_fps=30.0,
        audio_sample_rate=24000,
        audio_channels=1,
    )


@pytest.mark.asyncio
async def test_aclose_before_start_is_safe() -> None:
    """Closing a runner before start() should not raise."""
    room = _mock_room()
    audio_recv = MagicMock()
    audio_recv.aclose = AsyncMock()
    video_gen = MagicMock()

    audio_source = MagicMock()
    audio_source.aclose = AsyncMock()
    video_source = MagicMock()
    video_source.aclose = AsyncMock()
    av_sync = MagicMock()
    av_sync.aclose = AsyncMock()

    with (
        patch("livekit.agents.voice.avatar._runner.rtc.AudioSource", return_value=audio_source),
        patch("livekit.agents.voice.avatar._runner.rtc.VideoSource", return_value=video_source),
        patch("livekit.agents.voice.avatar._runner.rtc.AVSynchronizer", return_value=av_sync),
    ):
        runner = AvatarRunner(
            room,
            audio_recv=audio_recv,
            video_gen=video_gen,
            options=_avatar_options(),
        )

        await runner.aclose()

    audio_recv.aclose.assert_awaited_once()
    av_sync.aclose.assert_awaited_once()
    audio_source.aclose.assert_awaited_once()
    video_source.aclose.assert_awaited_once()
