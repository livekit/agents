"""Tests for OjinVideoGenerator behavior."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd
from livekit.plugins.ojin.avatar import OjinVideoGenerator


def _make_mock_client():
    """Create a mock OjinClient."""
    client = MagicMock()
    client.send_message = AsyncMock()
    client.receive_message = AsyncMock()
    client.start_interaction = MagicMock()
    return client


@pytest.mark.asyncio
async def test_audio_segment_end_calls_end_interaction():
    """Pushing AudioSegmentEnd should send OjinEndInteractionMessage."""
    client = _make_mock_client()
    gen = OjinVideoGenerator(client)

    with patch("livekit.plugins.ojin.avatar.OjinEndInteractionMessage") as MockEndMsg:
        mock_instance = MockEndMsg.return_value
        await gen.push_audio(AudioSegmentEnd())
        client.send_message.assert_awaited_once_with(mock_instance)


@pytest.mark.asyncio
async def test_push_audio_frame_sends_audio_input():
    """Pushing an AudioFrame should send OjinAudioInputMessage with audio_int16_bytes."""
    client = _make_mock_client()
    gen = OjinVideoGenerator(client)

    frame = rtc.AudioFrame(
        data=b"\x00\x00" * 320,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=320,
    )

    with patch("livekit.plugins.ojin.avatar.OjinAudioInputMessage") as MockAudioMsg:
        mock_instance = MockAudioMsg.return_value
        await gen.push_audio(frame)
        MockAudioMsg.assert_called_once_with(audio_int16_bytes=bytes(frame.data))
        client.send_message.assert_awaited_once_with(mock_instance)


@pytest.mark.asyncio
async def test_push_audio_starts_interaction():
    """First AudioFrame push should call start_interaction()."""
    client = _make_mock_client()
    gen = OjinVideoGenerator(client)

    frame = rtc.AudioFrame(
        data=b"\x00\x00" * 320,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=320,
    )

    with patch("livekit.plugins.ojin.avatar.OjinAudioInputMessage"):
        await gen.push_audio(frame)
        client.start_interaction.assert_called_once()

        # Second push should NOT call start_interaction again
        client.start_interaction.reset_mock()
        await gen.push_audio(frame)
        client.start_interaction.assert_not_called()


@pytest.mark.asyncio
async def test_interrupted_yields_audio_segment_end():
    """After clear_buffer, _stream_impl should yield AudioSegmentEnd when client returns None."""
    client = _make_mock_client()
    gen = OjinVideoGenerator(client)

    calls = 0

    async def _receive_message():
        nonlocal calls
        calls += 1
        if calls == 1:
            return None  # simulates cancelled client after clear_buffer
        gen._closed = True
        return None

    client.receive_message = AsyncMock(side_effect=_receive_message)
    gen._interrupted = True  # simulates clear_buffer having been called

    frames = [frame async for frame in gen]

    assert len(frames) == 1
    assert isinstance(frames[0], AudioSegmentEnd)
    assert gen._interrupted is False  # flag cleared after delivery
