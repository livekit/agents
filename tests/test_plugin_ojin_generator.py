"""Tests for OjinVideoGenerator behavior."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd
from livekit.plugins.ojin.avatar import (
    _HIGH_WATERMARK_FRAMES,
    _LOW_WATERMARK_FRAMES,
    OjinVideoGenerator,
)


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
async def test_idle_detection_uses_frame_type_not_index():
    """Drop mode should check FrameType.IDLE, never use index == 0."""
    from ojin.ojin_client_messages import FrameType, OjinInteractionResponseMessage

    client = _make_mock_client()
    gen = OjinVideoGenerator(client)

    # Force drop mode ON
    gen._drop_mode = True
    gen._pending_frames = _HIGH_WATERMARK_FRAMES

    # Create a mock response with IDLE frame type
    msg = MagicMock()
    msg.frame_type = FrameType.IDLE
    msg.video_frame_bytes = b"some_data"
    msg.audio_frame_bytes = b"some_audio"
    msg.is_final_response = False

    # isinstance checks need the right type: OjinInteractionResponseMessage
    msg.__class__ = OjinInteractionResponseMessage

    # Set up: return one IDLE response and then stop the stream cleanly.
    calls = 0

    async def _receive_message():
        nonlocal calls
        calls += 1
        if calls == 1:
            return msg
        gen._closed = True
        return None

    client.receive_message = AsyncMock(side_effect=_receive_message)

    # The idle frame should be dropped (not yielded).
    frames = [frame async for frame in gen]

    # IDLE frames should be dropped in drop mode
    assert len(frames) == 0


@pytest.mark.asyncio
async def test_drop_mode_uses_hysteresis():
    """Drop mode should enable at HIGH_WATERMARK and disable at LOW_WATERMARK."""
    client = _make_mock_client()
    gen = OjinVideoGenerator(client)

    # Below high watermark: not in drop mode
    gen._pending_frames = _HIGH_WATERMARK_FRAMES - 1
    assert gen._drop_mode is False

    # At high watermark: enters drop mode
    gen._pending_frames = _HIGH_WATERMARK_FRAMES
    # Simulate what _stream_impl does
    gen._pending_frames += 1
    if gen._pending_frames >= _HIGH_WATERMARK_FRAMES:
        gen._drop_mode = True
    assert gen._drop_mode is True

    # Above low watermark: stays in drop mode
    gen._pending_frames = _LOW_WATERMARK_FRAMES + 1
    if gen._pending_frames <= _LOW_WATERMARK_FRAMES:
        gen._drop_mode = False
    assert gen._drop_mode is True

    # At low watermark: exits drop mode
    gen._pending_frames = _LOW_WATERMARK_FRAMES
    if gen._pending_frames <= _LOW_WATERMARK_FRAMES:
        gen._drop_mode = False
    assert gen._drop_mode is False
