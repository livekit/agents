"""Tests for OjinVideoGenerator behavior."""

import asyncio
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
    client.start_interaction = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_audio_segment_end_keeps_interaction_open():
    """AudioSegmentEnd should not stop Ojin's continuous idle stream."""
    client = _make_mock_client()
    gen = OjinVideoGenerator(client)
    gen._interaction_started = True

    await gen.push_audio(AudioSegmentEnd())

    client.send_message.assert_not_awaited()
    client.start_interaction.assert_not_awaited()
    assert gen._interaction_started is True


@pytest.mark.asyncio
async def test_start_idle_stream_sends_one_silent_frame():
    """Starting idle mode should prime Ojin with one 25fps silent audio frame."""
    client = _make_mock_client()
    gen = OjinVideoGenerator(client)

    with patch("livekit.plugins.ojin.avatar.OjinAudioInputMessage") as MockAudioMsg:
        mock_instance = MockAudioMsg.return_value

        await gen.start_idle_stream()

        MockAudioMsg.assert_called_once_with(audio_int16_bytes=b"\x00" * 1280)
        client.start_interaction.assert_awaited_once()
        client.send_message.assert_awaited_once_with(mock_instance)

        MockAudioMsg.reset_mock()
        client.start_interaction.reset_mock()
        client.send_message.reset_mock()

        await gen.start_idle_stream()

        MockAudioMsg.assert_not_called()
        client.start_interaction.assert_not_awaited()
        client.send_message.assert_not_awaited()


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
        client.start_interaction.assert_awaited_once()

        client.start_interaction.reset_mock()
        await gen.push_audio(frame)
        client.start_interaction.assert_not_awaited()


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
            return None
        gen._closed = True
        return None

    client.receive_message = AsyncMock(side_effect=_receive_message)
    gen._interrupted = True

    frames = [frame async for frame in gen]

    assert len(frames) == 1
    assert isinstance(frames[0], AudioSegmentEnd)
    assert gen._interrupted is False


@pytest.mark.asyncio
async def test_aclose_interrupts_pending_receive_message():
    """aclose() should stop an active stream even without closing the client."""
    client = _make_mock_client()
    gen = OjinVideoGenerator(client)

    receive_started = asyncio.Event()
    receive_cancelled = asyncio.Event()

    async def _receive_message():
        receive_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            receive_cancelled.set()
            raise

    client.receive_message = AsyncMock(side_effect=_receive_message)

    next_frame_task = asyncio.create_task(anext(gen.__aiter__()))
    await receive_started.wait()

    await gen.aclose()

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(next_frame_task, timeout=1.0)

    assert receive_cancelled.is_set()
