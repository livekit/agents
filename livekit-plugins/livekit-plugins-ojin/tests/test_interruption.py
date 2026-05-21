"""Tests for interruption and cancel semantics."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ojin.ojin_client_messages import OjinInteractionResponseMessage

from livekit.agents.voice.avatar import AudioSegmentEnd
from livekit.plugins.ojin.avatar import OjinVideoGenerator


@pytest.mark.asyncio
async def test_clear_buffer_triggers_cancel_and_immediate_segment_end():
    """clear_buffer() should send CancelInteractionMessage and immediately yield AudioSegmentEnd."""
    client = MagicMock()
    client.send_message = AsyncMock()
    gen = OjinVideoGenerator(client)

    with patch("livekit.plugins.ojin.avatar.OjinCancelInteractionMessage") as MockCancelMsg:
        mock_cancel = MockCancelMsg.return_value
        await gen.clear_buffer()

        client.send_message.assert_awaited_once_with(mock_cancel)

    assert gen._interaction_started is False
    assert gen._interrupted is True


@pytest.mark.asyncio
async def test_clear_buffer_resets_interaction_state():
    """clear_buffer() should reset interaction_started so next push calls start_interaction()."""
    client = MagicMock()
    client.send_message = AsyncMock()
    client.start_interaction = MagicMock()
    gen = OjinVideoGenerator(client)

    gen._interaction_started = True

    with patch("livekit.plugins.ojin.avatar.OjinCancelInteractionMessage"):
        await gen.clear_buffer()

    assert gen._interaction_started is False


@pytest.mark.asyncio
async def test_clear_buffer_drains_pending_client_audio():
    """clear_buffer() should discard queued outbound audio from interrupted speech."""
    client = MagicMock()
    client.send_message = AsyncMock()
    client._pending_client_messages_queue = asyncio.Queue()
    await client._pending_client_messages_queue.put("old-audio-1")
    await client._pending_client_messages_queue.put("old-audio-2")
    gen = OjinVideoGenerator(client)

    with patch("livekit.plugins.ojin.avatar.OjinCancelInteractionMessage"):
        await gen.clear_buffer()

    assert client._pending_client_messages_queue.empty()


@pytest.mark.asyncio
async def test_interrupted_stream_drops_stale_interaction_frames():
    """Frames from the cancelled interaction should not be forwarded after interruption."""
    client = MagicMock()
    client.send_message = AsyncMock()
    gen = OjinVideoGenerator(client)
    gen._drop_interaction_id = "cancelled-interaction"

    stale_frame = OjinInteractionResponseMessage(
        interaction_id="cancelled-interaction",
        video_frame_bytes=b"",
        audio_frame_bytes=b"",
        is_final_response=True,
        index=1,
    )
    next_frame = OjinInteractionResponseMessage(
        interaction_id="new-interaction",
        video_frame_bytes=b"",
        audio_frame_bytes=b"",
        is_final_response=True,
        index=1,
    )
    client.receive_message = AsyncMock(side_effect=[stale_frame, next_frame])

    frame = await anext(gen.__aiter__())
    await gen.aclose()

    assert isinstance(frame, AudioSegmentEnd)
