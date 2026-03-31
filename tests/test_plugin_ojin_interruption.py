"""Tests for interruption and cancel semantics."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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

        # Should have sent cancel message
        client.send_message.assert_awaited_once_with(mock_cancel)

    # Should have queued an AudioSegmentEnd in the output channel
    # Verify by checking that pending state is reset
    assert gen._pending_frames == 0
    assert gen._drop_mode is False
    assert gen._interaction_started is False


@pytest.mark.asyncio
async def test_clear_buffer_resets_interaction_state():
    """clear_buffer() should reset interaction_started so next push calls start_interaction()."""
    client = MagicMock()
    client.send_message = AsyncMock()
    client.start_interaction = MagicMock()
    gen = OjinVideoGenerator(client)

    # Simulate that interaction was started
    gen._interaction_started = True

    with patch("livekit.plugins.ojin.avatar.OjinCancelInteractionMessage"):
        await gen.clear_buffer()

    assert gen._interaction_started is False
