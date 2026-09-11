import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from livekit import rtc
from livekit.agents.utils.robust_microphone import RobustMicrophone

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_robust_microphone_startup_and_shutdown():
    with patch("livekit.rtc.MediaDevices", autospec=True) as mock_media_devices, \
         patch("livekit.rtc.AudioSource", autospec=True) as mock_audio_source, \
         patch("livekit.rtc.LocalAudioTrack.create_audio_track", autospec=True) as mock_create_track, \
         patch("livekit.rtc.AudioStream.from_track", autospec=True) as mock_from_track:

        # Mock the stream iterator to just hang (we'll shut down before it times out)
        mock_stream = AsyncMock()
        mock_stream.__anext__.side_effect = asyncio.TimeoutError
        mock_from_track.return_value = mock_stream

        mic = RobustMicrophone(stall_timeout=10.0)
        mic.start()
        
        # Give it a moment to start the internal stream
        await asyncio.sleep(0.1)
        
        assert mock_media_devices.called
        assert mock_create_track.called
        assert mock_from_track.called

        await mic.aclose()


@pytest.mark.asyncio
async def test_robust_microphone_restarts_on_stall():
    with patch("livekit.rtc.MediaDevices", autospec=True) as mock_media_devices, \
         patch("livekit.rtc.AudioSource", autospec=True) as mock_audio_source, \
         patch("livekit.rtc.LocalAudioTrack.create_audio_track", autospec=True) as mock_create_track, \
         patch("livekit.rtc.AudioStream.from_track", autospec=True) as mock_from_track:

        mock_stream = AsyncMock()
        mock_stream.aclose = AsyncMock()
        
        # First call times out immediately, causing a restart
        # Second call hangs so we can shut down
        call_count = 0
        async def mock_anext():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError()
            else:
                await asyncio.sleep(10)
                return MagicMock()

        mock_stream.__anext__ = mock_anext
        mock_from_track.return_value = mock_stream

        mic = RobustMicrophone(stall_timeout=0.1)
        mic.start()

        # Wait for the timeout and restart to happen
        await asyncio.sleep(0.5)

        # MediaDevices.open_input should have been called twice (initial + restart)
        assert mock_from_track.call_count >= 2

        await mic.aclose()
