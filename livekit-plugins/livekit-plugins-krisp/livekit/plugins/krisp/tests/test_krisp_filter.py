"""Unit tests for Krisp VIVA filter.

Note: These tests require the krisp_audio package and a valid model file.
Set KRISP_VIVA_MODEL_PATH environment variable before running tests.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from livekit import rtc

# Skip all tests if krisp_audio is not available
pytest.importorskip("krisp_audio")

from livekit.plugins.krisp import KrispVivaFilter


@pytest.fixture
def mock_krisp_session():
    """Mock Krisp session for testing without actual SDK calls."""
    with patch("livekit.plugins.krisp.viva_filter.krisp_audio") as mock:
        # Mock version
        mock_version = MagicMock()
        mock_version.major = 1
        mock_version.minor = 0
        mock_version.patch = 0
        mock.getVersion.return_value = mock_version

        # Mock NcInt16 session
        mock_session = MagicMock()
        mock_session.process = MagicMock(
            side_effect=lambda data, level: data  # Return input as-is
        )
        mock.NcInt16.create.return_value = mock_session

        # Mock enums
        mock.SamplingRate.Sr16000Hz = "16000Hz"
        mock.FrameDuration.Fd10ms = "10ms"
        mock.LogLevel.Off = 0

        yield mock


@pytest.fixture
def sample_audio_frame():
    """Create a sample audio frame for testing."""
    sample_rate = 16000
    duration_ms = 10
    samples = int(sample_rate * duration_ms / 1000)  # 160 samples for 10ms at 16kHz

    # Generate some sample audio data (sine wave)
    audio_data = np.sin(2 * np.pi * 440 * np.arange(samples) / sample_rate)
    audio_data = (audio_data * 32767).astype(np.int16)

    return rtc.AudioFrame(
        data=audio_data.tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


@pytest.mark.skipif(
    not os.getenv("KRISP_VIVA_MODEL_PATH"),
    reason="KRISP_VIVA_MODEL_PATH not set",
)
class TestKrispVivaFilterIntegration:
    """Integration tests requiring actual Krisp SDK and model."""

    def test_initialization(self):
        """Test filter initialization with model path."""
        filter = KrispVivaFilter()
        assert filter is not None
        assert filter.is_enabled is True

    @pytest.mark.asyncio
    async def test_filter_single_frame(self, sample_audio_frame):
        """Test filtering a single audio frame."""
        filter = KrispVivaFilter()
        filtered_frame = await filter.filter(sample_audio_frame)

        assert isinstance(filtered_frame, rtc.AudioFrame)
        assert filtered_frame.sample_rate == sample_audio_frame.sample_rate
        assert filtered_frame.num_channels == sample_audio_frame.num_channels

    @pytest.mark.asyncio
    async def test_process_stream(self, sample_audio_frame):
        """Test processing a stream of audio frames."""
        filter = KrispVivaFilter()

        # Create a stream of frames
        async def audio_stream():
            for _ in range(10):
                yield sample_audio_frame

        frame_count = 0
        async for filtered_frame in filter.process_stream(audio_stream()):
            assert isinstance(filtered_frame, rtc.AudioFrame)
            frame_count += 1

        assert frame_count > 0

    @pytest.mark.asyncio
    async def test_enable_disable(self, sample_audio_frame):
        """Test enabling and disabling the filter."""
        filter = KrispVivaFilter()

        # Enable (default state)
        assert filter.is_enabled is True
        filtered_1 = await filter.filter(sample_audio_frame)
        assert filtered_1.samples_per_channel > 0

        # Disable
        filter.disable()
        assert filter.is_enabled is False
        filtered_2 = await filter.filter(sample_audio_frame)
        # When disabled, should return original frame
        assert filtered_2.data == sample_audio_frame.data

        # Re-enable
        filter.enable()
        assert filter.is_enabled is True


class TestKrispVivaFilterUnit:
    """Unit tests with mocked Krisp SDK."""

    def test_initialization_with_model_path(self, mock_krisp_session, tmp_path):
        """Test initialization with explicit model path."""
        # Create a dummy model file
        model_file = tmp_path / "test_model.kef"
        model_file.write_text("dummy model")

        filter = KrispVivaFilter(model_path=str(model_file))
        assert filter._model_path == str(model_file)

    def test_initialization_without_model_path(self, mock_krisp_session):
        """Test initialization fails without model path or env var."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Model path"):
                KrispVivaFilter()

    def test_initialization_with_env_var(self, mock_krisp_session, tmp_path):
        """Test initialization with environment variable."""
        model_file = tmp_path / "test_model.kef"
        model_file.write_text("dummy model")

        with patch.dict(os.environ, {"KRISP_VIVA_MODEL_PATH": str(model_file)}):
            filter = KrispVivaFilter()
            assert filter._model_path == str(model_file)

    def test_invalid_model_extension(self, mock_krisp_session, tmp_path):
        """Test that non-.kef files are rejected."""
        model_file = tmp_path / "test_model.txt"
        model_file.write_text("dummy model")

        with pytest.raises(Exception, match=".kef extension"):
            KrispVivaFilter(model_path=str(model_file))

    def test_nonexistent_model_file(self, mock_krisp_session):
        """Test that nonexistent files are rejected."""
        with pytest.raises(FileNotFoundError):
            KrispVivaFilter(model_path="/nonexistent/model.kef")

    @pytest.mark.asyncio
    async def test_buffering_small_frames(self, mock_krisp_session, tmp_path):
        """Test that small frames are buffered until complete."""
        model_file = tmp_path / "test_model.kef"
        model_file.write_text("dummy model")

        filter = KrispVivaFilter(model_path=str(model_file))

        # Create a very small frame (less than 10ms)
        small_data = np.zeros(80, dtype=np.int16)  # 5ms at 16kHz
        small_frame = rtc.AudioFrame(
            data=small_data.tobytes(),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=80,
        )

        # First small frame should buffer and return empty
        result = await filter.filter(small_frame)
        assert result.samples_per_channel == 0

    @pytest.mark.asyncio
    async def test_disabled_filter_passthrough(self, mock_krisp_session, tmp_path):
        """Test that disabled filter passes through audio unchanged."""
        model_file = tmp_path / "test_model.kef"
        model_file.write_text("dummy model")

        filter = KrispVivaFilter(model_path=str(model_file))
        filter.disable()

        # Create test frame
        test_data = np.arange(160, dtype=np.int16)
        test_frame = rtc.AudioFrame(
            data=test_data.tobytes(),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=160,
        )

        result = await filter.filter(test_frame)
        assert result.data == test_frame.data

    def test_enable_disable_toggle(self, mock_krisp_session, tmp_path):
        """Test toggling filter on and off."""
        model_file = tmp_path / "test_model.kef"
        model_file.write_text("dummy model")

        filter = KrispVivaFilter(model_path=str(model_file))

        assert filter.is_enabled is True

        filter.disable()
        assert filter.is_enabled is False

        filter.enable()
        assert filter.is_enabled is True

    def test_noise_suppression_level(self, mock_krisp_session, tmp_path):
        """Test noise suppression level configuration."""
        model_file = tmp_path / "test_model.kef"
        model_file.write_text("dummy model")

        filter = KrispVivaFilter(
            model_path=str(model_file),
            noise_suppression_level=50,
        )

        assert filter._noise_suppression_level == 50

    def test_supported_sample_rates(self, mock_krisp_session):
        """Test that all documented sample rates are supported."""
        expected_rates = [8000, 16000, 24000, 32000, 44100, 48000]
        for rate in expected_rates:
            assert rate in KrispVivaFilter.SAMPLE_RATES

    def test_supported_frame_durations(self, mock_krisp_session):
        """Test that all documented frame durations are supported."""
        expected_durations = [10, 15, 20, 30, 32]
        for duration in expected_durations:
            assert duration in KrispVivaFilter.FRAME_DURATIONS

    def test_invalid_frame_duration(self, mock_krisp_session, tmp_path):
        """Test that invalid frame durations are rejected."""
        model_file = tmp_path / "test_model.kef"
        model_file.write_text("dummy model")

        with pytest.raises(ValueError, match="Unsupported frame duration"):
            KrispVivaFilter(model_path=str(model_file), frame_duration_ms=25)

    def test_frame_duration_configuration(self, mock_krisp_session, tmp_path):
        """Test frame duration can be configured."""
        model_file = tmp_path / "test_model.kef"
        model_file.write_text("dummy model")

        for duration in [10, 15, 20, 30, 32]:
            filter = KrispVivaFilter(
                model_path=str(model_file),
                frame_duration_ms=duration,
            )
            assert filter._frame_duration_ms == duration


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
