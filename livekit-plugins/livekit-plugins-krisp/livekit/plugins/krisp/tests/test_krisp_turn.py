# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Krisp turn detection."""

import os

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import llm

# Skip if krisp_audio not available
pytest.importorskip("krisp_audio")

from livekit.plugins.krisp import KrispVivaTurn  # noqa: E402


def create_audio_frame(
    duration_ms: int,
    sample_rate: int = 16000,
    frequency: int = 440,
) -> rtc.AudioFrame:
    """Create a synthetic audio frame for testing."""
    num_samples = int(sample_rate * duration_ms / 1000)

    if frequency > 0:
        # Generate a sine wave
        t = np.linspace(0, duration_ms / 1000, num_samples, False)
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
    else:
        # Generate silence
        audio = np.zeros(num_samples)

    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)

    return rtc.AudioFrame(
        data=audio_int16.tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=num_samples,
    )


@pytest.fixture
def model_path():
    """Get model path from environment or skip."""
    path = os.getenv("KRISP_VIVA_TURN_MODEL_PATH")
    if not path:
        pytest.skip("KRISP_VIVA_TURN_MODEL_PATH not set")
    if not os.path.isfile(path):
        pytest.skip(f"Model file not found: {path}")
    return path


@pytest.mark.asyncio
class TestKrispVivaTurn:
    """Test suite for Krisp VIVA turn detection."""

    async def test_initialization(self, model_path):
        """Test basic initialization."""
        detector = KrispVivaTurn(model_path=model_path)

        assert detector.model == "krisp-viva-turn"
        assert detector.provider == "krisp"
        assert detector.threshold == 0.5
        assert not detector.speech_triggered
        assert detector.last_probability is None

        detector.close()

    async def test_initialization_with_params(self, model_path):
        """Test initialization with custom parameters."""
        detector = KrispVivaTurn(
            model_path=model_path,
            threshold=0.7,
            frame_duration_ms=10,
            sample_rate=48000,
        )

        assert detector.threshold == 0.7
        detector.close()

    async def test_invalid_frame_duration(self, model_path):
        """Test that invalid frame duration raises error."""
        with pytest.raises(ValueError, match="Unsupported frame duration"):
            KrispVivaTurn(model_path=model_path, frame_duration_ms=25)

    async def test_missing_model_path(self):
        """Test that missing model path raises error."""
        original = os.environ.pop("KRISP_VIVA_TURN_MODEL_PATH", None)
        try:
            with pytest.raises(ValueError, match="Model path must be provided"):
                KrispVivaTurn(model_path=None)
        finally:
            if original:
                os.environ["KRISP_VIVA_TURN_MODEL_PATH"] = original

    async def test_process_audio(self, model_path):
        """Test processing audio frames."""
        detector = KrispVivaTurn(
            model_path=model_path,
            frame_duration_ms=20,
            sample_rate=16000,
        )

        # Create and process audio frames
        for _ in range(10):
            frame = create_audio_frame(20, sample_rate=16000, frequency=440)
            prob = detector.process_audio(frame, is_speech=True)

            # Probability should be between -1 and 1
            assert -1.0 <= prob <= 1.0

        # Should have triggered speech detection
        assert detector.speech_triggered

        detector.close()

    async def test_threshold_property(self, model_path):
        """Test threshold getter and setter."""
        detector = KrispVivaTurn(model_path=model_path, threshold=0.5)

        assert detector.threshold == 0.5

        detector.threshold = 0.8
        assert detector.threshold == 0.8

        detector.close()

    async def test_protocol_methods(self, model_path):
        """Test protocol method implementations."""
        detector = KrispVivaTurn(model_path=model_path)

        # Test supports_language (should always be True)
        assert await detector.supports_language("en")
        assert await detector.supports_language("es")
        assert await detector.supports_language(None)

        # Test unlikely_threshold
        threshold = await detector.unlikely_threshold("en")
        assert threshold == 0.5

        detector.close()

    async def test_predict_end_of_turn(self, model_path):
        """Test predict_end_of_turn method."""
        detector = KrispVivaTurn(model_path=model_path)

        # Create chat context
        chat_ctx = llm.ChatContext()
        chat_ctx.append(role="user", text="Hello")

        # Should return 0.0 initially (no audio processed)
        prob = await detector.predict_end_of_turn(chat_ctx)
        assert prob == 0.0

        # Process some audio
        frame = create_audio_frame(20, sample_rate=16000)
        detector.process_audio(frame, is_speech=True)

        # After processing, might have a probability
        prob = await detector.predict_end_of_turn(chat_ctx)
        assert isinstance(prob, float)

        detector.close()

    async def test_clear_state(self, model_path):
        """Test that clear resets state."""
        detector = KrispVivaTurn(model_path=model_path, sample_rate=16000)

        # Process audio to set state
        frame = create_audio_frame(20, sample_rate=16000)
        detector.process_audio(frame, is_speech=True)

        # Clear state
        detector.clear()

        assert not detector.speech_triggered
        assert detector.last_probability is None

        detector.close()

    async def test_context_manager(self, model_path):
        """Test using as context manager."""
        with KrispVivaTurn(model_path=model_path) as detector:
            assert detector.model == "krisp-viva-turn"
        # Should close automatically

    async def test_different_sample_rates(self, model_path):
        """Test handling different sample rates."""
        supported_rates = [8000, 16000, 24000, 32000, 44100, 48000]

        for sample_rate in supported_rates:
            detector = KrispVivaTurn(
                model_path=model_path,
                frame_duration_ms=20,
                sample_rate=sample_rate,
            )

            frame = create_audio_frame(20, sample_rate=sample_rate)
            prob = detector.process_audio(frame, is_speech=True)

            # Should process without error
            assert isinstance(prob, float)

            detector.close()

    async def test_frame_probabilities(self, model_path):
        """Test that frame probabilities are tracked."""
        detector = KrispVivaTurn(
            model_path=model_path,
            frame_duration_ms=20,
            sample_rate=16000,
        )

        # Process multiple frames
        for _ in range(5):
            frame = create_audio_frame(20, sample_rate=16000, frequency=440)
            detector.process_audio(frame, is_speech=True)

        # Should have probabilities (may be empty initially during warmup)
        assert isinstance(detector.frame_probabilities, list)

        detector.close()

    async def test_speech_detection(self, model_path):
        """Test speech detection tracking."""
        detector = KrispVivaTurn(model_path=model_path, sample_rate=16000)

        assert not detector.speech_triggered

        # Process with speech flag
        frame = create_audio_frame(20, sample_rate=16000)
        detector.process_audio(frame, is_speech=True)

        # After warmup, should trigger
        for _ in range(10):
            frame = create_audio_frame(20, sample_rate=16000)
            detector.process_audio(frame, is_speech=True)

        assert detector.speech_triggered

        detector.close()


@pytest.mark.asyncio
class TestKrispVivaTurnIntegration:
    """Integration tests for Krisp VIVA turn detection."""

    async def test_realistic_audio_sequence(self, model_path):
        """Test with a realistic sequence of audio frames."""
        detector = KrispVivaTurn(
            model_path=model_path,
            threshold=0.6,
            frame_duration_ms=20,
            sample_rate=16000,
        )

        # Simulate speech: 1 second of audio
        for _ in range(50):
            frame = create_audio_frame(20, sample_rate=16000, frequency=440)
            prob = detector.process_audio(frame, is_speech=True)

            if prob >= detector.threshold:
                # Turn detected as ended
                break

        detector.close()

    async def test_with_silence(self, model_path):
        """Test behavior with silent frames."""
        detector = KrispVivaTurn(model_path=model_path, sample_rate=16000)

        # Process silence
        for _ in range(10):
            frame = create_audio_frame(20, sample_rate=16000, frequency=0)
            prob = detector.process_audio(frame, is_speech=False)
            assert isinstance(prob, float)

        detector.close()

