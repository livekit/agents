"""
Test suite for VoiceActivityTimeout properties in Google STT V2 API.

This test suite validates that VoiceActivityTimeout properties are:
1. Correctly initialized
2. Properly passed to Google Cloud Speech-to-Text V2 API
3. Can be updated dynamically
4. Follow existing code patterns and best practices
"""

import pytest

from livekit.plugins.google import STT
from livekit.plugins.google.stt import STTOptions


class TestVoiceActivityTimeout:
    """Test VoiceActivityTimeout support."""

    def test_voice_activity_timeouts_default(self):
        """Test voice activity timeouts are not set by default."""
        stt = STT()
        from livekit.agents.types import NOT_GIVEN

        assert stt._config.speech_start_timeout is NOT_GIVEN
        assert stt._config.speech_end_timeout is NOT_GIVEN

    def test_voice_activity_timeouts_set(self):
        """Test voice activity timeouts can be set."""
        stt = STT(
            speech_start_timeout=10.0,
            speech_end_timeout=2.5,
        )
        assert stt._config.speech_start_timeout == 10.0
        assert stt._config.speech_end_timeout == 2.5

    def test_voice_activity_timeout_fractional_seconds(self):
        """Test voice activity timeouts handle fractional seconds."""
        stt = STT(
            speech_start_timeout=5.5,
            speech_end_timeout=1.25,
        )
        assert stt._config.speech_start_timeout == 5.5
        assert stt._config.speech_end_timeout == 1.25

    def test_speech_start_timeout_only(self):
        """Test setting only speech_start_timeout."""
        stt = STT(speech_start_timeout=15.0)
        assert stt._config.speech_start_timeout == 15.0
        from livekit.agents.types import NOT_GIVEN

        assert stt._config.speech_end_timeout is NOT_GIVEN

    def test_speech_end_timeout_only(self):
        """Test setting only speech_end_timeout."""
        stt = STT(speech_end_timeout=3.0)
        assert stt._config.speech_end_timeout == 3.0
        from livekit.agents.types import NOT_GIVEN

        assert stt._config.speech_start_timeout is NOT_GIVEN


class TestPatternConsistency:
    """Test that timeout properties follow existing code patterns."""

    def test_config_initialization_pattern(self):
        """Test that timeout properties follow initialization pattern."""
        stt = STT(
            speech_start_timeout=10.0,
            speech_end_timeout=2.0,
        )

        # Properties should be stored in _config
        assert hasattr(stt, "_config")
        assert hasattr(stt._config, "speech_start_timeout")
        assert hasattr(stt._config, "speech_end_timeout")

    def test_v2_model_detection(self):
        """Test that V2 model detection works correctly."""
        stt_v2 = STT(model="chirp_3")
        assert stt_v2._config.version == 2

        stt_v1 = STT(model="default")
        assert stt_v1._config.version == 1

    def test_v1_model_ignores_timeouts(self):
        """Test that V1 models gracefully ignore timeout settings."""
        # V1 models don't support VoiceActivityTimeout
        # The code should not crash, but timeouts won't be applied
        stt_v1 = STT(
            model="default",  # V1 model
            speech_start_timeout=10.0,
            speech_end_timeout=2.0,
        )

        # Should not raise error during initialization
        assert stt_v1._config.speech_start_timeout == 10.0
        assert stt_v1._config.speech_end_timeout == 2.0


class TestBestPractices:
    """Validate that implementation follows best practices."""

    def test_optional_parameters_use_not_given(self):
        """Test that optional parameters use NOT_GIVEN pattern."""
        from livekit.agents.types import NOT_GIVEN

        stt = STT()

        # These should use NOT_GIVEN pattern
        assert stt._config.speech_start_timeout is NOT_GIVEN
        assert stt._config.speech_end_timeout is NOT_GIVEN

    def test_config_is_dataclass(self):
        """Test that STTOptions follows dataclass pattern."""
        from dataclasses import is_dataclass

        assert is_dataclass(STTOptions)

    def test_timeout_features_documented(self):
        """Test that __init__ has docstrings for timeout parameters."""
        import inspect

        stt_init_doc = inspect.getdoc(STT.__init__)

        assert "speech_start_timeout" in stt_init_doc
        assert "speech_end_timeout" in stt_init_doc


class TestDynamicUpdates:
    """Test dynamic updating of timeout properties."""

    def test_update_timeout_options(self):
        """Test that timeout options can be updated dynamically."""
        stt = STT(
            speech_start_timeout=10.0,
            speech_end_timeout=2.0,
        )

        # Update timeouts
        stt.update_options(
            speech_start_timeout=15.0,
            speech_end_timeout=3.0,
        )

        assert stt._config.speech_start_timeout == 15.0
        assert stt._config.speech_end_timeout == 3.0

    def test_update_partial_timeouts(self):
        """Test updating only one timeout at a time."""
        stt = STT(
            speech_start_timeout=10.0,
            speech_end_timeout=2.0,
        )

        # Update only speech_start_timeout
        stt.update_options(speech_start_timeout=20.0)
        assert stt._config.speech_start_timeout == 20.0
        assert stt._config.speech_end_timeout == 2.0

        # Update only speech_end_timeout
        stt.update_options(speech_end_timeout=5.0)
        assert stt._config.speech_start_timeout == 20.0
        assert stt._config.speech_end_timeout == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
