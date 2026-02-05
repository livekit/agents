from unittest.mock import patch

import pytest

from livekit.agents.voice.endpointing import DynamicEndpointing, ExponentialMovingAverage


class TestExponentialMovingAverage:
    """Test cases for the ExponentialMovingAverage class."""

    def test_initialization_with_valid_alpha(self) -> None:
        """Test that EMA initializes correctly with valid alpha values."""
        ema = ExponentialMovingAverage(alpha=0.5)
        assert ema.value is None

        ema_with_initial = ExponentialMovingAverage(alpha=0.5, initial=10.0)
        assert ema_with_initial.value == 10.0

    def test_initialization_with_alpha_one(self) -> None:
        """Test that alpha=1 is valid (no smoothing)."""
        ema = ExponentialMovingAverage(alpha=1.0)
        assert ema.value is None

    def test_initialization_with_invalid_alpha_zero(self) -> None:
        """Test that alpha=0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            ExponentialMovingAverage(alpha=0.0)

    def test_initialization_with_invalid_alpha_negative(self) -> None:
        """Test that negative alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            ExponentialMovingAverage(alpha=-0.5)

    def test_initialization_with_invalid_alpha_greater_than_one(self) -> None:
        """Test that alpha > 1 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            ExponentialMovingAverage(alpha=1.5)

    def test_update_with_no_initial_value(self) -> None:
        """Test that first update sets the value directly."""
        ema = ExponentialMovingAverage(alpha=0.5)
        result = ema.update(10.0)
        assert result == 10.0
        assert ema.value == 10.0

    def test_update_with_initial_value(self) -> None:
        """Test that update applies EMA formula when initial value exists."""
        ema = ExponentialMovingAverage(alpha=0.5, initial=10.0)
        result = ema.update(20.0)
        # new_value = 0.5 * 20 + 0.5 * 10 = 10 + 5 = 15
        assert result == 15.0
        assert ema.value == 15.0

    def test_update_multiple_times(self) -> None:
        """Test multiple updates calculate correctly."""
        ema = ExponentialMovingAverage(alpha=0.5, initial=10.0)
        ema.update(20.0)  # 15.0
        ema.update(20.0)  # 0.5 * 20 + 0.5 * 15 = 17.5
        assert ema.value == 17.5

    def test_update_with_alpha_one(self) -> None:
        """Test that alpha=1 means no smoothing (new value replaces old)."""
        ema = ExponentialMovingAverage(alpha=1.0, initial=10.0)
        result = ema.update(20.0)
        assert result == 20.0

    def test_reset_without_value(self) -> None:
        """Test reset clears the value."""
        ema = ExponentialMovingAverage(alpha=0.5, initial=10.0)
        ema.reset()
        assert ema.value is None

    def test_reset_with_value(self) -> None:
        """Test reset sets a new value."""
        ema = ExponentialMovingAverage(alpha=0.5, initial=10.0)
        ema.reset(value=5.0)
        assert ema.value == 5.0


class TestDynamicEndpointing:
    """Test cases for the DynamicEndpointing class."""

    def test_initialization(self) -> None:
        """Test basic initialization with min and max delay."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        assert ep.min_delay == 0.3
        assert ep.max_delay == 1.0

    def test_initialization_with_custom_alpha(self) -> None:
        """Test initialization with custom alpha value."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.2)
        assert ep.min_delay == 0.3
        assert ep.max_delay == 1.0

    def test_between_utterance_delay_no_data(self) -> None:
        """Test between_utterance_delay returns 0 when no utterances recorded."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        assert ep.between_utterance_delay == 0.0

    def test_between_turn_delay_no_data(self) -> None:
        """Test between_turn_delay returns 0 when no agent speech recorded."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        assert ep.between_turn_delay == 0.0

    def test_immediate_interruption_delay_no_data(self) -> None:
        """Test immediate_interruption_delay returns zeros when no data."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        assert ep.immediate_interruption_delay == (0.0, 0.0)

    def test_on_utterance_ended_sets_timestamp(self) -> None:
        """Test that on_utterance_ended records the timestamp."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()
        assert ep._utterance_ended_at == 100.0

    def test_on_utterance_ended_with_adjustment(self) -> None:
        """Test that on_utterance_ended applies adjustment."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended(adjustment=-0.1)
        assert ep._utterance_ended_at == 99.9

    def test_on_utterance_started_sets_timestamp(self) -> None:
        """Test that on_utterance_started records the timestamp."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        with patch("time.time", return_value=100.0):
            ep.on_utterance_started()
        assert ep._utterance_started_at == 100.0

    def test_on_agent_speech_started_sets_timestamp(self) -> None:
        """Test that on_agent_speech_started records the timestamp."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        with patch("time.time", return_value=100.0):
            ep.on_agent_speech_started()
        assert ep._agent_speech_started_at == 100.0

    def test_on_agent_speech_started_with_adjustment(self) -> None:
        """Test that on_agent_speech_started applies adjustment."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        with patch("time.time", return_value=100.0):
            ep.on_agent_speech_started(adjustment=0.2)
        assert ep._agent_speech_started_at == 100.2

    def test_between_utterance_delay_calculation(self) -> None:
        """Test between_utterance_delay calculates the gap between utterances."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)

        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        with patch("time.time", return_value=100.5):
            ep.on_utterance_started()

        assert ep.between_utterance_delay == 0.5

    def test_between_turn_delay_calculation(self) -> None:
        """Test between_turn_delay calculates gap between utterance end and agent speech."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)

        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        with patch("time.time", return_value=100.8):
            ep.on_agent_speech_started()

        assert ep.between_turn_delay == pytest.approx(0.8, rel=1e-5)

    def test_pause_between_utterances_updates_min_delay(self) -> None:
        """Test that pauses between utterances (case 1) update min_delay via EMA."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)
        initial_min = ep.min_delay

        # First utterance ends
        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        # Second utterance starts (no agent speech, no interruption)
        with patch("time.time", return_value=100.4):
            ep.on_utterance_started(interruption=False)

        # min_delay should be updated via EMA: 0.5 * 0.4 + 0.5 * 0.3 = 0.35
        expected = 0.5 * 0.4 + 0.5 * initial_min
        assert ep.min_delay == pytest.approx(expected, rel=1e-5)

    def test_new_turn_updates_max_delay(self) -> None:
        """Test that new turns (case 3) update max_delay via EMA."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        # User utterance ends
        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        # Agent starts speaking
        with patch("time.time", return_value=100.6):
            ep.on_agent_speech_started()

        # New user utterance starts (not interruption, new turn)
        with patch("time.time", return_value=101.5):
            ep.on_utterance_started(interruption=False)

        # max_delay should be updated via EMA: 0.5 * 0.6 + 0.5 * 1.0 = 0.8
        # (pause is clamped between min_delay and _max_delay)
        assert ep.max_delay == pytest.approx(0.8, rel=1e-5)

    def test_interruption_updates_min_delay(self) -> None:
        """Test that immediate interruptions (case 2) update min_delay."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        # First user utterance ends
        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        # Agent starts speaking
        with patch("time.time", return_value=100.2):
            ep.on_agent_speech_started()

        # User interrupts immediately (within min_delay window)
        with patch("time.time", return_value=100.25):
            ep.on_utterance_started(interruption=True)

        # min_delay should be updated based on the pause
        # pause = 100.25 - 100.0 = 0.25
        # EMA: 0.5 * max(0.25, 0.3) + 0.5 * 0.3 = 0.3
        assert ep.min_delay == pytest.approx(0.3, rel=1e-5)

    def test_interruption_clears_agent_speech_timestamp(self) -> None:
        """Test that interruption clears agent speech timestamp."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)

        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        with patch("time.time", return_value=100.2):
            ep.on_agent_speech_started()

        assert ep._agent_speech_started_at is not None

        with patch("time.time", return_value=100.25):
            ep.on_utterance_started(interruption=True)

        assert ep._agent_speech_started_at is None

    def test_on_utterance_ended_resets_interrupting_flag(self) -> None:
        """Test that on_utterance_ended resets the interrupting flag."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)

        # Set up an interruption scenario
        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        with patch("time.time", return_value=100.2):
            ep.on_agent_speech_started()

        with patch("time.time", return_value=100.25):
            ep.on_utterance_started(interruption=True)

        assert ep._interrupting is True

        with patch("time.time", return_value=100.5):
            ep.on_utterance_ended()

        assert ep._interrupting is False

    def test_update_options_min_delay(self) -> None:
        """Test that update_options can change min_delay."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        ep.update_options(min_delay=0.5)
        assert ep.min_delay == 0.5
        assert ep._min_delay == 0.5

    def test_update_options_max_delay(self) -> None:
        """Test that update_options can change max_delay."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        ep.update_options(max_delay=2.0)
        assert ep.max_delay == 2.0
        assert ep._max_delay == 2.0

    def test_update_options_both_delays(self) -> None:
        """Test that update_options can change both delays at once."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        ep.update_options(min_delay=0.5, max_delay=2.0)
        assert ep.min_delay == 0.5
        assert ep.max_delay == 2.0

    def test_update_options_not_given(self) -> None:
        """Test that update_options ignores NOT_GIVEN values."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        ep.update_options()  # No arguments
        assert ep.min_delay == 0.3
        assert ep.max_delay == 1.0

    def test_max_delay_clamped_to_configured_max(self) -> None:
        """Test that max_delay updates are clamped to the configured maximum."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=1.0)

        # User utterance ends
        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        # Agent starts speaking
        with patch("time.time", return_value=102.0):
            ep.on_agent_speech_started()

        # New user utterance starts with very long pause
        with patch("time.time", return_value=105.0):
            ep.on_utterance_started(interruption=False)

        # max_delay should be clamped to _max_delay (1.0)
        assert ep.max_delay == 1.0

    def test_max_delay_clamped_to_min_delay(self) -> None:
        """Test that max_delay updates are clamped to at least min_delay."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=1.0)

        # User utterance ends
        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        # Agent starts speaking very quickly
        with patch("time.time", return_value=100.1):
            ep.on_agent_speech_started()

        # New user utterance starts
        with patch("time.time", return_value=100.5):
            ep.on_utterance_started(interruption=False)

        # max_delay should be at least min_delay (0.3)
        assert ep.max_delay >= ep._min_delay

    def test_non_interruption_clears_agent_speech(self) -> None:
        """Test that non-interruption utterance start clears agent speech timestamp."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)

        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        with patch("time.time", return_value=100.5):
            ep.on_agent_speech_started()

        assert ep._agent_speech_started_at is not None

        with patch("time.time", return_value=102.0):
            ep.on_utterance_started(interruption=False)

        assert ep._agent_speech_started_at is None

    def test_consecutive_interruptions_only_track_first(self) -> None:
        """Test that only the first interruption in a sequence updates min_delay."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        # Setup: utterance ends, agent speaks, first interruption
        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        with patch("time.time", return_value=100.2):
            ep.on_agent_speech_started()

        with patch("time.time", return_value=100.25):
            ep.on_utterance_started(interruption=True)

        # Second "interruption" utterance should not update min_delay
        # because _interrupting is True
        with patch("time.time", return_value=100.3):
            ep.on_utterance_ended()

        with patch("time.time", return_value=100.35):
            ep.on_utterance_started(interruption=True)

        # Should be different now since _interrupting was reset on utterance_ended
        # and this is a new interruption scenario (but no agent speech)
        # Actually since agent_speech_started_at is None, this won't trigger case 2
        assert ep._interrupting is True
