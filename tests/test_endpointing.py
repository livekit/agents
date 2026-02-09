from unittest.mock import patch

import pytest

from livekit.agents.utils.exp_filter import ExpFilter
from livekit.agents.voice.endpointing import DynamicEndpointing


class TestExponentialMovingAverage:
    """Test cases for the ExponentialMovingAverage class."""

    def test_initialization_with_valid_alpha(self) -> None:
        """Test that EMA initializes correctly with valid alpha values."""
        ema = ExpFilter(alpha=0.5)
        assert ema.value is None

        ema_with_initial = ExpFilter(alpha=0.5, initial=10.0)
        assert ema_with_initial.value == 10.0

        ema = ExpFilter(alpha=1.0)
        assert ema.value is None

    def test_initialization_with_invalid_alpha(self) -> None:
        """Test with invalid alpha values."""
        with pytest.raises(ValueError, match="alpha must be in"):
            ExpFilter(alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            ExpFilter(alpha=-0.5)

        with pytest.raises(ValueError, match="alpha must be in"):
            ExpFilter(alpha=1.5)

    def test_update_with_no_initial_value(self) -> None:
        """Test that first update sets the value directly."""
        ema = ExpFilter(alpha=0.5)
        result = ema.apply(1.0, 10.0)
        assert result == 10.0
        assert ema.value == 10.0

    def test_update_with_initial_value(self) -> None:
        """Test that update applies EMA formula when initial value exists."""
        ema = ExpFilter(alpha=0.5, initial=10.0)
        result = ema.apply(1.0, 20.0)  # 0.5 * 20 + 0.5 * 10 = 10 + 5 = 15
        assert result == 15.0
        assert ema.value == 15.0

    def test_update_multiple_times(self) -> None:
        """Test multiple updates calculate correctly."""
        ema = ExpFilter(alpha=0.5, initial=10.0)
        ema.apply(1.0, 20.0)  # 15.0
        ema.apply(1.0, 20.0)  # 0.5 * 20 + 0.5 * 15 = 17.5
        assert ema.value == 17.5

    def test_reset(self) -> None:
        ema = ExpFilter(alpha=0.5, initial=10.0)
        ema.reset()
        assert ema.value is None

        ema = ExpFilter(alpha=0.5, initial=10.0)
        ema.reset(initial=5.0)
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

    def test_empty_delays(self) -> None:
        """Test between_utterance_delay returns 0 when no utterances recorded."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        assert ep.between_utterance_delay == 0.0
        assert ep.between_turn_delay == 0.0
        assert ep.immediate_interruption_delay == (0.0, 0.0)

    def test_on_utterance_ended(self) -> None:
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()
        assert ep._utterance_ended_at == 100.0

        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended(adjustment=-0.1)
        assert ep._utterance_ended_at == 99.9

    def test_on_utterance_started(self) -> None:
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        with patch("time.time", return_value=100.0):
            ep.on_utterance_started()
        assert ep._utterance_started_at == 100.0

    def test_on_agent_speech_started(self) -> None:
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        with patch("time.time", return_value=100.0):
            ep.on_agent_speech_started(adjustment=-0.1)
        assert ep._agent_speech_started_at == 99.9

    def test_between_utterance_delay_calculation(self) -> None:
        """Test between_utterance_delay calculates the gap between utterances."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)

        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        with patch("time.time", return_value=100.5):
            ep.on_utterance_started()

        assert ep.between_utterance_delay == pytest.approx(0.5, rel=1e-5)

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

        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        with patch("time.time", return_value=100.4):
            ep.on_utterance_started(interruption=False)
        # min_delay should be updated via EMA: 0.5 * 0.4 + 0.5 * 0.3 = 0.35
        expected = 0.5 * 0.4 + 0.5 * initial_min
        assert ep.min_delay == pytest.approx(expected, rel=1e-5)

    def test_new_turn_updates_max_delay(self) -> None:
        """Test that new turns (case 3) update max_delay via EMA."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        with patch("time.time", return_value=100.6):
            ep.on_agent_speech_started()

        with patch("time.time", return_value=101.5):
            ep.on_utterance_started(interruption=False)

        assert ep.max_delay == pytest.approx(0.5 * 0.6 + 0.5 * 1.0, rel=1e-5)

    def test_interruption_updates_min_delay(self) -> None:
        """Test that immediate interruptions (case 2) update min_delay."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        with patch("time.time", return_value=100.2):
            ep.on_agent_speech_started()

        assert ep._agent_speech_started_at is not None

        with patch("time.time", return_value=100.25):
            ep.on_utterance_started(interruption=True)

        assert ep._interrupting is True

        with patch("time.time", return_value=100.5):
            ep.on_utterance_ended()

        # pause = 100.25 - 100.0 = 0.25
        # EMA: 0.5 * max(0.25, 0.3) + 0.5 * 0.3 = 0.3
        assert ep._interrupting is False
        assert ep._agent_speech_started_at is None  # already used
        assert ep.min_delay == pytest.approx(0.3, rel=1e-5)

    def test_update_options(self) -> None:
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        ep.update_options(min_delay=0.5)
        assert ep.min_delay == 0.5
        assert ep._min_delay == 0.5

        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        ep.update_options(max_delay=2.0)
        assert ep.max_delay == 2.0
        assert ep._max_delay == 2.0

        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        ep.update_options(min_delay=0.5, max_delay=2.0)
        assert ep.min_delay == 0.5
        assert ep.max_delay == 2.0

        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        ep.update_options()
        assert ep.min_delay == 0.3
        assert ep.max_delay == 1.0

    def test_max_delay_clamped_to_configured_max(self) -> None:
        """Test that max_delay updates are clamped to the configured maximum."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=1.0)

        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        with patch("time.time", return_value=102.0):
            ep.on_agent_speech_started()

        with patch("time.time", return_value=105.0):
            ep.on_utterance_started(interruption=False)

        assert ep.max_delay == 1.0  # pause=2.0 clamped to _max_delay

    def test_max_delay_clamped_to_min_delay(self) -> None:
        """Test that max_delay updates are clamped to at least min_delay."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=1.0)

        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        with patch("time.time", return_value=100.1):
            ep.on_agent_speech_started()

        with patch("time.time", return_value=100.5):
            ep.on_utterance_started(interruption=False)

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

        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        with patch("time.time", return_value=100.2):
            ep.on_agent_speech_started()

        with patch("time.time", return_value=100.25):
            ep.on_utterance_started(interruption=True)

        assert ep._interrupting is True
        prev_val = ep.min_delay, ep.max_delay

        with patch("time.time", return_value=100.35):
            ep.on_utterance_started(interruption=True)

        assert ep._interrupting is True
        assert prev_val == (ep.min_delay, ep.max_delay)

    def test_delayed_interruption_updates_max_delay_without_crashing(self) -> None:
        """Delayed interruptions should update max delay via the EMA path."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()

        with patch("time.time", return_value=100.9):
            ep.on_agent_speech_started()

        with patch("time.time", return_value=101.8):
            ep.on_utterance_started(interruption=True)

        assert ep.max_delay == pytest.approx(0.5 * 0.9 + 0.5 * 1.0, rel=1e-5)

    def test_update_options_preserves_filter_alpha(self) -> None:
        """Changing delays should not overwrite the EMA smoothing coefficient."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        ep.update_options(min_delay=0.6, max_delay=2.0)

        assert ep._utterance_pause._alpha == pytest.approx(0.5, rel=1e-5)
        assert ep._turn_pause._alpha == pytest.approx(0.5, rel=1e-5)

    def test_update_options_updates_filter_clamp_bounds(self) -> None:
        """Changing delays should propagate into exp-filter min/max clamp limits."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        ep.update_options(min_delay=0.5, max_delay=2.0)
        assert ep._utterance_pause._min_val == 0.5
        assert ep._turn_pause._max_val == 2.0

        # min_delay updated from 0.3 to 0.5
        with patch("time.time", return_value=100.0):
            ep.on_utterance_ended()
        with patch("time.time", return_value=100.2):
            ep.on_utterance_started(interruption=False)
        assert ep.min_delay == pytest.approx(0.5, rel=1e-5)

        # max_delay updated from 1.0 to 2.0
        with patch("time.time", return_value=101.0):
            ep.on_utterance_ended()
        with patch("time.time", return_value=102.8):
            ep.on_agent_speech_started()
        with patch("time.time", return_value=103.5):
            ep.on_utterance_started(interruption=False)
        assert 1.0 < ep.max_delay <= 2.0
