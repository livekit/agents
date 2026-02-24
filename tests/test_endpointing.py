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
        assert ema.value == 10.0
        ema.reset()
        assert ema.value == 10.0

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

    def test_initialization_uses_updated_default_alpha(self) -> None:
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        assert ep._utterance_pause._alpha == pytest.approx(0.9, rel=1e-5)
        assert ep._turn_pause._alpha == pytest.approx(0.9, rel=1e-5)

    def test_empty_delays(self) -> None:
        """Test between_utterance_delay returns 0 when no utterances recorded."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        assert ep.between_utterance_delay == 0.0
        assert ep.between_turn_delay == 0.0
        assert ep.immediate_interruption_delay == (0.0, 0.0)

    def test_on_utterance_ended(self) -> None:
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        ep.on_end_of_speech(ended_at=100.0)
        assert ep._utterance_ended_at == 100.0

        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        ep.on_end_of_speech(ended_at=99.9)
        assert ep._utterance_ended_at == 99.9

    def test_on_utterance_started(self) -> None:
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        ep.on_start_of_speech(started_at=100.0)
        assert ep._utterance_started_at == 100.0

    def test_on_agent_speech_started(self) -> None:
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)
        ep.on_start_of_agent_speech(started_at=100.0)
        assert ep._agent_speech_started_at == 100.0

    def test_between_utterance_delay_calculation(self) -> None:
        """Test between_utterance_delay calculates the gap between utterances."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)

        ep.on_end_of_speech(ended_at=100.0)
        ep.on_start_of_speech(started_at=100.5)

        assert ep.between_utterance_delay == pytest.approx(0.5, rel=1e-5)

    def test_between_turn_delay_calculation(self) -> None:
        """Test between_turn_delay calculates gap between utterance end and agent speech."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)

        ep.on_end_of_speech(ended_at=100.0)
        ep.on_start_of_agent_speech(started_at=100.8)

        assert ep.between_turn_delay == pytest.approx(0.8, rel=1e-5)

    def test_pause_between_utterances_updates_min_delay(self) -> None:
        """Test that pauses between utterances (case 1) update min_delay via EMA."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)
        initial_min = ep.min_delay

        ep.on_end_of_speech(ended_at=100.0)
        ep.on_start_of_speech(started_at=100.4)
        ep.on_end_of_speech(ended_at=100.5, should_ignore=False)
        # min_delay should be updated via EMA: 0.5 * 0.4 + 0.5 * 0.3 = 0.35
        expected = 0.5 * 0.4 + 0.5 * initial_min
        assert ep.min_delay == pytest.approx(expected, rel=1e-5)

    def test_new_turn_updates_max_delay(self) -> None:
        """Test that new turns (case 3) update max_delay via EMA."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        ep.on_end_of_speech(ended_at=100.0)
        ep.on_start_of_agent_speech(started_at=100.6)
        ep.on_start_of_speech(started_at=101.5)
        ep.on_end_of_speech(ended_at=102.0, should_ignore=False)

        assert ep.max_delay == pytest.approx(0.5 * 0.6 + 0.5 * 1.0, rel=1e-5)

    def test_interruption_updates_min_delay(self) -> None:
        """Test that immediate interruptions (case 2) update min_delay."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        ep.on_end_of_speech(ended_at=100.0)
        ep.on_start_of_agent_speech(started_at=100.2)
        assert ep._agent_speech_started_at is not None
        ep.on_start_of_speech(started_at=100.25, overlapping=True)
        assert ep._overlapping is True

        ep.on_end_of_speech(ended_at=100.5)

        # pause = 100.25 - 100.0 = 0.25
        # EMA: 0.5 * max(0.25, 0.3) + 0.5 * 0.3 = 0.3
        assert ep._overlapping is False
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

        ep.on_end_of_speech(ended_at=100.0)
        ep.on_start_of_agent_speech(started_at=102.0)
        ep.on_start_of_speech(started_at=105.0)

        assert ep.max_delay == 1.0  # pause=2.0 clamped to _max_delay

    def test_max_delay_clamped_to_min_delay(self) -> None:
        """Test that max_delay updates are clamped to at least min_delay."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=1.0)

        ep.on_end_of_speech(ended_at=100.0)
        ep.on_start_of_agent_speech(started_at=100.1)
        ep.on_start_of_speech(started_at=100.5)

        assert ep.max_delay >= ep._min_delay

    def test_non_interruption_clears_agent_speech(self) -> None:
        """Test that non-interruption utterance start clears agent speech timestamp."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)

        ep.on_end_of_speech(ended_at=100.0)
        ep.on_start_of_agent_speech(started_at=100.5)
        assert ep._agent_speech_started_at is not None

        ep.on_start_of_speech(started_at=102.0)
        ep.on_end_of_speech(ended_at=103.0, should_ignore=False)
        assert ep._agent_speech_started_at is None

    def test_consecutive_interruptions_only_track_first(self) -> None:
        """Test that only the first interruption in a sequence updates min_delay."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        ep.on_end_of_speech(ended_at=100.0)
        ep.on_start_of_agent_speech(started_at=100.2)
        ep.on_start_of_speech(started_at=100.25, overlapping=True)

        assert ep._overlapping is True
        prev_val = ep.min_delay, ep.max_delay

        ep.on_start_of_speech(started_at=100.35)

        assert ep._overlapping is True
        assert prev_val == (ep.min_delay, ep.max_delay)

    def test_delayed_interruption_updates_max_delay_without_crashing(self) -> None:
        """Delayed interruptions should update max delay via the EMA path."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        ep.on_end_of_speech(ended_at=100.0)
        ep.on_start_of_agent_speech(started_at=100.9)
        ep.on_start_of_speech(started_at=101.8)
        ep.on_end_of_speech(ended_at=102.0, should_ignore=False)

        assert ep.max_delay == pytest.approx(0.5 * 0.9 + 0.5 * 1.0, rel=1e-5)

    def test_interruption_adjusts_stale_utterance_end_time(self) -> None:
        """Interruption path should adjust stale utterance end timestamp before delay updates."""
        ep = DynamicEndpointing(min_delay=0.06, max_delay=1.0, alpha=1.0)

        # Simulate stale ordering where end timestamp still belongs to a previous utterance.
        ep.on_end_of_speech(ended_at=99.0)
        ep.on_start_of_speech(started_at=100.0)

        ep.on_start_of_agent_speech(started_at=100.2)
        ep.on_start_of_speech(started_at=100.25, overlapping=True)

        assert ep._utterance_ended_at == pytest.approx(100.2, rel=1e-3)
        assert ep.min_delay == pytest.approx(0.06, rel=1e-5)
        assert ep.max_delay == pytest.approx(1.0, rel=1e-5)

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
        ep.on_end_of_speech(ended_at=100.0)
        ep.on_start_of_speech(started_at=100.2)
        assert ep.min_delay == pytest.approx(0.5, rel=1e-5)

        # max_delay updated from 1.0 to 2.0
        ep.on_end_of_speech(ended_at=101.0)
        ep.on_start_of_agent_speech(started_at=102.8)
        ep.on_start_of_speech(started_at=103.5)
        assert 1.0 < ep.max_delay <= 2.0

    def test_should_ignore_skips_filter_update(self) -> None:
        """should_ignore=True with overlapping=True skips EMA updates and resets state."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        ep.on_end_of_speech(ended_at=100.0)
        ep.on_start_of_agent_speech(started_at=100.5)
        # user starts 1.0s after agent (well outside 0.25s grace period)
        ep.on_start_of_speech(started_at=101.5, overlapping=True)

        prev_min = ep.min_delay
        prev_max = ep.max_delay

        ep.on_end_of_speech(ended_at=101.8, should_ignore=True)

        # filters should not have been updated
        assert ep.min_delay == prev_min
        assert ep.max_delay == prev_max
        # state should be reset
        assert ep._utterance_started_at is None
        assert ep._utterance_ended_at is None
        assert ep._overlapping is False
        assert ep._speaking is False

    def test_should_ignore_without_overlapping_still_updates(self) -> None:
        """should_ignore=True but overlapping=False follows the normal update path."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)
        initial_min = ep.min_delay

        ep.on_end_of_speech(ended_at=100.0)
        ep.on_start_of_speech(started_at=100.4, overlapping=False)
        ep.on_end_of_speech(ended_at=100.6, should_ignore=True)

        # should_ignore only gates when overlapping, so min_delay should update (case 1)
        expected = 0.5 * 0.4 + 0.5 * initial_min
        assert ep.min_delay == pytest.approx(expected, rel=1e-5)

    def test_should_ignore_grace_period_overrides(self) -> None:
        """User speech within grace period of agent speech overrides should_ignore=True."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        ep.on_end_of_speech(ended_at=100.0)
        ep.on_start_of_agent_speech(started_at=100.5)
        # user starts speaking 0.1s after agent (within 0.25s grace period)
        ep.on_start_of_speech(started_at=100.6, overlapping=True)

        prev_min = ep.min_delay
        ep.on_end_of_speech(ended_at=100.8, should_ignore=True)

        # grace period should override should_ignore, so the interruption path runs
        # and state is properly cleaned up (not left as None)
        assert ep._utterance_ended_at == 100.8
        assert ep._speaking is False

    def test_should_ignore_outside_grace_period(self) -> None:
        """User speech well after agent speech start is outside grace period."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        ep.on_end_of_speech(ended_at=100.0)
        ep.on_start_of_agent_speech(started_at=100.5)
        # user starts speaking 0.5s after agent (outside 0.25s grace period)
        ep.on_start_of_speech(started_at=101.0, overlapping=True)

        prev_min = ep.min_delay
        prev_max = ep.max_delay
        ep.on_end_of_speech(ended_at=101.5, should_ignore=True)

        # outside grace period, should_ignore takes effect — no filter update
        assert ep.min_delay == prev_min
        assert ep.max_delay == prev_max
        assert ep._utterance_started_at is None
        assert ep._utterance_ended_at is None

    def test_on_end_of_agent_speech_clears_state(self) -> None:
        """on_end_of_agent_speech sets ended_at, clears started_at and overlapping."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)

        ep.on_start_of_agent_speech(started_at=100.0)
        ep.on_start_of_speech(started_at=100.1, overlapping=True)
        assert ep._overlapping is True
        assert ep._agent_speech_started_at == 100.0

        ep.on_end_of_agent_speech(ended_at=101.0)

        assert ep._agent_speech_ended_at == 101.0
        assert ep._agent_speech_started_at is None
        assert ep._overlapping is False

    def test_overlapping_inferred_from_agent_speech(self) -> None:
        """When _agent_speech_started_at is set, on_end_of_speech takes the interruption path."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        ep.on_end_of_speech(ended_at=100.0)
        ep.on_start_of_agent_speech(started_at=100.9)
        # overlapping not explicitly set
        ep.on_start_of_speech(started_at=101.8, overlapping=False)
        ep.on_end_of_speech(ended_at=102.0)

        # _agent_speech_started_at is set → interruption path → case 3 (delayed) updates max_delay
        # between_turn_delay = 100.9 - 100.0 = 0.9
        assert ep.max_delay == pytest.approx(0.5 * 0.9 + 0.5 * 1.0, rel=1e-5)

    def test_speaking_flag_set_and_cleared(self) -> None:
        """_speaking is True after on_start_of_speech, False after on_end_of_speech."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0)

        assert ep._speaking is False
        ep.on_start_of_speech(started_at=100.0)
        assert ep._speaking is True
        ep.on_end_of_speech(ended_at=100.5)
        assert ep._speaking is False

    @pytest.mark.parametrize(
        "label, agent_speech, overlapping, should_ignore, within_grace, expect_min_change, expect_max_change",
        [
            # --- No agent speech ---
            # Case 1: pause between utterances updates min_delay
            ("no_agent/no_overlap/no_ignore", "none", False, False, False, True, False),
            # should_ignore is ignored when not overlapping
            ("no_agent/no_overlap/ignore", "none", False, True, False, True, False),
            # --- Agent speech ended (on_end_of_agent_speech called) ---
            # agent_speech_ended_at blocks case 1, no agent_speech_started_at blocks case 3
            ("agent_ended/no_overlap/no_ignore", "ended", False, False, False, False, False),
            ("agent_ended/no_overlap/ignore", "ended", False, True, False, False, False),
            # --- Agent speech active ---
            # Inferred interruption from agent_speech_started_at → case 3 (delayed)
            ("agent_active/no_overlap/no_ignore", "active", False, False, False, False, True),
            # should_ignore ignored when not _overlapping
            ("agent_active/no_overlap/ignore", "active", False, True, False, False, True),
            # Explicit overlapping, immediate → case 2 updates min_delay
            ("agent_active/overlap/no_ignore", "active", True, False, False, True, False),
            # Backchannel: overlapping + should_ignore outside grace → skip
            (
                "agent_active/overlap/ignore/outside_grace",
                "active",
                True,
                True,
                False,
                False,
                False,
            ),
            # Grace period override: overlapping + should_ignore inside grace → case 2 still runs
            ("agent_active/overlap/ignore/inside_grace", "active", True, True, True, True, False),
        ],
    )
    def test_all_overlapping_and_should_ignore_combos(
        self,
        label: str,
        agent_speech: str,
        overlapping: bool,
        should_ignore: bool,
        within_grace: bool,
        expect_min_change: bool,
        expect_max_change: bool,
    ) -> None:
        """Exhaustive test of all agent_speech × overlapping × should_ignore combinations."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        # Previous utterance
        ep.on_start_of_speech(started_at=99.0)
        ep.on_end_of_speech(ended_at=100.0)

        # Set up agent speech state
        if agent_speech == "ended":
            ep.on_start_of_agent_speech(started_at=100.5)
            ep.on_end_of_agent_speech(ended_at=101.0)
            user_start = 101.5
        elif agent_speech == "active":
            if within_grace:
                # Agent at 100.15, user at 100.35 (0.2s after agent, within 0.25s grace)
                # between_turn_delay=0.15, between_utterance_delay=0.35
                # interruption_delay=|0.35-0.15|=0.2 <= 0.3 → case 2 triggers
                # EMA: 0.5*0.35 + 0.5*0.3 = 0.325 → min changes
                ep.on_start_of_agent_speech(started_at=100.15)
                user_start = 100.35
            elif overlapping and should_ignore:
                # Outside grace: agent at 100.2, user at 101.5 (1.3s after agent)
                # should_ignore + overlapping + outside grace → skip
                ep.on_start_of_agent_speech(started_at=100.2)
                user_start = 101.5
            elif overlapping:
                # Agent at 100.15, user at 100.4 (0.25s after agent, at grace boundary)
                # between_turn_delay=0.15, between_utterance_delay=0.4
                # interruption_delay=|0.4-0.15|=0.25 <= 0.3 → case 2 triggers
                # EMA: 0.5*0.4 + 0.5*0.3 = 0.35 → min changes
                ep.on_start_of_agent_speech(started_at=100.15)
                user_start = 100.4
            else:
                # Delayed: agent spoke but user starts much later (inferred interruption)
                # between_turn_delay=0.9 → case 3 updates max_delay
                ep.on_start_of_agent_speech(started_at=100.9)
                user_start = 101.8
        else:
            # No agent speech
            user_start = 100.4

        ep.on_start_of_speech(started_at=user_start, overlapping=overlapping)

        prev_min = ep.min_delay
        prev_max = ep.max_delay

        ep.on_end_of_speech(ended_at=user_start + 0.5, should_ignore=should_ignore)

        min_changed = ep.min_delay != prev_min
        max_changed = ep.max_delay != prev_max

        assert min_changed == expect_min_change, (
            f"[{label}] min_delay {'should' if expect_min_change else 'should not'} change: "
            f"{prev_min} -> {ep.min_delay}"
        )
        assert max_changed == expect_max_change, (
            f"[{label}] max_delay {'should' if expect_max_change else 'should not'} change: "
            f"{prev_max} -> {ep.max_delay}"
        )

        # State should always be cleaned up after on_end_of_speech
        assert ep._speaking is False, f"[{label}] _speaking should be False"
        assert ep._overlapping is False, f"[{label}] _overlapping should be False"

    def test_full_conversation_sequence(self) -> None:
        """Simulate a realistic multi-turn conversation with backchannel ignored."""
        ep = DynamicEndpointing(min_delay=0.3, max_delay=1.0, alpha=0.5)

        # Turn 1: user speaks
        ep.on_start_of_speech(started_at=100.0)
        ep.on_end_of_speech(ended_at=101.0)

        # Agent responds
        ep.on_start_of_agent_speech(started_at=101.5)

        # Turn 2: user backchannel (ignored) — overlapping with agent, 1.0s after agent start
        ep.on_start_of_speech(started_at=102.5, overlapping=True)
        min_before_backchannel = ep.min_delay
        max_before_backchannel = ep.max_delay
        ep.on_end_of_speech(ended_at=102.8, should_ignore=True)

        # backchannel ignored — delays unchanged
        assert ep.min_delay == min_before_backchannel
        assert ep.max_delay == max_before_backchannel

        # Agent finishes
        ep.on_end_of_agent_speech(ended_at=103.0)

        # Turn 3: user speaks again (new turn after agent)
        ep.on_start_of_speech(started_at=103.5)
        ep.on_end_of_speech(ended_at=104.0)

        # agent_speech_ended_at was set, agent_speech_started_at is None
        # This is a normal (non-interruption) path
        # No agent_speech_started_at → case 1 (between utterances) doesn't apply
        # because _utterance_ended_at was reset to None by the ignored backchannel
        # so between_utterance_delay = 0 → no update
        assert ep._speaking is False
        assert ep._agent_speech_started_at is None
