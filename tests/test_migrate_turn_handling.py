"""Unit tests for the deprecated keyword-argument migration shim.

Targets `_migrate_turn_handling`, which converts AgentSession/Agent's
deprecated top-level keyword arguments (`preemptive_generation=`,
`min_endpointing_delay=`, etc.) into the `TurnHandlingOptions` dict shape.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from livekit.agents.voice.turn import _migrate_turn_handling


class TestMigratePreemptiveGeneration:
    """Regression tests for https://github.com/livekit/agents/issues/6112.

    The deprecated `preemptive_generation` kwarg accepts both a bare bool
    (legacy shorthand for `{"enabled": <bool>}`) and a full
    `PreemptiveGenerationOptions` dict. Before the fix, the dict form was
    re-wrapped as `{"enabled": <dict>}` and the resolution chain saw a
    truthy dict in the `enabled` slot, so the session report always
    reported `enabled: True` regardless of what the caller passed.
    """

    def test_bool_false_wraps_to_enabled_false(self) -> None:
        result = _migrate_turn_handling(preemptive_generation=False)
        assert result == {"preemptive_generation": {"enabled": False}}

    def test_bool_true_wraps_to_enabled_true(self) -> None:
        result = _migrate_turn_handling(preemptive_generation=True)
        assert result == {"preemptive_generation": {"enabled": True}}

    def test_dict_passes_through_without_re_wrapping(self) -> None:
        """The bug — passing `{"enabled": False}` produced
        `{"enabled": {"enabled": False}}` which the resolver merged into
        a truthy `enabled` value."""
        result = _migrate_turn_handling(
            preemptive_generation={"enabled": False},
        )
        assert result == {"preemptive_generation": {"enabled": False}}

    def test_dict_with_extra_fields_passes_through_untouched(self) -> None:
        """`PreemptiveGenerationOptions` has more than just `enabled`
        (`preemptive_tts`, `max_speech_duration`, `max_retries`); the dict
        form must let callers set those too."""
        opts = {
            "enabled": True,
            "preemptive_tts": True,
            "max_speech_duration": 5.0,
            "max_retries": 1,
        }
        result = _migrate_turn_handling(preemptive_generation=opts)
        assert result == {"preemptive_generation": opts}

    def test_omitted_yields_no_preemptive_generation_key(self) -> None:
        result = _migrate_turn_handling()
        assert "preemptive_generation" not in result
