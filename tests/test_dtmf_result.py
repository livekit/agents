"""The native keypad result joins the pressed keys and refuses anything else."""

import pytest

from livekit.agents import _core

pytestmark = pytest.mark.unit


def test_dtmf_results_join_the_keys_and_refuse_a_non_key() -> None:
    assert _core.GetDtmfResult.from_dtmf_inputs(["1", "*", "#"]).user_input == "1 * #"
    with pytest.raises(ValueError, match="not a DTMF key"):
        _core.GetDtmfResult.from_dtmf_inputs(["1", "invalid"])
