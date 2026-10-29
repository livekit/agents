"""The native keypad result keeps its value-class behavior."""

import pytest

from livekit.agents import _core

pytestmark = pytest.mark.unit


def test_dtmf_results_construct_compare_and_keep_escaped_representation() -> None:
    result = _core.GetDtmfResult.from_dtmf_inputs(["1", "*", "#"])
    assert result.user_input == "1 * #"
    assert result == _core.GetDtmfResult(user_input="1 * #")
    assert result != _core.GetDtmfResult(user_input="1")
    assert result != object()
    assert result != "1 * #"
    text = "one\n'\"two"
    escaped = _core.GetDtmfResult(user_input=text)
    assert repr(escaped) == f"GetDtmfResult(user_input={text!r})"
    with pytest.raises(ValueError, match="not a DTMF key"):
        _core.GetDtmfResult.from_dtmf_inputs(["1", "invalid"])
