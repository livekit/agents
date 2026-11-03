from __future__ import annotations

import pytest

from livekit.agents.utils.aio import interval


@pytest.mark.parametrize("period", [0, -1, -0.5, float("nan")])
def test_a_non_positive_interval_is_named(period: float) -> None:
    with pytest.raises(ValueError, match="interval must be a positive number"):
        interval(period)


def test_a_positive_interval_still_waits() -> None:
    ticker = interval(0.01)

    assert ticker._interval == 0.01
