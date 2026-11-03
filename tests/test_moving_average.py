from __future__ import annotations

import pytest

from livekit.agents.utils import MovingAverage


@pytest.mark.parametrize("window_size", [0, -1, -3])
def test_non_positive_window_size_is_rejected_at_construction(window_size: int) -> None:
    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        MovingAverage(window_size)


def test_average_over_the_window() -> None:
    average = MovingAverage(2)

    average.add_sample(1.0)
    average.add_sample(3.0)

    assert average.get_avg() == 2.0
    assert average.size() == 2
