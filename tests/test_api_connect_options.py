import pytest

from livekit.agents import APIConnectOptions

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_invalid_liveness_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="liveness_timeout"):
        APIConnectOptions(liveness_timeout=timeout)
