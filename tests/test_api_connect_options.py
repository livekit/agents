import pytest

from livekit.agents import APIConnectOptions

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_invalid_liveness_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="liveness_timeout"):
        APIConnectOptions(liveness_timeout=timeout)


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_invalid_final_response_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="final_response_timeout"):
        APIConnectOptions(final_response_timeout=timeout)


def test_final_response_timeout_is_independent_of_connect_timeout():
    options = APIConnectOptions(timeout=0.01, final_response_timeout=3)
    assert options.timeout == 0.01
    assert options.final_response_timeout == 3


def test_existing_positional_options_keep_the_liveness_timeout():
    options = APIConnectOptions(3, 2.0, 0.5, 1.5)
    assert options.timeout == 0.5
    assert options.liveness_timeout == 1.5
    assert options.final_response_timeout == 10.0
