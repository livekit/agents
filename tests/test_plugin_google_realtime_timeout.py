import pytest

from livekit.plugins.google.realtime import RealtimeModel

pytestmark = pytest.mark.unit


def test_generate_reply_timeout_is_configurable() -> None:
    assert RealtimeModel(api_key="test")._opts.generate_reply_timeout == 5.0
    model = RealtimeModel(api_key="test", generate_reply_timeout=12.0)
    assert model._opts.generate_reply_timeout == 12.0
