import pytest

from livekit.plugins.slng import STT

pytestmark = pytest.mark.unit


def test_model_reports_the_configured_model():
    assert STT(api_key="test-key", model="deepgram/nova:3").model == "deepgram/nova:3"
