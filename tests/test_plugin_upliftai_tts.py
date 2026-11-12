import pytest

from livekit.plugins.upliftai import TTS

pytestmark = pytest.mark.plugin("upliftai")


def test_reports_provider():
    assert TTS(api_key="test-key").provider == "UpliftAI"
