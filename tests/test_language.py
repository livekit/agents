import pytest

from livekit.agents import LanguageCode
from livekit.agents.stt import SpeechData

pytestmark = pytest.mark.unit


def test_provider_language_tags_keep_their_dialect_and_resolve_the_base_language() -> None:
    language = LanguageCode("cmn-hans-cn")
    assert language == "cmn-Hans-CN"
    assert language.language == "zh"
    assert language.iso == "zh-CN"
    assert language.to_language_name() == "chinese"


def test_speech_data_normalizes_a_language_name() -> None:
    data = SpeechData(language="english", text="hello")
    assert data.language == "en"
