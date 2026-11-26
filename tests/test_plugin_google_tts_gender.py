from __future__ import annotations

import pytest
from google.cloud.texttospeech import SsmlVoiceGender

from livekit.plugins.google.tts import TTS

pytestmark = pytest.mark.plugin("google")


def test_gender_is_unspecified_when_not_given() -> None:
    """Every Google voice has a fixed gender; NEUTRAL is rejected for Neural2 voices (#7951)."""
    tts = TTS(
        voice_name="en-IN-Neural2-B",
        language="en-IN",
        credentials_info={"type": "service_account"},
    )
    assert tts._opts.voice.ssml_gender == SsmlVoiceGender.SSML_VOICE_GENDER_UNSPECIFIED


@pytest.mark.parametrize(
    "gender, expected",
    [
        ("male", SsmlVoiceGender.MALE),
        ("female", SsmlVoiceGender.FEMALE),
        ("neutral", SsmlVoiceGender.NEUTRAL),
    ],
)
def test_explicit_gender_is_kept(gender: str, expected: SsmlVoiceGender) -> None:
    tts = TTS(gender=gender, credentials_info={"type": "service_account"})
    assert tts._opts.voice.ssml_gender == expected
