"""Each plugin picks the markup dialect its live options make it read.

The key drives both how ``say()`` markers are lowered for the TTS and which native tags
are stripped from the transcript and chat history.
"""

from __future__ import annotations

import pytest

from livekit.plugins import aws, azure, elevenlabs, google

pytestmark = pytest.mark.unit


def test_google_markup_key() -> None:
    creds = {"type": "service_account"}
    assert google.TTS(credentials_info=creds).markup._provider_key() == ""
    assert (
        google.TTS(
            credentials_info=creds, enable_ssml=True, use_streaming=False
        ).markup._provider_key()
        == "ssml"
    )
    assert google.TTS(credentials_info=creds, use_markup=True).markup._provider_key() == (
        "bracket_pauses"
    )


def test_azure_markup_key() -> None:
    assert azure.TTS(speech_key="k", speech_region="eastus").markup._provider_key() == "ssml"


def test_aws_markup_key_follows_text_type() -> None:
    tts = aws.TTS(api_key="a", api_secret="b", region="us-east-1")
    assert tts.markup._provider_key() == ""
    tts.update_options(text_type="ssml")
    assert tts.markup._provider_key() == "ssml"


@pytest.mark.parametrize(
    ("model", "ssml", "key"),
    [
        ("eleven_flash_v2_5", False, ""),
        ("eleven_flash_v2_5", True, "ssml"),
        ("eleven_v3", False, "eleven_v3"),
        ("eleven_v3", True, "eleven_v3"),  # v3 reads cues, not SSML
    ],
)
def test_elevenlabs_markup_key(model: str, ssml: bool, key: str) -> None:
    tts = elevenlabs.TTS(api_key="k", model=model, enable_ssml_parsing=ssml)
    assert tts.markup._provider_key() == key
