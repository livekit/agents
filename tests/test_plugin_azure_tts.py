import pytest

pytestmark = pytest.mark.plugin("azure")

VOICE = "en-US-JennyNeural"


@pytest.fixture(autouse=True)
def _clear_azure_env(monkeypatch):
    # clearing in order to make sure naming rules behave correctly (ci exports these)
    for var in ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_SPEECH_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)


def _tts(**kwargs):
    from livekit.plugins.azure import TTS

    return TTS(voice=VOICE, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # region path (key + region)
        ({"speech_key": "k", "speech_region": "eastus"}, f"eastus:{VOICE}"),
        # endpoint path: host from a well-formed URL
        (
            {"speech_endpoint": "https://eastus.tts.speech.microsoft.com/cognitiveservices/v1"},
            f"eastus.tts.speech.microsoft.com:{VOICE}",
        ),
        # both set: endpoint wins over region
        (
            {
                "speech_key": "k",
                "speech_region": "eastus",
                "speech_endpoint": "https://custom.example.com/v1",
            },
            f"custom.example.com:{VOICE}",
        ),
        # no scheme endpoint + path: host-only
        (
            {"speech_endpoint": "eastus.tts.speech.microsoft.com/path"},
            f"eastus.tts.speech.microsoft.com:{VOICE}",
        ),
        # surrounding whitespaces stripped
        (
            {"speech_endpoint": "  https://eastus.tts.speech.microsoft.com  "},
            f"eastus.tts.speech.microsoft.com:{VOICE}",
        ),
        # deployment id appended to endpoint
        (
            {"speech_endpoint": "https://custom.example.com/v1", "deployment_id": "cnv-123"},
            f"custom.example.com:{VOICE}:cnv-123",
        ),
        # deployment id appended to region
        (
            {"speech_key": "k", "speech_region": "eastus", "deployment_id": "cnv-123"},
            f"eastus:{VOICE}:cnv-123",
        ),
    ],
)
def test_model_derivation(kwargs, expected):
    assert _tts(**kwargs).model == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rate": 1},
        {"rate": 2},
        {"rate": 0.75},
        {"rate": "fast"},
        {"volume": 0},
        {"volume": 50},
        {"volume": 100.0},
        {"volume": "loud"},
    ],
)
def test_prosody_accepts_valid_values(kwargs):
    from livekit.plugins.azure.tts import ProsodyConfig

    ProsodyConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        # integers are range-checked like floats
        ({"rate": 3}, "between 0.5 and 2"),
        ({"rate": 0}, "between 0.5 and 2"),
        ({"volume": 150}, "between 0 and 100"),
        ({"volume": -1}, "between 0 and 100"),
        # bool is an int subclass, not a prosody value
        ({"rate": True}, "must be one of"),
        ({"volume": False}, "must be one of"),
        # an empty string used to skip validation and render as rate=""
        ({"rate": ""}, "must be one of"),
    ],
)
def test_prosody_rejects_invalid_values(kwargs, match):
    from livekit.plugins.azure.tts import ProsodyConfig

    with pytest.raises(ValueError, match=match):
        ProsodyConfig(**kwargs)
