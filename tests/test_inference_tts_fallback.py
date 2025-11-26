from livekit.agents.inference.tts import (
    TTS,
    ConnectionOptions,
    Fallback,
    FallbackModel,
    _normalize_fallback,
)


def _make_inference_tts(fallback_input):
    return TTS(
        model="cartesia/sonic",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
        fallback=fallback_input,
    )

def test_normalize_fallback_from_string_names() -> None:
    result = _normalize_fallback(["cartesia/sonic", "elevenlabs/eleven_flash_v2"])

    assert isinstance(result, dict)
    assert result["models"] == [
        {"name": "cartesia/sonic", "voice": None},
        {"name": "elevenlabs/eleven_flash_v2", "voice": None},
    ]
    assert result["connection"] == {}


def test_normalize_fallback_from_string_with_voice() -> None:
    result = _normalize_fallback(["cartesia/sonic:my-voice-id", "rime/mist:voice-123"])

    assert result["models"] == [
        {"name": "cartesia/sonic", "voice": "my-voice-id"},
        {"name": "rime/mist", "voice": "voice-123"},
    ]


def test_normalize_fallback_mixed_voice_and_no_voice() -> None:
    result = _normalize_fallback([
        "cartesia/sonic:my-voice",
        "elevenlabs/eleven_flash_v2",
        "rime/mist:another-voice",
    ])

    assert result["models"] == [
        {"name": "cartesia/sonic", "voice": "my-voice"},
        {"name": "elevenlabs/eleven_flash_v2", "voice": None},
        {"name": "rime/mist", "voice": "another-voice"},
    ]


def test_normalize_fallback_from_model_objects() -> None:
    fallback_models: list[FallbackModel] = [
        FallbackModel(name="cartesia/sonic", voice="voice-1"),
        FallbackModel(name="elevenlabs/eleven_flash_v2", voice="voice-2"),
    ]

    result = _normalize_fallback(fallback_models)

    assert result["models"] == fallback_models


def test_normalize_fallback_model_with_extra_kwargs() -> None:
    fallback_models: list[FallbackModel] = [
        FallbackModel(
            name="cartesia/sonic",
            voice="my-voice",
            extra_kwargs={"duration": 30.0, "speed": "fast"},
        ),
        FallbackModel(
            name="elevenlabs/eleven_flash_v2",
            voice="other-voice",
            extra_kwargs={"inactivity_timeout": 120},
        ),
    ]

    result = _normalize_fallback(fallback_models)

    assert result["models"][0]["extra_kwargs"] == {"duration": 30.0, "speed": "fast"}
    assert result["models"][1]["extra_kwargs"] == {"inactivity_timeout": 120}


def test_normalize_fallback_model_without_optional_fields() -> None:
    fallback_models: list[FallbackModel] = [
        FallbackModel(name="cartesia/sonic", voice="v1"),
    ]

    result = _normalize_fallback(fallback_models)

    assert result["models"][0] == {"name": "cartesia/sonic", "voice": "v1"}


def test_normalize_fallback_mixed_strings_and_models() -> None:
    result = _normalize_fallback([
        "cartesia/sonic:voice-from-string",
        FallbackModel(name="elevenlabs/eleven_flash_v2", voice="voice-from-dict"),
        "rime/mist",
    ])

    assert result["models"] == [
        {"name": "cartesia/sonic", "voice": "voice-from-string"},
        {"name": "elevenlabs/eleven_flash_v2", "voice": "voice-from-dict"},
        {"name": "rime/mist", "voice": None},
    ]

def test_normalize_fallback_from_fallback_dict_models_only() -> None:
    fallback_input: Fallback = Fallback(
        models=["cartesia/sonic:voice1", "rime/mist"],
    )

    result = _normalize_fallback(fallback_input)

    assert result["models"] == [
        {"name": "cartesia/sonic", "voice": "voice1"},
        {"name": "rime/mist", "voice": None},
    ]
    assert result["connection"] == {}


def test_normalize_fallback_from_fallback_dict_with_connection() -> None:
    fallback_input: Fallback = Fallback(
        models=[FallbackModel(name="cartesia/sonic", voice="v1")],
        connection=ConnectionOptions(timeout=5.0, retries=3),
    )

    result = _normalize_fallback(fallback_input)

    assert result["connection"] == {"timeout": 5.0, "retries": 3}


def test_normalize_fallback_from_fallback_dict_with_partial_connection() -> None:
    fallback_input: Fallback = Fallback(
        models=["cartesia/sonic:v1"],
        connection=ConnectionOptions(timeout=2.5),
    )

    result = _normalize_fallback(fallback_input)

    assert result["connection"] == {"timeout": 2.5}


def test_normalize_fallback_from_fallback_dict_with_retries_only() -> None:
    fallback_input: Fallback = Fallback(
        models=["cartesia/sonic:v1"],
        connection=ConnectionOptions(retries=5),
    )

    result = _normalize_fallback(fallback_input)

    assert result["connection"] == {"retries": 5}


def test_normalize_fallback_preserves_empty_connection() -> None:
    fallback_input: Fallback = Fallback(
        models=["cartesia/sonic:v1"],
        connection=ConnectionOptions(),
    )

    result = _normalize_fallback(fallback_input)

    assert result["connection"] == {}


def test_normalize_fallback_empty_list() -> None:
    result = _normalize_fallback([])

    assert result["models"] == []
    assert result["connection"] == {}


def test_normalize_fallback_single_model_string() -> None:
    result = _normalize_fallback(["cartesia/sonic:voice1"])

    assert len(result["models"]) == 1
    assert result["models"][0] == {"name": "cartesia/sonic", "voice": "voice1"}


def test_normalize_fallback_single_model_dict() -> None:
    result = _normalize_fallback([FallbackModel(name="cartesia/sonic", voice="v1")])

    assert len(result["models"]) == 1
    assert result["models"][0] == {"name": "cartesia/sonic", "voice": "v1"}


def test_normalize_fallback_model_string_no_provider_prefix() -> None:
    result = _normalize_fallback(["custom-model:voice1"])

    assert result["models"][0] == {"name": "custom-model", "voice": "voice1"}


def test_normalize_fallback_model_string_empty_voice() -> None:
    result = _normalize_fallback(["cartesia/sonic:"])
    assert result["models"][0] == {"name": "cartesia/sonic", "voice": ""}


def test_normalize_fallback_preserves_model_order() -> None:
    models = [
        "model-1:v1",
        "model-2:v2",
        "model-3:v3",
        "model-4:v4",
        "model-5:v5",
    ]

    result = _normalize_fallback(models)

    for i, model in enumerate(result["models"]):
        assert model["name"] == f"model-{i + 1}"
        assert model["voice"] == f"v{i + 1}"


def test_tts_fallback_from_string_names() -> None:
    tts = _make_inference_tts(["cartesia/sonic:v1", "elevenlabs/eleven_flash_v2"])

    fallback = tts._opts.fallback
    assert isinstance(fallback, dict)
    assert fallback["models"] == [
        {"name": "cartesia/sonic", "voice": "v1"},
        {"name": "elevenlabs/eleven_flash_v2", "voice": None},
    ]


def test_tts_fallback_from_model_objects() -> None:
    fallback_models: list[FallbackModel] = [
        FallbackModel(name="cartesia/sonic", voice="voice-1", extra_kwargs={"speed": "fast"}),
        FallbackModel(
            name="elevenlabs/eleven_flash_v2",
            voice="voice-2",
            extra_kwargs={"apply_text_normalization": "on"},
        ),
    ]

    tts = _make_inference_tts(fallback_models)

    fallback = tts._opts.fallback
    assert isinstance(fallback, dict)
    assert fallback["models"][0]["extra_kwargs"]["speed"] == "fast"
    assert fallback["models"][1]["extra_kwargs"]["apply_text_normalization"] == "on"


def test_tts_fallback_passes_through_existing_instance() -> None:
    fallback_obj: Fallback = Fallback(
        models=[FallbackModel(name="cartesia/sonic", voice="my-voice")],
        connection=ConnectionOptions(timeout=0.25, retries=2),
    )

    tts = _make_inference_tts(fallback_obj)

    fallback = tts._opts.fallback
    assert fallback["connection"] == {"timeout": 0.25, "retries": 2}
    assert fallback["models"][0]["voice"] == "my-voice"


def test_tts_no_fallback() -> None:
    from livekit.agents.utils import is_given

    tts = TTS(
        model="cartesia/sonic",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
    )

    assert not is_given(tts._opts.fallback)
