from livekit.agents.inference.tts import TTS, ConnectionOptions, Fallback, FallbackModel


def _make_inference_tts(fallback_input):
    return TTS(
        model="cartesia/sonic",
        voice="primary",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
        fallback=fallback_input,
    )


def test_tts_fallback_from_string_names() -> None:
    tts = _make_inference_tts(["cartesia/sonic", "elevenlabs/eleven_flash_v2"])

    fallback = tts._opts.fallback
    assert isinstance(fallback, Fallback)
    assert fallback.models == ["cartesia/sonic", "elevenlabs/eleven_flash_v2"]
    assert fallback.to_dict() == {
        "models": [
            {"name": "cartesia/sonic"},
            {"name": "elevenlabs/eleven_flash_v2"},
        ]
    }


def test_tts_fallback_from_model_objects() -> None:
    fallback_models = [
        FallbackModel(
            name="cartesia/sonic",
            voice="sonic-voice",
            extra={"speed": "fast"},
        ),
        FallbackModel(
            name="elevenlabs/eleven_flash_v2",
            voice="flash-voice",
            extra={"style": "casual"},
        ),
    ]

    tts = _make_inference_tts(fallback_models)

    fallback = tts._opts.fallback
    assert isinstance(fallback, Fallback)
    assert all(isinstance(model, FallbackModel) for model in fallback.models)

    models_dict = fallback.to_dict()["models"]
    assert models_dict[0]["voice"] == "sonic-voice"
    assert models_dict[0]["extra"]["speed"] == "fast"
    assert models_dict[1]["voice"] == "flash-voice"
    assert models_dict[1]["extra"]["style"] == "casual"


def test_fallback_from_fallback_object() -> None:
    fallback_obj = Fallback(
        models=["cartesia/sonic"],
        connection=ConnectionOptions(timeout=0.5, retries=1),
    )

    tts = _make_inference_tts(fallback_obj)

    fallback = tts._opts.fallback
    assert fallback is fallback_obj
    assert fallback.to_dict()["connection"] == {"timeout": 0.5, "retries": 1}

    fallback_model_obj = Fallback(
        models=[FallbackModel(name="cartesia/sonic", voice="primary")],
        connection=ConnectionOptions(timeout=0.25, retries=2),
    )

    tts = _make_inference_tts(fallback_model_obj)
    assert tts._opts.fallback is fallback_model_obj
    assert tts._opts.fallback.to_dict()["connection"] == {"timeout": 0.25, "retries": 2}
