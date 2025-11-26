from livekit.agents.inference.stt import STT, ConnectionOptions, Fallback, FallbackModel


def _make_inference_stt(fallback_input):
    return STT(
        model="deepgram",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
        fallback=fallback_input,
    )


def test_fallback_from_string_names() -> None:
    stt = _make_inference_stt(["deepgram/nova-3", "cartesia/ink-whisper"])

    fallback = stt._opts.fallback
    assert isinstance(fallback, Fallback)
    assert fallback.models == ["deepgram/nova-3", "cartesia/ink-whisper"]
    assert fallback.to_dict() == {
        "models": [
            {"name": "deepgram/nova-3"},
            {"name": "cartesia/ink-whisper"},
        ]
    }


def test_fallback_from_model_objects() -> None:
    fallback_models = [
        FallbackModel(name="deepgram/nova-3", extra={"keywords": ["livekit"]}),
        FallbackModel(name="cartesia/ink-whisper", extra={"max_silence_duration_secs": 1.5}),
    ]

    stt = _make_inference_stt(fallback_models)

    fallback = stt._opts.fallback
    assert isinstance(fallback, Fallback)
    assert all(isinstance(model, FallbackModel) for model in fallback.models)

    models_dict = fallback.to_dict()["models"]
    assert models_dict[0]["extra"]["keywords"] == ["livekit"]
    assert models_dict[1]["extra"]["max_silence_duration_secs"] == 1.5


def test_fallback_passes_through_existing_instance() -> None:
    fallback_obj = Fallback(
        models=["deepgram/nova-3"],
        connection=ConnectionOptions(timeout=0.25, retries=2),
    )

    stt = _make_inference_stt(fallback_obj)

    fallback = stt._opts.fallback
    assert fallback is fallback_obj
    assert fallback.to_dict()["connection"] == {"timeout": 0.25, "retries": 2}
