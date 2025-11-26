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
    # Fallback is a TypedDict, so it's just a dict at runtime
    assert isinstance(fallback, dict)
    # Strings are converted to FallbackModel dicts
    assert fallback["models"] == [
        {"name": "deepgram/nova-3"},
        {"name": "cartesia/ink-whisper"},
    ]


def test_fallback_from_model_objects() -> None:
    fallback_models: list[FallbackModel] = [
        FallbackModel(name="deepgram/nova-3", extra_kwargs={"keywords": ["livekit"]}),
        FallbackModel(name="cartesia/ink-whisper", extra_kwargs={"max_silence_duration_secs": 1.5}),
    ]

    stt = _make_inference_stt(fallback_models)

    fallback = stt._opts.fallback
    assert isinstance(fallback, dict)
    assert fallback["models"][0]["extra_kwargs"]["keywords"] == ["livekit"]
    assert fallback["models"][1]["extra_kwargs"]["max_silence_duration_secs"] == 1.5


def test_fallback_passes_through_existing_instance() -> None:
    fallback_obj: Fallback = Fallback(
        models=[FallbackModel(name="deepgram/nova-3")],
        connection=ConnectionOptions(timeout=0.25, retries=2),
    )

    stt = _make_inference_stt(fallback_obj)

    fallback = stt._opts.fallback
    assert fallback["connection"] == {"timeout": 0.25, "retries": 2}
