from __future__ import annotations

HG_MODEL = "KittenML/kitten-tts-nano-0.2"


ONNX_FILENAME = "kitten_tts_nano_v0_2.onnx"
VOICES_FILENAME = "voices.npz"


def resolve_model_name(model_name: str) -> str:
    legacy_models = {
        "nano-0.1": "KittenML/kitten-tts-nano-0.1",
        "nano-0.2": "KittenML/kitten-tts-nano-0.2",
    }

    if model_name in legacy_models:
        return legacy_models[model_name]

    if "/" not in model_name:
        return f"KittenML/{model_name}"

    return model_name
