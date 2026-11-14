"""Public model selection for the original and conversational STT implementations."""

from typing import Any, Literal

from .stt import STT as LegacySTT
from .stt_conv import STTConv

NabrahRecognitionModel = Literal["nabrah_stt_v1", "eot_nabrah", "nabrah_stt_conv"]


def STT(
    *, recognition_model: NabrahRecognitionModel = "eot_nabrah", **options: Any
) -> LegacySTT | STTConv:
    """Select the implementation matching recognition_model, preserving the original default.

    api_key, base_url and http_session are shared options. STTConv fixes metadata to ar-SA.
    Other options are passed to the selected implementation.
    """
    if recognition_model not in ("nabrah_stt_v1", "eot_nabrah", "nabrah_stt_conv"):
        raise ValueError(f"Unsupported Nabrah recognition_model: {recognition_model!r}")
    if recognition_model == "nabrah_stt_conv":
        return STTConv(**options)
    return LegacySTT(recognition_model=recognition_model, **options)
