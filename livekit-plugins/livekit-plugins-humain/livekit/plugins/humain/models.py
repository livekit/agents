from __future__ import annotations

from typing import Literal

# TTS models. ``nebula`` is the only one today and returns 24 kHz PCM.
# See https://docs.voice.humain.com/en/models
TTSModels = Literal["nebula"]
DEFAULT_MODEL: TTSModels = "nebula"

# Voice profiles returned by the voice list, by label. Pass a label or the profile UUID.
#   mul_nourah, mul_sara        female, Saudi (Najdi)
#   mul_abdullah, mul_abdulaziz male, Saudi (Najdi)
#   mul_fadi                    male, Levantine
#   mul_yassin                  male, Moroccan
#   mul_youssef                 male, Egyptian
TTSVoices = Literal[
    "mul_nourah",
    "mul_sara",
    "mul_abdullah",
    "mul_abdulaziz",
    "mul_fadi",
    "mul_yassin",
    "mul_youssef",
]
DEFAULT_VOICE: TTSVoices = "mul_nourah"

DEFAULT_API_URL = "https://api.voice.humain.com"
