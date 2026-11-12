from typing import Literal

TTSModels = Literal["avaz3", "avaz2"]
"""Avaz models: ``avaz3`` returns 48 kHz audio and ``avaz2`` returns 24 kHz audio."""

DEFAULT_MODEL: TTSModels = "avaz3"
DEFAULT_BASE_URL = "https://dashboard.mankyazilim.com.tr/api"

# Avaz returns 16-bit mono PCM at a fixed rate per model.
MODEL_SAMPLE_RATES: dict[str, int] = {
    "avaz3": 48000,
    "avaz2": 24000,
}
