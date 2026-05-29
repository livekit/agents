"""Per-language ``unlikely`` thresholds for the audio EOT detector.

Calibrated separately per checkpoint — do NOT unify CLOUD and LOCAL tables.
"""

from __future__ import annotations

from typing import Literal, cast

from ...language import LanguageCode
from ...types import NotGivenOr
from ...utils.misc import is_given

CLOUD_LANGUAGES: dict[str, float] = {
    "ar": 0.3550,
    "de": 0.4950,
    "en": 0.5600,
    "es": 0.5900,
    "fr": 0.5750,
    "hi": 0.5750,
    "id": 0.4700,
    "it": 0.6400,
    "ja": 0.3700,
    "ko": 0.6950,
    "nl": 0.7500,
    "pt": 0.6650,
    "tr": 0.6500,
    "zh": 0.5900,
}

LOCAL_LANGUAGES: dict[str, float] = {
    "ar": 0.3500,
    "de": 0.2450,
    "en": 0.3600,
    "es": 0.3500,
    "fr": 0.2850,
    "hi": 0.3050,
    "id": 0.3450,
    "it": 0.2300,
    "ja": 0.2950,
    "ko": 0.4000,
    "nl": 0.2000,
    "pt": 0.3200,
    "tr": 0.2550,
    "zh": 0.3550,
}

TurnDetectorModels = Literal["turn-detector", "turn-detector-mini"]
_BASE: dict[TurnDetectorModels, dict[str, float]] = {
    "turn-detector": CLOUD_LANGUAGES,
    "turn-detector-mini": LOCAL_LANGUAGES,
}


def materialize_thresholds(
    user_value: NotGivenOr[float | dict[LanguageCode | str, float]],
    model: TurnDetectorModels,
) -> dict[str, float]:
    """Resolve user override + per-model defaults into a complete per-language map.

    - NOT_GIVEN: returns the bare per-model table.
    - scalar: fills every language with the same value.
    - dict: overrides per-language (keys go through ``LanguageCode`` so
      "English"/"en"/"en-US" collapse to "en"); unmapped languages keep the default.
    """
    base = _BASE[model]
    if not is_given(user_value):
        return dict(base)
    if isinstance(user_value, dict):
        norm = {LanguageCode(k).language: float(v) for k, v in user_value.items()}
        return {lang: norm.get(lang, default) for lang, default in base.items()}
    # mypy 2.1.0 doesn't narrow NotGivenOr[T | dict] through is_given() above.
    return dict.fromkeys(base, float(cast(float, user_value)))


def rescale_for_local_fallback(cloud_thresholds: dict[str, float]) -> dict[str, float]:
    """Preserve the user's cloud-vs-default ratio when promoting local:
    ``local = LOCAL[lang] * (cloud_t / CLOUD[lang])`` per language."""
    return {
        lang: LOCAL_LANGUAGES[lang] * (cloud_t / CLOUD_LANGUAGES[lang])
        for lang, cloud_t in cloud_thresholds.items()
        if lang in LOCAL_LANGUAGES and lang in CLOUD_LANGUAGES and CLOUD_LANGUAGES[lang] != 0
    }
