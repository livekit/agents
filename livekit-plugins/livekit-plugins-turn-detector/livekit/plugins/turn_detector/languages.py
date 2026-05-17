# Per-language "unlikely" thresholds. Calibrated separately per checkpoint —
# do NOT unify.

CLOUD_LANGUAGES: dict[str, float] = {
    "en": 0.4,
    "fr": 0.4,
    "de": 0.4,
    "hi": 0.4,
    "ja": 0.4,
    "ko": 0.4,
    "zh": 0.4,
    "es": 0.4,
}

LOCAL_LANGUAGES: dict[str, float] = {
    "en": 0.3,
    "fr": 0.3,
    "de": 0.3,
    "hi": 0.3,
    "ja": 0.3,
    "ko": 0.3,
    "zh": 0.3,
    "es": 0.3,
}
