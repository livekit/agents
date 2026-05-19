# Per-language "unlikely" thresholds. Calibrated separately per checkpoint —
# do NOT unify.

CLOUD_LANGUAGES: dict[str, float] = {
    "ar": 0.3500,
    "de": 0.4000,
    "en": 0.4500,
    "es": 0.4100,
    "fr": 0.3900,
    "hi": 0.5350,
    "id": 0.4700,
    "it": 0.3250,
    "ja": 0.4800,
    "ko": 0.5650,
    "nl": 0.4250,
    "pt": 0.4150,
    "tr": 0.3450,
    "zh": 0.4700,
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
