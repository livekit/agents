# Per-language "unlikely" thresholds for the audio EOT models. Each table is
# tuned against its specific checkpoint — do NOT unify them. The user-facing
# `unlikely_threshold` kwarg on `AudioTurnDetector` scales the local table
# multiplicatively against the cloud default so fallback preserves user intent.

# Cloud model: `eot-audio` (hosted via the LiveKit inference gateway).
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

# Local model: `eot-audio-mini` (in-process ctypes inference).
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
