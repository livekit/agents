from typing import Literal

TTSEncoding = Literal[
    "pcm_s16le",
    # Not yet supported
    # "pcm_f32le",
    # "pcm_mulaw",
    # "pcm_alaw",
]

TTSModels = Literal["sonic-english", "sonic-multilingual"]
TTSLanguages = Literal["en", "es", "fr", "de", "pt", "zh", "ja"]
TTSDefaultVoiceId = "c2ac25f9-ecc4-4f56-9095-651354df60c0"
TTSVoiceSpeed = Literal["fastest", "fast", "normal", "slow", "slowest"]
TTSVoiceEmotion = Literal[
    "anger:lowest",
    "anger:low",
    "anger",
    "anger:high",
    "anger:highest",
    "positivity:lowest",
    "positivity:low",
    "positivity",
    "positivity:high",
    "positivity:highest",
    "surprise:lowest",
    "surprise:low",
    "surprise",
    "surprise:high",
    "surprise:highest",
    "sadness:lowest",
    "sadness:low",
    "sadness",
    "sadness:high",
    "sadness:highest",
    "curiosity:lowest",
    "curiosity:low",
    "curiosity",
    "curiosity:high",
    "curiosity:highest",
]
