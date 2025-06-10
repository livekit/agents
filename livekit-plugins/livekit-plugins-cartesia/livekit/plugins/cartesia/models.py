from typing import Literal

TTSEncoding = Literal[
    "pcm_s16le",
    # Not yet supported
    # "pcm_f32le",
    # "pcm_mulaw",
    # "pcm_alaw",
]

TTSModels = Literal["sonic", "sonic-2", "sonic-lite", "sonic-preview", "sonic-turbo"]
TTSLanguages = Literal["en", "es", "fr", "de", "pt", "zh", "ja"]
TTSDefaultVoiceId = "794f9389-aac1-45b6-b726-9d9369183238"
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

# STT model definitions
STTEncoding = Literal["pcm_s16le",]

STTModels = Literal["ink-whisper"]
STTLanguages = Literal[
    "en",
    "de",
    "es",
    "fr",
    "ja",
    "pt",
    "zh",
    "hi",
    "ko",
    "it",
    "nl",
    "pl",
    "ru",
    "sv",
    "tr",
    "tl",
    "bg",
    "ro",
    "ar",
    "cs",
    "el",
    "fi",
    "hr",
    "ms",
    "sk",
    "da",
    "ta",
    "uk",
    "hu",
    "no",
    "vi",
    "bn",
    "th",
    "he",
    "ka",
    "id",
    "te",
    "gu",
    "kn",
    "ml",
    "mr",
    "or",
    "pa",
]
