from typing import Literal

TTSLocales = Literal[
    "en-US",
    "en-UK",
    "en-AU",
    "en-IN",
    "en-SCOTT",
    "es-ES",
    "es-MX",
    "hi-IN",
    "ta-IN",
    "bn-IN",
    "fr-FR",
    "de-DE",
    "it-IT",
    "pt-BR",
    "zh-CN",
    "nl-NL",
    "ja-JP",
    "id-ID",
    "ko-KR",
    "ro-RO",
    "tr-TR",
    "pl-PL",
    "sk-SK",
    "hr-HR",
    "el-GR",
    "bg-BG",
]

TTSModels = Literal["GEN2", "FALCON"]

TTSStyles = Literal[
    "Promo",
    "Narration",
    "Calm",
    "Conversation",
    "Conversational",
    "Sad",
    "Angry",
    "Sports Commentary",
    "Newscast",
    "Terrified",
    "Inspirational",
    "Customer Support Agent",
    "Narration",
    "Audiobook",
    "Storytelling",
    "Furious",
    "Sobbing",
    "Wizard",
    "Clown",
]

TTSEncoding = Literal[
    "pcm",  # pcm_s16le
]

TTSDefaultVoiceId = "en-US-matthew"
TTSDefaultVoiceStyle = "Conversation"

STTModels = Literal["finch_indic_1.0", "finch_global_1.0"]

STTLanguages = Literal[
    "auto",
    # finch_indic_1.0
    "en_IN",
    "as",
    "bn",
    "gu",
    "hi",
    "kn",
    "ml",
    "mr",
    "or",
    "pa",
    "ta",
    "te",
    # finch_global_1.0
    "en",
    "de",
    "es",
    "fr",
    "it",
    "pt",
    "nl",
]
