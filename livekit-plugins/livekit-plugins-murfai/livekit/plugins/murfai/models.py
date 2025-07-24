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

TTSModels = Literal["GEN2",]

TTSStyles = Literal[
    "Promo",
    "Narration",
    "Calm",
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

TTSDefaultVoiceId = "en-US-amara"
TTSDefaultVoiceStyle = "Conversational"
