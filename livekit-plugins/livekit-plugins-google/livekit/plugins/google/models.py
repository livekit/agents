from typing import Literal

ChatModels = Literal[ 
    "gemini-pro"
]

# Speech to Text v2

SpeechModels = Literal[
    "long", "short", "telephony", "medical_dictation", "medical_conversation", "chirp"
]

SpeechLanguages = Literal[
    "en-US",
    "ja-JP",
    "en-IN",
    "en-GB",
    "hi-IN",
    "af-ZA",
    "sq-AL",
    "am-ET",
    "ar-EG",
    "hy-AM",
    "ast-ES",
    "az-AZ",
    "eu-ES",
    "be-BY",
    "bs-BA",
    "bg-BG",
    "my-MM",
    "ca-ES",
    "ceb-PH",
    "ckb-IQ",
    "zh-Hans-CN",
    "yue-Hant-HK",
    "zh-TW",
    "hr-HR",
    "cs-CZ",
    "da-DK",
    "nl-NL",
    "en-AU",
    "et-EE",
    "fil-PH",
    "fi-FI",
    "fr-CA",
    "fr-FR",
    "gl-ES",
    "ka-GE",
    "de-DE",
    "el-GR",
    "gu-IN",
    "ha-NG",
    "iw-IL",
    "hi-IN",
    "hu-HU",
    "is-IS",
    "id-ID",
    "it-IT",
    "ja-JP",
    "jv-ID",
    "kea-CV",
    "kam-KE",
    "kn-IN",
    "kk-KZ",
    "km-KH",
    "ko-KR",
    "ky-KG",
    "lo-LA",
    "lv-LV",
    "ln-CD",
    "lt-LT",
    "luo-KE",
    "lb-LU",
    "mk-MK",
    "no-NO",
    "pl-PL",
    "pt-BR",
    "pt-PT",
    "ro-RO",
    "ru-RU",
    "es-CO",
    "es-MX",
    "es-US",
    "th-TH",
    "tr-TR",
    "uk-UA",
    "vi-VN",
    "da-DK",
]

Gender = Literal["male", "female", "neutral"]

AudioEncoding = Literal["wav", "mp3", "ogg", "mulaw", "alaw", "linear16"]
