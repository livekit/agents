from typing import Literal

# Voice gender types
Gender = Literal["Male", "Female"]

# Common languages supported by Edge TTS
EdgeTTSLanguages = Literal[
    "ar-EG",  # Arabic (Egypt)
    "ar-SA",  # Arabic (Saudi Arabia)
    "bg-BG",  # Bulgarian
    "ca-ES",  # Catalan
    "cs-CZ",  # Czech
    "cy-GB",  # Welsh
    "da-DK",  # Danish
    "de-AT",  # German (Austria)
    "de-CH",  # German (Switzerland)
    "de-DE",  # German (Germany)
    "el-GR",  # Greek
    "en-AU",  # English (Australia)
    "en-CA",  # English (Canada)
    "en-GB",  # English (UK)
    "en-HK",  # English (Hong Kong)
    "en-IE",  # English (Ireland)
    "en-IN",  # English (India)
    "en-KE",  # English (Kenya)
    "en-NG",  # English (Nigeria)
    "en-NZ",  # English (New Zealand)
    "en-PH",  # English (Philippines)
    "en-SG",  # English (Singapore)
    "en-TZ",  # English (Tanzania)
    "en-US",  # English (United States)
    "en-ZA",  # English (South Africa)
    "es-AR",  # Spanish (Argentina)
    "es-CO",  # Spanish (Colombia)
    "es-ES",  # Spanish (Spain)
    "es-MX",  # Spanish (Mexico)
    "es-US",  # Spanish (United States)
    "et-EE",  # Estonian
    "fi-FI",  # Finnish
    "fr-BE",  # French (Belgium)
    "fr-CA",  # French (Canada)
    "fr-CH",  # French (Switzerland)
    "fr-FR",  # French (France)
    "ga-IE",  # Irish
    "gu-IN",  # Gujarati
    "he-IL",  # Hebrew
    "hi-IN",  # Hindi
    "hr-HR",  # Croatian
    "hu-HU",  # Hungarian
    "id-ID",  # Indonesian
    "it-IT",  # Italian
    "ja-JP",  # Japanese
    "ko-KR",  # Korean
    "lt-LT",  # Lithuanian
    "lv-LV",  # Latvian
    "mr-IN",  # Marathi
    "ms-MY",  # Malay
    "mt-MT",  # Maltese
    "nb-NO",  # Norwegian
    "nl-BE",  # Dutch (Belgium)
    "nl-NL",  # Dutch (Netherlands)
    "pl-PL",  # Polish
    "pt-BR",  # Portuguese (Brazil)
    "pt-PT",  # Portuguese (Portugal)
    "ro-RO",  # Romanian
    "ru-RU",  # Russian
    "sk-SK",  # Slovak
    "sl-SI",  # Slovenian
    "sv-SE",  # Swedish
    "ta-IN",  # Tamil
    "te-IN",  # Telugu
    "th-TH",  # Thai
    "tr-TR",  # Turkish
    "uk-UA",  # Ukrainian
    "ur-PK",  # Urdu
    "vi-VN",  # Vietnamese
    "zh-CN",  # Chinese (Mainland)
    "zh-HK",  # Chinese (Hong Kong)
    "zh-TW",  # Chinese (Taiwan)
]