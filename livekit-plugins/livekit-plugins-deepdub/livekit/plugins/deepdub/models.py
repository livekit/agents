from typing import Literal, get_args

DEFAULT_TTS_BASE_URL = "wss://wss.deepdub.ai"
"""US host for streaming text-to-speech. The EU host is ``wss://wss.eu.deepdub.ai``."""

DEFAULT_STT_BASE_URL = "wss://wsgw.deepdub.ai"

TTS_PATH = "/ws"
STT_PATH = "/transcribe-websocket"

TTSModels = Literal["dd-etts-4.0", "dd-etts-3.4", "dd-etts-3.0"]
DEFAULT_TTS_MODEL: TTSModels = "dd-etts-4.0"

STTModels = Literal["stt-egf"]
DEFAULT_STT_MODEL: STTModels = "stt-egf"

TTSSampleRate = Literal[8000, 16000, 22050, 24000, 32000, 36000, 44100, 48000]
TTS_SAMPLE_RATES: frozenset[int] = frozenset(get_args(TTSSampleRate))
DEFAULT_TTS_SAMPLE_RATE: TTSSampleRate = 24000

DEFAULT_STT_SAMPLE_RATE = 16000

DEFAULT_VOICE_PROMPT_ID = "b2abb241-ac92-48bf-a890-fec03f43e209"
"""English preset "Call Center Agent (Neutral)". See https://docs.deepdub.ai/voice-presets."""

DEFAULT_LOCALE = "en-US"

TTSLocales = Literal[
    "ar-LB",
    "ar-QA",
    "ar-SA",
    "ar-SY",
    "cs-CZ",
    "da-DK",
    "de-DE",
    "el-GR",
    "en-AU",
    "en-GB",
    "en-US",
    "es-419",
    "es-ES",
    "es-MX",
    "et-EE",
    "fi-FI",
    "fr-FR",
    "he-IL",
    "hi-IN",
    "hu-HU",
    "id-ID",
    "it-IT",
    "ja-JP",
    "ko-KR",
    "mk-MK",
    "nb-NO",
    "nl-NL",
    "pl-PL",
    "pt-BR",
    "ro-RO",
    "ru-RU",
    "sv-SE",
    "ta-IN",
    "th-TH",
    "tr-TR",
]

DEFAULT_LOCALE_FOR_LANGUAGE: dict[str, str] = {
    "ar": "ar-SA",
    "cs": "cs-CZ",
    "da": "da-DK",
    "de": "de-DE",
    "el": "el-GR",
    "en": "en-US",
    "es": "es-ES",
    "et": "et-EE",
    "fi": "fi-FI",
    "fr": "fr-FR",
    "he": "he-IL",
    "hi": "hi-IN",
    "hu": "hu-HU",
    "id": "id-ID",
    "it": "it-IT",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "mk": "mk-MK",
    "nb": "nb-NO",
    "no": "nb-NO",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "pt": "pt-BR",
    "ro": "ro-RO",
    "ru": "ru-RU",
    "sv": "sv-SE",
    "ta": "ta-IN",
    "th": "th-TH",
    "tr": "tr-TR",
}
"""Locale used when only a base language such as ``"en"`` is given."""

STTLanguages = Literal["en", "he"]
STT_LANGUAGES: frozenset[str] = frozenset(get_args(STTLanguages))
