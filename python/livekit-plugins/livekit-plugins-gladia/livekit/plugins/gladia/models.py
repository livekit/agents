from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

GladiaModels = Literal["solaria-1"]

GladiaLanguages = Literal[
    "af",
    "sq",
    "am",
    "ar",
    "hy",
    "as",
    "ast",
    "az",
    "ba",
    "eu",
    "be",
    "bn",
    "bs",
    "br",
    "bg",
    "my",
    "es",
    "ca",
    "ceb",
    "zh",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fo",
    "fi",
    "fr",
    "fy",
    "ff",
    "gd",
    "gl",
    "lg",
    "ka",
    "de",
    "el",
    "gu",
    "ht",
    "ha",
    "haw",
    "he",
    "hi",
    "hu",
    "is",
    "ig",
    "ilo",
    "id",
    "ga",
    "it",
    "ja",
    "jv",
    "kn",
    "kk",
    "km",
    "ko",
    "lo",
    "la",
    "lv",
    "lb",
    "ln",
    "lt",
    "mk",
    "mg",
    "ms",
    "ml",
    "mt",
    "mi",
    "mr",
    "mo",
    "mn",
    "ne",
    "no",
    "nn",
    "oc",
    "or",
    "pa",
    "ps",
    "fa",
    "pl",
    "pt",
    "ro",
    "ru",
    "sa",
    "sr",
    "sn",
    "sd",
    "si",
    "sk",
    "sl",
    "so",
    "su",
    "sw",
    "ss",
    "sv",
    "tl",
    "tg",
    "ta",
    "tt",
    "te",
    "th",
    "bo",
    "tn",
    "tr",
    "tk",
    "uk",
    "ur",
    "uz",
    "vi",
    "cy",
    "wo",
    "xh",
    "yi",
    "yo",
    "zu",
]


class TranscriptionWord(BaseModel):
    model_config = ConfigDict(strict=True)

    word: str = ""
    start: float = 0.0
    end: float = 0.0


class TranscriptionUtterance(BaseModel):
    model_config = ConfigDict(strict=True)

    text: str = ""
    language: str | None = None
    start: float = 0.0
    end: float = 0.0
    confidence: float = 1.0
    words: list[TranscriptionWord] = Field(default_factory=list[TranscriptionWord])


class TranscriptMessage(BaseModel):
    model_config = ConfigDict(strict=True)

    is_final: bool
    utterance: TranscriptionUtterance


class TranslationMessage(BaseModel):
    model_config = ConfigDict(strict=True)

    translated_utterance: TranscriptionUtterance | None = None
    utterance: TranscriptionUtterance = Field(default_factory=TranscriptionUtterance)
    target_language: str = ""


class StreamMessage(BaseModel):
    model_config = ConfigDict(strict=True)

    type: str
    data: object = None


class LiveSession(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    url: str
