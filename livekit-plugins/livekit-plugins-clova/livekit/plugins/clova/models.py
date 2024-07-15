from typing import Literal

ClovaSttLanguages = Literal[
    "zh",
    "zh-CN",
    "zh-TW",
    "ko",
    "ja",
    "en",
]

ClovaSpeechAPIType = Literal[
    "recognizer/object-storage", "recognizer/url", "recognizer/upload"
]
