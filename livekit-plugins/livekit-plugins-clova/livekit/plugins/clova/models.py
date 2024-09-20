from typing import Literal

ClovaSttLanguages = Literal["ko-KR", "en-US", "enko", "ja", "zh-cn", "zh-tw"]

ClovaSpeechAPIType = Literal[
    "recognizer/object-storage", "recognizer/url", "recognizer/upload"
]

clova_languages_mapping = {
    "en": "en-US",
    "ko-KR": "ko-KR",
    "en-US": "en-US",
    "enko": "enko",
    "ja": "ja",
    "zh-cn": "zh-cn",
    "zh-tw": "zh-tw",
}
