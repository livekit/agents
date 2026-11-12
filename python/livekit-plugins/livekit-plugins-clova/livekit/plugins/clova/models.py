from typing import Literal

ClovaSttLanguages = Literal["ko-KR", "en-US", "enko", "ja", "zh-CN", "zh-TW"]

ClovaSpeechAPIType = Literal["recognizer/object-storage", "recognizer/url", "recognizer/upload"]

clova_languages_mapping = {
    "en": "en-US",
    "zh-CN": "zh-cn",
    "zh-TW": "zh-tw",
}
