from typing import Literal

TTSEncodings = Literal[
    "pcm_linear",
    "pcm_mulaw",
]

TTSModels = Literal[
    "neu-fast", 
    "neu-hq"
]

TTSLangCodes = Literal[
    "en", "nl", "es", "de", "hi", "en-hi", "ar"
]
