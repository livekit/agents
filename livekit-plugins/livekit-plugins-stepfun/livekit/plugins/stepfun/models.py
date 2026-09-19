from typing import Literal

StepAudioRealtimeModels = Literal[
    "stepaudio-3-realtime-preview",
    "stepaudio-2.5-realtime",
    "step-1o-audio",
    # Note on excluded/legacy models:
    # - "step-audio-r1.5": Deep reasoning model that outputs function calls as raw plain text
    #   tokens into speech transcripts rather than structured Realtime tool events, causing it to
    #   speak code syntax aloud rather than executing tools.
    # - "step-audio-2" and "step-audio-2-mini": Older architecture models where StepFun's backend
    #   only streams text transcripts (response.audio_transcript.delta) and does not emit audio
    #   delta frames (response.audio.delta), causing speech playout to hang.
    # "step-audio-r1.5",
    # "step-audio-2",
    # "step-audio-2-mini",
]

StepAudioRealtimeVoices = Literal[
    "lively-girl",
    "elegantgentle-female",
    "livelybreezy-female",
    "magnetic-voiced-male",
    "soft-spoken-gentleman",
    "vibrant-youth",
    "zixinnansheng",
    "linjiajiejie",
    "qinqienvsheng",
    "wenrounvsheng",
]

DEFAULT_MODEL = "stepaudio-2.5-realtime"
DEFAULT_VOICE = "vibrant-youth"
DOMESTIC_BASE_URL = "wss://api.stepfun.com/v1/realtime"
OVERSEAS_BASE_URL = "wss://api.stepfun.ai/v1/realtime"
DEFAULT_BASE_URL = OVERSEAS_BASE_URL

# Voice mappings between international (stepfun.ai) and domestic (stepfun.com) clusters
VOICE_MAPPING_TO_DOMESTIC: dict[str, str] = {
    "lively-girl": "yuanqishaonv",
    "elegantgentle-female": "youyanvsheng",
    "livelybreezy-female": "shuangkuaijiejie",
    "magnetic-voiced-male": "cixingnansheng",
    "soft-spoken-gentleman": "ruyananshi",
    "vibrant-youth": "yuanqinansheng",
    "zixinnansheng": "zixinnansheng",
    "linjiajiejie": "linjiajiejie",
    "qinqienvsheng": "qinqienvsheng",
    "wenrounvsheng": "wenrounvsheng",
}

VOICE_MAPPING_TO_OVERSEAS: dict[str, str] = {
    "yuanqishaonv": "lively-girl",
    "youyanvsheng": "elegantgentle-female",
    "shuangkuaijiejie": "livelybreezy-female",
    "cixingnansheng": "magnetic-voiced-male",
    "ruyananshi": "soft-spoken-gentleman",
    "yuanqinansheng": "vibrant-youth",
    "zixinnansheng": "zixinnansheng",
    "jingdiannvsheng": "lively-girl",
    "qingchunshaonv": "lively-girl",
    "linjiajiejie": "elegantgentle-female",
    "qinqienvsheng": "lively-girl",
    "wenrounvsheng": "elegantgentle-female",
}
