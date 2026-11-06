from __future__ import annotations

from typing import Literal

#: StepFun's two regional deployments: "global" (api.stepfun.ai) and "china"
#: (api.stepfun.com, mainland China). Credentials are cluster-specific.
Cluster = Literal["global", "china"]

DEFAULT_CLUSTER: Cluster = "global"

CLUSTER_WS_BASE_URLS: dict[Cluster, str] = {
    "global": "wss://api.stepfun.ai",
    "china": "wss://api.stepfun.com",
}

CLUSTER_HTTP_BASE_URLS: dict[Cluster, str] = {
    "global": "https://api.stepfun.ai",
    "china": "https://api.stepfun.com",
}


Language = Literal["zh", "en", "ja", "ko", "fr", "es"]


#: Streaming ASR, over the bidirectional WebSocket.
STTModels = Literal[
    "stepaudio-2.5-asr-stream",
    # Mainland-China cluster only, and documented as being phased out. The
    # global endpoint accepts stepaudio-2.5-asr-stream exclusively.
    "step-asr-1.1-stream",
]

INCREMENTAL_DELTA_STT_MODELS = frozenset({"step-asr-1.1-stream"})

TIMESTAMP_ALIGN_STT_MODELS = frozenset({"stepaudio-2.5-asr-stream"})

#: Batch ASR, over HTTP + SSE. Used by `STT._recognize_impl` and by
#: StreamAdapter-based setups.
STTBatchModels = Literal[
    "stepaudio-3-asr-max",
    "stepaudio-2.5-asr",
    # Mainland-China cluster only.
    "stepaudio-2-asr-pro",
]


TTSModels = Literal[
    "stepaudio-3-tts",
    "stepaudio-2.5-tts",
    # Mainland-China cluster only.
    "step-tts-2",
    "step-tts-mini",
    # Deprecated by the provider; existing callers keep working.
    "step-tts-vivid",
]

#: Per-model cap on `instruction` length, in characters.
INSTRUCTION_MAX_CHARS: dict[str, int] = {
    "stepaudio-2.5-tts": 200,
    "stepaudio-3-tts": 500,
}

#: Maximum characters per `tts.text.delta` frame, and per one-shot HTTP request.
TTS_MAX_INPUT_CHARS = 1000


TTSVoices = Literal[
    # -- Chinese, male
    "cixingnansheng",
    "boyinnansheng",
    "zhengpaiqingnian",
    "zixinnansheng",
    "yuanqinansheng",
    "ruyananshi",
    "shuangkuainansheng",
    "qingniandaxuesheng",
    "shenchennanyin",
    "wenrougongzi",
    "wenrounansheng",
    # -- Chinese, female
    "elegantgentle-female",
    "linjiameimei",
    "linjiajiejie",
    "ruanmengnvsheng",
    "shuangkuaijiejie",
    "wenjingxuejie",
    "qingchunshaonv",
    "livelybreezy-female",
    "ganliannvsheng",
    "tianmeinvsheng",
    "youyanvsheng",
    "jingdiannvsheng",
    "yuanqishaonv",
    "zhixingjiejie",
    "qinqienvsheng",
    "wenroushunv",
    "qinhenvsheng",
    "huolinvsheng",
    "wenrounvsheng",
    "jilingshaonv",
    "lengyanyujie",
    # -- English
    "vibrant-youth",
    "soft-spoken-gentleman",
    "magnetic-voiced-male",
    "lively-girl",
    "energeticconfident-female",
    "yingwennansheng",
    "yingwennvsheng",
]

#: Voices published on the global cluster, i.e. available outside mainland China.
GLOBAL_TTS_VOICES = frozenset(
    {
        "elegantgentle-female",
        "lively-girl",
        "livelybreezy-female",
        "magnetic-voiced-male",
        "soft-spoken-gentleman",
        "vibrant-youth",
        "zixinnansheng",
    }
)

#: English-language voice, available on both clusters. English is also what the
#: service falls back to when no language is given, so the two defaults agree.
DEFAULT_TTS_VOICE = "vibrant-youth"

SYSTEM_VOICES_MODEL: dict[Cluster, str] = {
    "global": "stepaudio-2.5-tts",
    "china": "step-tts-2",
}


#: Sample rate sent to the ASR endpoints. The base RecognizeStream resamples
#: the 48kHz WebRTC audio down to this for us.
DEFAULT_STT_SAMPLE_RATE = 16000

DEFAULT_TTS_SAMPLE_RATE = 24000

TTS_SAMPLE_RATES = frozenset({8000, 16000, 22050, 24000, 48000})

NUM_CHANNELS = 1

STT_FRAME_MS = 50

#: Raw PCM16 little-endian, the codec string the ASR `format` block expects.
PCM_CODEC = "pcm_s16le"

PCM_BITS = 16


DEFAULT_VAD_SILENCE_DURATION_MS = 800

#: Higher is stricter.
DEFAULT_VAD_THRESHOLD = 0.6


#: The streaming TTS connection is dropped after roughly 60s of inactivity, so
#: pooled connections are retired well before that.
TTS_MAX_SESSION_DURATION = 30.0

STT_MAX_SESSION_DURATION = 15.0


STT_STREAM_PATH = "/v1/realtime/asr/stream"

STT_RECOGNIZE_PATH = "/v1/audio/asr/sse"

#: Streaming TTS. Takes the model as a query parameter, unlike ASR.
TTS_STREAM_PATH = "/v1/realtime/audio"

#: One-shot TTS over HTTP.
TTS_SYNTHESIZE_PATH = "/v1/audio/speech"

#: Official voice catalogue; see SYSTEM_VOICES_MODEL for the required `model`.
SYSTEM_VOICES_PATH = "/v1/audio/system_voices"

#: Suffix appended to the HTTP base URL for the OpenAI-compatible client, which
#: expects the API version in the base URL rather than in each request path.
OPENAI_COMPAT_SUFFIX = "/v1"


LLMModels = Literal[
    # Flagship reasoning model. Reasons by default and takes a documented
    # effort level; there is no setting that turns reasoning off.
    "step-5-preview",
    # Reasoning model with three documented effort levels. The provider's own
    # voice-candidate suggestion, and the default here.
    "step-3.7-flash",
    # Text only, 256K context, sparse MoE. No documented effort levels.
    "step-3.5-flash",
]

DEFAULT_LLM_MODEL = "step-3.7-flash"

ReasoningEffort = Literal["low", "medium", "high"]

#: Models whose API documents `reasoning_effort`. Sending it elsewhere is
#: accepted but may be ignored, so the plugin warns instead of guessing.
REASONING_EFFORT_LLM_MODELS = frozenset({"step-5-preview", "step-3.7-flash"})
