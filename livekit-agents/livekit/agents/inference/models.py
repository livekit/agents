from __future__ import annotations

from typing import Literal, TypedDict, Union

TTSModels_Cartesia = Literal[
    "cartesia", "cartesia/sonic", "cartesia/sonic-2", "cartesia/sonic-turbo"
]
TTSModels_ElevenLabs = Literal[
    "elevenlabs",
    "elevenlabs/eleven_flash_v2",
    "elevenlabs/eleven_flash_v2_5",
    "elevenlabs/eleven_turbo_v2",
    "elevenlabs/eleven_turbo_v2_5",
    "elevenlabs/eleven_multilingual_v2",
    "elevenlabs/eleven_multilingual_v3",
]
TTSModels_Rime = Literal["rime", "rime/mist", "rime/mistv2", "rime/arcana"]
TTSModels_Inworld = Literal["inworld", "inworld/inworld-tts-1"]

TTSModels = Union[TTSModels_Cartesia, TTSModels_ElevenLabs, TTSModels_Rime, TTSModels_Inworld]


class TTSOptions_Cartesia(TypedDict, total=False):
    pass


class _ElevenLabsVoiceSettings(TypedDict, total=False):
    stability: float  # [0.0 - 1.0]
    similarity_boost: float  # [0.0 - 1.0]
    style: float  # [0.0 - 1.0]
    speed: float  # [0.8 - 1.2]
    use_speaker_boost: bool


class TTSOptions_ElevenLabs(TypedDict, total=False):
    voice_settings: _ElevenLabsVoiceSettings
    streaming_latency: int
    inactivity_timeout: int
    apply_text_normalization: Literal["auto", "off", "on"]
    enable_ssml_parsing: bool
    chunk_length_schedule: list[int]


class TTSOptions_Rime(TypedDict, total=False):
    # Arcana options
    repetition_penalty: float
    temperature: float
    top_p: float
    max_tokens: int
    # Mistv2 options
    speed_alpha: float
    reduce_latency: bool
    pause_between_brackets: bool
    phonemize_between_brackets: bool


class TTSOptions_Inworld(TypedDict, total=False):
    bit_rate: int
    pitch: float
    speaking_rate: float
    temperature: float


STTModels_Deepgram = Literal[
    "deepgram",
    "deepgram/nova-3",
    "deepgram/nova-3-general",
    "deepgram/nova-3-medical",
    "deepgram/nova-2",
    "deepgram/nova-2-general",
    "deepgram/nova-2-medical",
    "deepgram/nova-2-phonecall",
]
STTModels_Cartesia = Literal["cartesia", "cartesia/ink-whisper"]
STTModels_Assemblyai = Literal[
    "assemblyai",
    # "assemblyai/universal-streaming",  # wrong name?
]

STTModels = Union[STTModels_Deepgram, STTModels_Cartesia, STTModels_Assemblyai]


class STTOptions_Cartesia(TypedDict, total=False):
    pass


class STTOptions_Deepgram(TypedDict, total=False):
    detect_language: bool
    punctuate: bool
    smart_format: bool
    no_delay: bool
    endpointing_ms: int
    enable_diarization: bool
    filler_words: bool
    keywords: list[tuple[str, float]]
    keyterms: list[str]
    profanity_filter: bool
    numerals: bool
    mip_opt_out: bool


class STTOptions_Assemblyai(TypedDict, total=False):
    end_of_turn_confidence_threshold: float
    min_end_of_turn_silence_when_confident: int
    max_turn_silence: int
    format_turns: bool
    buffer_size_seconds: float


STTLanguages = Literal["en", "de", "es", "fr", "ja", "pt", "zh"]


LLMModels_OpenAI = Literal[
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-4o",
    "openai/gpt-4o-2024-05-13",
    "openai/gpt-4o-2024-07-18",
    "openai/gpt-4o-mini",
]
LLMModels_Google = Literal[
    "google/gemini-2.5-pro-preview-05-06",
    "google/gemini-2.5-flash-preview-04-17",
    "google/gemini-2.5-flash-preview-05-20",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.0-flash-lite-preview-02-05",
]
LLMModels_Cerebras = Literal[
    "cerebras/llama3.1-8b",
    "cerebras/llama-3.3-70b",
    "cerebras/llama-4-scout-17b-16e-instruct",
    "cerebras/qwen-3-32b",
    "cerebras/qwen-3-235b-a22b-instruct-2507",
    "cerebras/gpt-oss-120b",
]
LLMModels_Groq = Literal[
    "groq/llama3-8b-8192",
    "groq/llama3-70b-8192",
    "groq/llama-3.3-70b-versatile",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "groq/meta-llama/llama-4-maverick-17b-128e-instruct",
    "groq/openai/gpt-oss-120b",
    "groq/moonshotai/kimi-k2-instruct",
    "groq/qwen/qwen3-32b",
]
LLMModels_Baseten = Literal[
    "baseten/deepseek-ai/DeepSeek-V3-0324",
    "baseten/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "baseten/meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "baseten/moonshotai/Kimi-K2-Instruct",
    "baseten/openai/gpt-oss-120b",
    "baseten/Qwen/Qwen3-235B-A22B-Instruct-2507",
]

LLMModels = Union[
    LLMModels_OpenAI, LLMModels_Google, LLMModels_Cerebras, LLMModels_Groq, LLMModels_Baseten
]


class LLMOptions_OpenAI(TypedDict, total=False):
    top_p: float


class LLMOptions_Google(TypedDict, total=False):
    presence_penalty: float
    frequency_penalty: float


class LLMOptions_Cerebras(TypedDict, total=False):
    top_p: float


class LLMOptions_Groq(TypedDict, total=False):
    top_p: float


class LLMOptions_Baseten(TypedDict, total=False):
    top_p: float
