from typing import Literal

TTSModels = Literal[
    # cartesia
    "cartesia",
    "cartesia/sonic",
    "cartesia/sonic-2",
    "cartesia/sonic-turbo",
    # elevenlabs
    "elevenlabs",
    "elevenlabs/eleven_flash_v2",
    "elevenlabs/eleven_flash_v2_5",
    "elevenlabs/eleven_turbo_v2",
    "elevenlabs/eleven_turbo_v2_5",
    "elevenlabs/eleven_multilingual_v2",
    "elevenlabs/eleven_multilingual_v3",
    # rime
    "rime",
    "rime/mist",
    "rime/mistv2",
    "rime/arcana",
    # inword
    "inworld",
    "inworld/inworld-tts-1",
]

STTModels = Literal[
    # deepgram
    "deepgram",
    "deepgram/nova-3",
    "deepgram/nova-3-general",
    "deepgram/nova-3-medical",
    "deepgram/nova-2",
    "deepgram/nova-2-general",
    "deepgram/nova-2-medical",
    "deepgram/nova-2-phonecall",
    # cartesia
    "cartesia",
    "cartesia/ink-whisper",
    "assemblyai",
    # "assemblyai/universal-streaming",  # wrong name?
]

STTLanguages = Literal["en", "de", "es", "fr", "ja", "pt", "zh"]


LLMModels = Literal[
    # openai
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-4o",
    "openai/gpt-4o-2024-05-13",
    "openai/gpt-4o-2024-07-18",
    "openai/gpt-4o-mini",
    # google gemini
    "google/gemini-2.5-pro-preview-05-06",
    "google/gemini-2.5-flash-preview-04-17",
    "google/gemini-2.5-flash-preview-05-20",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.0-flash-lite-preview-02-05",
    # cerebras
    "cerebras/llama3.1-8b",
    "cerebras/llama-3.3-70b",
    "cerebras/llama-4-scout-17b-16e-instruct",
    "cerebras/qwen-3-32b",
    "cerebras/qwen-3-235b-a22b-instruct-2507",
    "cerebras/gpt-oss-120b",
    # groq
    "groq/llama3-8b-8192",
    "groq/llama3-70b-8192",
    "groq/llama-3.3-70b-versatile",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "groq/meta-llama/llama-4-maverick-17b-128e-instruct",
    "groq/openai/gpt-oss-120b",
    "groq/moonshotai/kimi-k2-instruct",
    "groq/qwen/qwen3-32b",
    # baseten
    "baseten/deepseek-ai/DeepSeek-V3-0324",
    "baseten/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "baseten/meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "baseten/moonshotai/Kimi-K2-Instruct",
    "baseten/openai/gpt-oss-120b",
    "baseten/Qwen/Qwen3-235B-A22B-Instruct-2507",
]
