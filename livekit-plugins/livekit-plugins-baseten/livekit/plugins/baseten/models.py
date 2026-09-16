from typing import Literal

LLMModels = Literal[
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V3-0324",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "moonshotai/Kimi-K2-Instruct",
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
]

# Which wire protocol a deployment speaks. Baseten hosts several model families
# behind the same WebSocket surface and they are not interchangeable, so the
# plugin picks an implementation from this rather than sniffing the endpoint.
STTModels = Literal[
    "whisper",  # Streaming Whisper truss (raw PCM frames, message_type replies)
    "qwen3-asr",  # Qwen3-ASR Streaming (OpenAI-realtime frames, transcription replies)
]

TTSModels = Literal[
    "orpheus",  # Orpheus truss (HTTP prompt/voice, or WS + __END__ sentinel)
    "qwen3-tts",  # Qwen3-TTS (session.config / input.text / input.done)
]
