from typing import Literal

from openai.types import AudioModel

STTModels = AudioModel
TTSModels = Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
TTSVoices = Literal[
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
]
DalleModels = Literal["dall-e-2", "dall-e-3"]
ChatModels = Literal[
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4-1106-vision-preview",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k-0613",
]
EmbeddingModels = Literal[
    "text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"
]

AssistantTools = Literal["code_interpreter", "file_search", "function"]

# adapters for OpenAI-compatible LLMs

TelnyxChatModels = Literal[
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
]

CerebrasChatModels = Literal[
    "llama3.1-8b",
    "llama-3.3-70b",
    "llama-4-scout-17b-16e-instruct",
    "qwen-3-32b",
    "deepseek-r1-distill-llama-70b",
]

PerplexityChatModels = Literal[
    "llama-3.1-sonar-small-128k-online",
    "llama-3.1-sonar-small-128k-chat",
    "llama-3.1-sonar-large-128k-online",
    "llama-3.1-sonar-large-128k-chat",
    "llama-3.1-8b-instruct",
    "llama-3.1-70b-instruct",
]

GroqChatModels = Literal[
    "llama-3.1-405b-reasoning",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
    "gemma2-9b-it",
]

GroqAudioModels = Literal[
    "whisper-large-v3", "distil-whisper-large-v3-en", "whisper-large-v3-turbo"
]

DeepSeekChatModels = Literal[
    "deepseek-coder",
    "deepseek-chat",
]

VertexModels = Literal[
    "google/gemini-2.0-flash-exp",
    "google/gemini-1.5-flash",
    "google/gemini-1.5-pro",
    "google/gemini-1.0-pro-vision",
    "google/gemini-1.0-pro-vision-001",
    "google/gemini-1.0-pro-002",
    "google/gemini-1.0-pro-001",
    "google/gemini-1.0-pro",
]

TogetherChatModels = Literal[
    "Austism/chronos-hermes-13b",
    "Gryphe/MythoMax-L2-13b",
    "NousResearch/Nous-Capybara-7B-V1p9",
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
    "NousResearch/Nous-Hermes-2-Yi-34B",
    "NousResearch/Nous-Hermes-Llama2-13b",
    "NousResearch/Nous-Hermes-llama-2-7b",
    "Open-Orca/Mistral-7B-OpenOrca",
    "Qwen/Qwen1.5-0.5B-Chat",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-110B-Chat",
    "Qwen/Qwen1.5-14B-Chat",
    "Qwen/Qwen1.5-32B-Chat",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen1.5-72B-Chat",
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen2-72B-Instruct",
    "Snowflake/snowflake-arctic-instruct",
    "Undi95/ReMM-SLERP-L2-13B",
    "Undi95/Toppy-M-7B",
    "WizardLM/WizardLM-13B-V1.2",
    "allenai/OLMo-7B",
    "allenai/OLMo-7B-Instruct",
    "allenai/OLMo-7B-Twin-2T",
    "codellama/CodeLlama-13b-Instruct-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
    "codellama/CodeLlama-70b-Instruct-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
    "cognitivecomputations/dolphin-2.5-mixtral-8x7b",
    "databricks/dbrx-instruct",
    "deepseek-ai/deepseek-coder-33b-instruct",
    "deepseek-ai/deepseek-llm-67b-chat",
    "garage-bAInd/Platypus2-70B-instruct",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2b-it",
    "google/gemma-7b-it",
    "lmsys/vicuna-13b-v1.5",
    "lmsys/vicuna-7b-v1.5",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-3-70b-chat-hf",
    "meta-llama/Llama-3-8b-chat-hf",
    "meta-llama/Meta-Llama-3-70B-Instruct-Lite",
    "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "openchat/openchat-3.5-1210",
    "snorkelai/Snorkel-Mistral-PairRM-DPO",
    "teknium/OpenHermes-2-Mistral-7B",
    "teknium/OpenHermes-2p5-Mistral-7B",
    "togethercomputer/Llama-2-7B-32K-Instruct",
    "togethercomputer/RedPajama-INCITE-7B-Chat",
    "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
    "togethercomputer/StripedHyena-Nous-7B",
    "togethercomputer/alpaca-7b",
    "upstage/SOLAR-10.7B-Instruct-v1.0",
    "zero-one-ai/Yi-34B-Chat",
]

OctoChatModels = Literal[
    "meta-llama-3-70b-instruct",
    "meta-llama-3.1-405b-instruct",
    "meta-llama-3.1-70b-instruct",
    "meta-llama-3.1-8b-instruct",
    "mistral-7b-instruct",
    "mixtral-8x7b-instruct",
    "wizardlm-2-8x22bllamaguard-2-7b",
]


XAIChatModels = Literal[
    "grok-3",
    "grok-3-fast",
    "grok-3-mini",
    "grok-3-mini-fast",
    "grok-2-vision-1212",
    "grok-2-image-1212",
    "grok-2-1212",
]
