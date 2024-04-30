from typing import Literal

WhisperModels = Literal["whisper-1"]
TTSModels = Literal["tts-1", "tts-1-hd"]
TTSVoices = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
DalleModels = Literal["dall-e-2", "dall-e-3"]
ChatModels = Literal[
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
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
