from dataclasses import dataclass
from typing import Any, Literal, Optional

TTSModels = Literal[
    # Respeecher's English model, multilanguage models will be added later
    "/public/tts/en-rt",
]

TTSEncoding = Literal[
    "pcm_s16le",
    "pcm_f32le",
]


"""Check https://space.respeecher.com/docs/api/tts/sampling-params-guide for details"""
SamplingParams = dict[str, Any]


@dataclass
class VoiceSettings:
    """Voice settings for Respeecher TTS"""

    sampling_params: Optional[SamplingParams] = None


@dataclass
class Voice:
    """Voice model for Respeecher"""

    id: str
    gender: Optional[str] = None
    accent: Optional[str] = None
    age: Optional[str] = None
    sampling_params: Optional[SamplingParams] = None
