from dataclasses import dataclass
from typing import Literal, Optional

TTSModels = Literal[
    # Respeecher's English model, multilanguage models will be added later
    "/v1/public/tts/en-rt",
]

TTSEncoding = Literal[
    "pcm_s16le",
    "pcm_f32le",
]

TTSLanguages = Literal["en"]


@dataclass
class SamplingParams:
    """Check https://space.respeecher.com/docs/api/tts/sampling-params-guide for details"""

    seed: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None


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
