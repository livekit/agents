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


class Voice(dict):
    """Voice model for Respeecher - behaves like a dict with guaranteed `id` and optional `sampling_params`"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "id" not in self:
            raise ValueError("Voice must have an 'id' field")

    @property
    def id(self) -> str:
        return self["id"]

    @property
    def sampling_params(self) -> Optional[SamplingParams]:
        return self.get("sampling_params")
