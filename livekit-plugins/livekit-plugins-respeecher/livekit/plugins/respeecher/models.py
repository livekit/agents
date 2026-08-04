from dataclasses import dataclass
from typing import Any, Literal

TTSModels = Literal[
    # Respeecher's public English model
    "/public/tts/en-rt",
    # Respeecher's public Ukrainian model
    "/public/tts/ua-rt",
]

TTSEncoding = Literal["pcm_s16le",]


"""Check https://space.respeecher.com/docs/api/tts/sampling-params-guide for details"""
SamplingParams = dict[str, Any]


@dataclass
class VoiceSettings:
    """Voice settings for Respeecher TTS"""

    sampling_params: SamplingParams | None = None


class Voice(dict):
    """Voice model for Respeecher - behaves like a dict with guaranteed `id` and optional `sampling_params`"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if "id" not in self:
            raise ValueError("Voice must have an 'id' field")

    @property
    def id(self) -> str:
        return str(self["id"])

    @property
    def sampling_params(self) -> SamplingParams | None:
        return self.get("sampling_params")
