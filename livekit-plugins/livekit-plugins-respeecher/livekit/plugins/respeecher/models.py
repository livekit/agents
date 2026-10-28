from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue, TypeAdapter

TTSModels = Literal[
    # Respeecher's public English model
    "/public/tts/en-rt",
    # Respeecher's public Ukrainian model
    "/public/tts/ua-rt",
]

TTSEncoding = Literal["pcm_s16le",]


"""Check https://space.respeecher.com/docs/api/tts/sampling-params-guide for details"""
SamplingParams = dict[str, JsonValue]
_SAMPLING_PARAMS = TypeAdapter(SamplingParams)


@dataclass
class VoiceSettings:
    """Voice settings for Respeecher TTS"""

    sampling_params: SamplingParams | None = None


class Voice(dict[str, object]):
    """Voice model for Respeecher - behaves like a dict with guaranteed `id` and optional `sampling_params`"""

    def __init__(
        self,
        mapping: Mapping[str, object] | Iterable[tuple[str, object]] = (),
        /,
        **kwargs: object,
    ) -> None:
        super().__init__(mapping, **kwargs)
        if "id" not in self:
            raise ValueError("Voice must have an 'id' field")

    @property
    def id(self) -> str:
        return str(self["id"])

    @property
    def sampling_params(self) -> SamplingParams | None:
        value = self.get("sampling_params")
        return _SAMPLING_PARAMS.validate_python(value) if value is not None else None
