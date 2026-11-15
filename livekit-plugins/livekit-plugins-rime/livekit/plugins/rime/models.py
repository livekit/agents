from typing import Literal, TypedDict

TTSModels = Literal["mistv2", "mistv3", "coda"]


class CustomLexiconEntry(TypedDict):
    """A Coda pronunciation override, matched as a whole word or phrase, ignoring case.

    See https://docs.rime.ai/platform/phoneme-control for the format.
    """

    spelling: str
    """The word or phrase as Coda reads it, e.g. ``Doctor`` rather than ``Dr.``."""
    pronunciation: str
    """X-SAMPA phonemes separated by spaces, e.g. ``h E . " l oU``."""


MODEL_CODA = "coda"
MODEL_MIST_V2 = "mistv2"
MODEL_MIST_V3 = "mistv3"

DefaultMistVoice = "cove"
DefaultCodaVoice = "lyra"


def is_mist_model(model: str) -> bool:
    return model.startswith("mist")


def supports_time_scale_factor(model: str) -> bool:
    return model != MODEL_MIST_V2


def supports_reduce_latency(model: str) -> bool:
    return model == MODEL_MIST_V2
