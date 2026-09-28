from typing import Literal

TTSModels = Literal["mistv2", "mistv3", "coda"]

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
