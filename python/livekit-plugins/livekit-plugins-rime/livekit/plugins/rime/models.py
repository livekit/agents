from typing import Literal

TTSModels = Literal["mistv2", "mistv3", "coda"]

MODEL_CODA = "coda"
MODEL_MIST_V2 = "mistv2"
MODEL_MIST_V3 = "mistv3"

DefaultMistVoice = "cove"
DefaultCodaVoice = "lyra"

# the WebSocket v1 endpoint of each model that streams; mistv2 synthesizes over HTTP only
WEBSOCKET_URLS: dict[str, str] = {
    MODEL_CODA: "wss://coda.api.rime.ai/ws",
    MODEL_MIST_V3: "wss://mist.api.rime.ai/ws",
}


def is_mist_model(model: str) -> bool:
    return model.startswith("mist")


def supports_time_scale_factor(model: str) -> bool:
    return model != MODEL_MIST_V2
