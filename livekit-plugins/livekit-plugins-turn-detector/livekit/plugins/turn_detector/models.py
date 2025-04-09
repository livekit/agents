from typing import Literal

EOUModelType = Literal["en", "multilingual"]
MODEL_REVISIONS: dict[EOUModelType, str] = {
    "en": "v1.2.2-en",
    "multilingual": "v0.1.0-intl",
}
HG_MODEL = "livekit/turn-detector"
ONNX_FILENAME = "model_q8.onnx"
