# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local speech-to-text for LiveKit Agents, powered by sherpa-onnx.

`STT` streams with an online recognizer (zipformer and other transducer, paraformer and CTC
models). `BatchSTT` transcribes whole utterances with an offline recognizer (Whisper,
Moonshine, SenseVoice, Parakeet and others). Both run on ONNX Runtime, in process.
"""

import os

from livekit.agents import Plugin

from .batch_stt import BatchSTT
from .log import logger
from .models import (
    BATCH_STT_MODELS,
    STT_MODELS,
    BatchSTTModels,
    HubModel,
    STTModels,
    download_model,
)
from .stt import STT, SpeechStream
from .version import __version__

__all__ = ["STT", "BatchSTT", "BatchSTTModels", "STTModels", "SpeechStream", "__version__"]

MODELS_ENV_VAR = "LIVEKIT_SHERPA_MODELS"
DEFAULT_MODELS = ("zipformer-en-2023-06-26",)


def models_to_download() -> list[HubModel]:
    """Built-in models fetched by `download-files`: the default streaming model, or the
    comma separated names in `LIVEKIT_SHERPA_MODELS` (e.g. "moonshine-tiny-en-int8")."""
    registry: dict[str, HubModel] = {
        name: spec for table in (STT_MODELS, BATCH_STT_MODELS) for name, spec in table.items()
    }
    names = [n.strip() for n in os.environ.get(MODELS_ENV_VAR, "").split(",") if n.strip()]
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"unknown {MODELS_ENV_VAR} entries {unknown}; known: {list(registry)}")
    return [registry[n] for n in names or DEFAULT_MODELS]


class SherpaPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        for model in models_to_download():
            download_model(model)


Plugin.register_plugin(SherpaPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
