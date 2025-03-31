# Copyright 2023 LiveKit, Inc.
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

from livekit.agents import Plugin
from livekit.agents.inference_runner import _InferenceRunner

from .eou import EOUModel, _EUORunner
from .log import logger
from .version import __version__

__all__ = ["EOUModel", "__version__"]


class EOUPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        from transformers import AutoTokenizer

        from .eou import HG_MODEL, MODEL_REVISION, ONNX_FILENAME, _download_from_hf_hub

        AutoTokenizer.from_pretrained(HG_MODEL, revision=MODEL_REVISION)
        _download_from_hf_hub(HG_MODEL, ONNX_FILENAME, subfolder="onnx", revision=MODEL_REVISION)


Plugin.register_plugin(EOUPlugin())
_InferenceRunner.register_runner(_EUORunner)
