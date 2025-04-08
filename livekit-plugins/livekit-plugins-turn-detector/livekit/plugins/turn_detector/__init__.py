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

from .log import logger
from .version import __version__

__all__ = ["english", "multilingual", "__version__"]


class EOUPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        from transformers import AutoTokenizer

        from .base import _download_from_hf_hub
        from .models import HG_MODEL, MODEL_REVISIONS, ONNX_FILENAME

        for revision in MODEL_REVISIONS.values():
            AutoTokenizer.from_pretrained(HG_MODEL, revision=revision)
            _download_from_hf_hub(HG_MODEL, ONNX_FILENAME, subfolder="onnx", revision=revision)
            _download_from_hf_hub(HG_MODEL, "languages.json", revision=revision)


Plugin.register_plugin(EOUPlugin())
