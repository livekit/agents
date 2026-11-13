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

"""Hugging Face plugin for LiveKit Agents: run open chat models locally with transformers."""

import os

from livekit.agents import Plugin

from .download import download_model, models_to_download
from .llm import DEFAULT_MODEL, LLM, LLMStream
from .log import logger
from .version import __version__

__all__ = ["LLM", "LLMStream", "DEFAULT_MODEL", "download_model", "__version__"]


class HFPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        for model, revision in models_to_download(DEFAULT_MODEL):
            if os.path.isdir(model):
                continue
            download_model(model, revision=revision)


Plugin.register_plugin(HFPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
