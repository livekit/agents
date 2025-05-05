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

"""NLTK plugin for LiveKit Agents

Support for [NLTK](https://www.nltk.org/)-based text processing.
Currently featuring a `SentenceTokenizer`.
"""

from .sentence_tokenizer import SentenceTokenizer
from .version import __version__

__all__ = ["SentenceTokenizer", "__version__"]


import nltk  # type: ignore
from livekit.agents import Plugin

from .log import logger


class NltkPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self):
        try:
            _ = nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab")


Plugin.register_plugin(NltkPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
