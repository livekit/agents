# Copyright 2025 LiveKit, Inc.
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


from __future__ import annotations

import lk_blingfire as _cext

from .version import __version__


def text_to_sentences(text: str) -> str:
    return _cext.text_to_sentences(text)


def text_to_sentences_with_offsets(
    text: str,
) -> tuple[str, list[tuple[int, int]]]:
    return _cext.text_to_sentences_with_offsets(text)


def text_to_words(text: str) -> str:
    return _cext.text_to_words(text)


def text_to_words_with_offsets(
    text: str,
) -> tuple[str, list[tuple[int, int]]]:
    return _cext.text_to_words_with_offsets(text)


__all__ = [
    "text_to_sentences",
    "text_to_sentences_with_offsets",
    "text_to_words",
    "text_to_words_with_offsets",
    "__version__",
]
