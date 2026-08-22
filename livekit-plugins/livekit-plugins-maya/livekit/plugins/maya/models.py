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

from __future__ import annotations

from typing import Literal

TTSModels = Literal["Maya 2 Native", "Maya 2 Native Emotional"]
"""Models available on the websocket. ``Maya 2 Global`` is HTTP-only."""

TTSLanguages = Literal["hi", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te", "en"]
"""Ten Indian languages plus ``en``, which is Indian English rather than a
British or American variant. Omit the language for text that switches
languages mid-sentence, so each part follows its own script's rules."""
