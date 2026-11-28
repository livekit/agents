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

"""Public model selection for the original and conversational STT implementations."""

from typing import Any, Literal

from .stt import STT as LegacySTT
from .stt_conv import STTConv

NabrahRecognitionModel = Literal["nabrah_stt_v1", "eot_nabrah", "nabrah_stt_conv"]


def STT(
    *, recognition_model: NabrahRecognitionModel = "nabrah_stt_conv", **options: Any
) -> LegacySTT | STTConv:
    """Select the implementation matching recognition_model (default: nabrah_stt_conv).

    api_key, base_url and http_session are shared options. STTConv fixes metadata to ar-SA.
    Other options are passed to the selected implementation.
    """
    if recognition_model not in ("nabrah_stt_v1", "eot_nabrah", "nabrah_stt_conv"):
        raise ValueError(f"Unsupported Nabrah recognition_model: {recognition_model!r}")
    if recognition_model == "nabrah_stt_conv":
        return STTConv(**options)
    return LegacySTT(recognition_model=recognition_model, **options)
