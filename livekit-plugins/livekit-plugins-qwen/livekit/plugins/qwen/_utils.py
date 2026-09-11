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

"""Credential and endpoint resolution shared by the STT, TTS and LLM classes."""

from __future__ import annotations

import os

from .models import COMPAT_BASE_URLS, REALTIME_BASE_URLS, QwenRegion

API_KEY_ENV = "DASHSCOPE_API_KEY"


def resolve_api_key(api_key: str | None) -> str:
    """The explicit key, else ``DASHSCOPE_API_KEY``.

    Raises:
        ValueError: If neither is set.
    """
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        raise ValueError(
            f"Qwen API key is required: pass `api_key` or set the {API_KEY_ENV} environment variable"
        )
    return key


def resolve_realtime_url(base_url: str | None, region: QwenRegion) -> str:
    """An explicit realtime WebSocket URL, else the public endpoint for ``region``."""
    return base_url or REALTIME_BASE_URLS[region]


def resolve_compat_url(base_url: str | None, region: QwenRegion) -> str:
    """An explicit OpenAI-compatible base URL, else the public endpoint for ``region``."""
    return base_url or COMPAT_BASE_URLS[region]
