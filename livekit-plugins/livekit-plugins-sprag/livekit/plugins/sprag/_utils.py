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

from __future__ import annotations

import os

import openai

from livekit.agents.types import NotGivenOr
from livekit.agents.utils import is_given

from .version import __version__

SPRAG_BASE_URL = "https://api.sprag.ai/v1"
ATTRIBUTION_HEADERS = {"X-Sprag-Integration": f"livekit-agents/{__version__}"}


def resolve_api_key(api_key: NotGivenOr[str]) -> str:
    api_key = api_key if is_given(api_key) else os.environ.get("SPRAG_API_KEY", "")
    if not api_key:
        raise ValueError(
            "SPRAG_API_KEY is required, either as argument or set "
            "SPRAG_API_KEY environmental variable"
        )
    return api_key


def with_attribution(client: openai.AsyncClient) -> openai.AsyncClient:
    return client.with_options(default_headers=ATTRIBUTION_HEADERS)
