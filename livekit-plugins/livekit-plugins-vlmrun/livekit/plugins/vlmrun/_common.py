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

from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .version import __version__

# The gateway can hold a reply until the first token, and allows qwen/qwen3.8-27b up to 60 s for
# it (https://docs.vlm.run/gateway/error-codes). The framework's 10 s default cut those requests
# off and retried them, so wait past the gateway's own deadline and let its 504 arrive first.
DEFAULT_CONN_OPTIONS = APIConnectOptions(timeout=70.0)


def conn_options_or_default(conn_options: APIConnectOptions) -> APIConnectOptions:
    # A caller that leaves the timeout alone, AgentSession included, passes the framework's
    # default object. Identity, not equality, so an explicit APIConnectOptions() is kept.
    return DEFAULT_CONN_OPTIONS if conn_options is DEFAULT_API_CONNECT_OPTIONS else conn_options


DEFAULT_GATEWAY_URL = "https://gateway.vlm.run"


def gateway_url(base_url: str | None = None) -> str:
    url = (
        base_url
        or os.environ.get("VLMRUN_GATEWAY_BASE_URL")
        # The name VLM Run's SDK and CLI docs use (https://docs.vlm.run/cli/gateway).
        or os.environ.get("VLMRUN_GATEWAY_URL")
        or DEFAULT_GATEWAY_URL
    )
    # VLM Run's docs give the gateway URL with the /v1 suffix, and an openai SDK user has it
    # with /openai; accept the root or either form.
    return url.rstrip("/").removesuffix("/v1").removesuffix("/openai")


def openai_base_url(base_url: str | None = None) -> str:
    return f"{gateway_url(base_url)}/openai"


def resolve_api_key(api_key: str | None = None) -> str | None:
    return api_key or os.environ.get("VLMRUN_API_KEY") or None


CLIENT_HEADER = "X-VLMRun-Client"


def client_headers(api_key: str | None) -> dict[str, str]:
    headers = {CLIENT_HEADER: f"livekit-agents/{__version__}"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers
