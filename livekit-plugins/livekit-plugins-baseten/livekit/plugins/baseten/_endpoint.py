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

"""Shared endpoint resolution for the Qwen3 WebSocket models."""

from __future__ import annotations

import os

_TRUSS_URL_TEMPLATE = "wss://model-{model_id}.api.baseten.co/environments/production/websocket"
_CHAIN_URL_TEMPLATE = "wss://chain-{chain_id}.api.baseten.co/environments/production/websocket"


def resolve_endpoint(
    model_endpoint: str | None, model_id: str | None, chain_id: str | None, env_var: str
) -> str:
    """Resolve a WebSocket endpoint, mirroring `STT`'s precedence rules."""
    endpoint = model_endpoint
    if not endpoint and model_id:
        endpoint = _TRUSS_URL_TEMPLATE.format(model_id=model_id)
    if not endpoint and chain_id:
        endpoint = _CHAIN_URL_TEMPLATE.format(chain_id=chain_id)
    endpoint = endpoint or os.environ.get(env_var)
    if not endpoint:
        raise ValueError(
            f"An endpoint is required: pass model_endpoint, model_id, or chain_id, "
            f"or set {env_var}."
        )
    if not endpoint.startswith(("wss://", "ws://")):
        raise ValueError(
            f"This model is served over WebSocket only; got {endpoint!r}. Endpoints "
            "look like wss://model-<id>.api.baseten.co/environments/production/websocket"
        )
    return endpoint
