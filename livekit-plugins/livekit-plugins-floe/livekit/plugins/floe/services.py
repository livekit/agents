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

import os
from urllib.parse import urlparse

import httpx
import openai

from livekit.agents.llm import ToolChoice
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins.openai import LLM as OpenAILLM

from .models import LLMModels

# Keyless Floe gateway: Floe holds the upstream provider keys and bills your
# Floe balance. Authenticated with a Floe API key only.
FLOE_GATEWAY_URL = "https://credit-api.floelabs.xyz/v1"

# BYOK metered proxy: you bring your own upstream provider key (forwarded via
# the ``X-Floe-Provider-Key`` header); Floe meters spend against your budget.
FLOE_BYOK_PROXY_URL = "https://credit-api.floelabs.xyz/v1/llm"


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str | LLMModels = "openai/gpt-4o",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        provider_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        max_retries: NotGivenOr[int] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
    ) -> None:
        """Create a new instance of the Floe LLM.

        The Floe LLM is an OpenAI-compatible client that routes LiveKit's LLM
        traffic through Floe so spend is metered and guarded against a budget.
        Two modes are supported:

        Keyless gateway (default):
            Floe holds the upstream provider keys and bills your Floe balance.
            Only a Floe API key is required and requests go to
            ``FLOE_GATEWAY_URL``.

        Bring your own key (BYOK):
            You supply an upstream provider key. Floe forwards it via the
            ``X-Floe-Provider-Key`` header and meters spend against your budget.
            Requests default to the metered proxy at ``FLOE_BYOK_PROXY_URL``.

        Args:
            model: Model id to route, e.g. ``"openai/gpt-4o"``.
            api_key: Your Floe API key. Falls back to the ``FLOE_API_KEY``
                environment variable. Raises ``ValueError`` if neither is set.
            provider_key: Optional upstream provider key for BYOK mode. Falls
                back to the ``FLOE_PROVIDER_KEY`` environment variable. When
                present, BYOK mode is used.
            base_url: Override the resolved Floe base URL — e.g. to point at your
                own Floe instance, including a self-hosted deployment on a custom
                domain. Honored in both keyless and BYOK modes; in BYOK mode the
                ``X-Floe-Provider-Key`` header is sent to this base URL by design,
                so it must be ``https`` (loopback ``http`` is allowed for local
                dev) — the provider key is never sent over a cleartext connection.
            temperature: Sampling temperature forwarded to the model.
            top_p: Nucleus sampling probability forwarded to the model.
            parallel_tool_calls: Whether the model may call tools in parallel.
            tool_choice: Tool selection strategy.
            timeout: Per-request HTTP timeout.
            max_retries: Maximum number of request retries.
            client: A pre-configured ``openai.AsyncClient``. When provided it is
                used as-is and the resolved base URL/header are not applied.
        """
        floe_key = _get_api_key(api_key)

        resolved_provider_key = (
            provider_key if is_given(provider_key) else os.environ.get("FLOE_PROVIDER_KEY")
        )

        extra_headers: NotGivenOr[dict[str, str]] = NOT_GIVEN
        if resolved_provider_key:
            resolved_base_url = base_url if is_given(base_url) else FLOE_BYOK_PROXY_URL
            _require_secure_byok_url(resolved_base_url)
            extra_headers = {"X-Floe-Provider-Key": resolved_provider_key}
        else:
            resolved_base_url = base_url if is_given(base_url) else FLOE_GATEWAY_URL

        super().__init__(
            model=model,
            api_key=floe_key,
            base_url=resolved_base_url,
            client=client,
            extra_headers=extra_headers,
            temperature=temperature,
            top_p=top_p,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            timeout=timeout,
            max_retries=max_retries,
        )


def _require_secure_byok_url(url: str) -> None:
    """BYOK forwards a provider secret as a header — refuse to send it over cleartext.

    Allows https to any host (incl. self-hosted Floe on a custom domain); allows
    http only for loopback (local dev). Fail closed on anything else.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and host in {"localhost", "127.0.0.1", "::1"}:
        return
    raise ValueError(
        f"BYOK (X-Floe-Provider-Key) requires an https base_url; got {url!r}. "
        "Refusing to send your provider key over a non-TLS connection."
    )


def _get_api_key(key: NotGivenOr[str]) -> str:
    floe_api_key = key if is_given(key) else os.environ.get("FLOE_API_KEY")
    if not floe_api_key:
        raise ValueError(
            "FLOE_API_KEY is required, either as argument or set FLOE_API_KEY environment variable"  # noqa: E501
        )
    return floe_api_key
