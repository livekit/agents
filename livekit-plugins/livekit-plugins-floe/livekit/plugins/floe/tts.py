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

from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins.openai import TTS as OpenAITTS

from .models import TTSModels, TTSVoices

# Keyless Floe voice gateway: Floe holds the upstream provider keys and bills
# your Floe balance. `/v1/audio/speech` is OpenAI-shaped ({model, input, voice}
# -> binary audio), so this is a base-URL swap of the OpenAI TTS.
FLOE_GATEWAY_URL = "https://credit-api.floelabs.xyz/v1"

# Usage tag stamped on emitted metrics so a mixed session can tell Floe-routed
# speech apart from other providers. Mirrors the LLM leg's provider tag.
FLOE_PROVIDER = "floe"


class TTS(OpenAITTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = "openai/tts-1",
        voice: TTSVoices | str = "alloy",
        speed: float = 1.0,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        provider_key: NotGivenOr[str] = NOT_GIVEN,
        task_id: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Create a new instance of the Floe TTS.

        The Floe TTS is an OpenAI-compatible client that routes LiveKit's
        text-to-speech through Floe's ``/v1/audio/speech`` endpoint so spend is
        metered and guarded against a budget. Two modes are supported:

        Keyless gateway (default):
            Floe holds the upstream provider keys and bills your Floe balance.
            Only a Floe API key is required.

        Bring your own key (BYOK):
            You supply an upstream provider key. Floe forwards it via the
            ``X-Floe-Provider-Key`` header and meters spend against your budget.

        Args:
            model: Fully qualified ``provider/model`` TTS id, e.g.
                ``"openai/tts-1"``.
            voice: Voice id to synthesize with.
            speed: Playback speed forwarded to the model.
            instructions: Optional voice-direction prompt (newer models only).
            api_key: Your Floe API key. Falls back to the ``FLOE_API_KEY``
                environment variable. Raises ``ValueError`` if neither is set.
            provider_key: Optional upstream provider key for BYOK mode. Falls
                back to the ``FLOE_PROVIDER_KEY`` environment variable. When
                present it is sent as the ``X-Floe-Provider-Key`` header.
            task_id: Optional Floe task id sent as ``X-Floe-Task-Id`` so a
                per-task budget can bound one conversation.
            base_url: Override the resolved Floe base URL — e.g. to point at a
                self-hosted Floe on a custom domain. Because every request
                carries the Floe API key (and any BYOK provider key), the
                effective address must be ``https`` (loopback ``http`` is
                allowed for local dev).
        """
        floe_key = _get_api_key(api_key)
        resolved_base_url = base_url if is_given(base_url) else FLOE_GATEWAY_URL
        _require_secure_url(resolved_base_url)

        resolved_provider_key = (
            provider_key if is_given(provider_key) else os.environ.get("FLOE_PROVIDER_KEY")
        )

        # /v1/audio/speech serves both keyless and BYOK off the same base — the
        # provider key rides a header, not a distinct path (unlike the LLM leg).
        floe_headers: dict[str, str] = {}
        if resolved_provider_key:
            floe_headers["X-Floe-Provider-Key"] = resolved_provider_key
        if is_given(task_id):
            floe_headers["X-Floe-Task-Id"] = task_id

        if floe_headers:
            # OpenAI TTS builds its own client and takes no default_headers/
            # extra_headers, so hand it a pre-built client carrying the Floe
            # headers. Mirror the parent's timeouts/limits so behaviour is
            # unchanged apart from the added headers.
            client = openai.AsyncClient(
                max_retries=0,
                api_key=floe_key,
                base_url=resolved_base_url,
                default_headers=floe_headers,
                http_client=httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                    follow_redirects=True,
                    limits=httpx.Limits(
                        max_connections=50, max_keepalive_connections=50, keepalive_expiry=120
                    ),
                ),
            )
            super().__init__(
                model=model,
                voice=voice,
                speed=speed,
                instructions=instructions,
                client=client,
            )
            # The parent marks a passed-in client as not-owned, so its aclose()
            # would never close it — leaking the httpx client + its connection
            # pool. We built this client, so we own its lifecycle (mirrors the
            # parent's own with_azure()).
            self._owns_client = True
        else:
            super().__init__(
                model=model,
                voice=voice,
                speed=speed,
                instructions=instructions,
                api_key=floe_key,
                base_url=resolved_base_url,
            )

    @property
    def provider(self) -> str:
        # Tag emitted metrics as Floe-routed regardless of host (works for
        # self-hosted Floe on a custom domain).
        return FLOE_PROVIDER

    # synthesize() is inherited from the OpenAI TTS: it returns a ChunkedStream
    # that reads the body by the response Content-Type, so Floe's binary
    # /v1/audio/speech response is decoded correctly regardless of the
    # OpenAI-specific `stream_format` hint the parent sends.


def _require_secure_url(url: str) -> None:
    """Refuse to send a credential-bearing request over cleartext.

    Every request carries the Floe API key (plus any BYOK ``X-Floe-Provider-Key``).
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
        f"Floe requires an https base_url; got {url!r}. Refusing to send your "
        "Floe API key or provider key over a non-TLS connection."
    )


def _get_api_key(key: NotGivenOr[str]) -> str:
    floe_api_key = key if is_given(key) else os.environ.get("FLOE_API_KEY")
    if not floe_api_key:
        raise ValueError(
            "FLOE_API_KEY is required, either as argument or set FLOE_API_KEY environment variable"  # noqa: E501
        )
    return floe_api_key
