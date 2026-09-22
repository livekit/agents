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

import os
import platform
from typing import Any

import httpx
import openai
from openai.types import ReasoningEffort

from livekit.agents import __version__ as livekit_version, llm
from livekit.agents.llm import ChatContext, ToolChoice
from livekit.agents.llm.chat_context import ImageContent
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins.openai.llm import LLM as OpenAILLM

from .models import SarvamLLMModels

SARVAM_API_BASE = "https://api.sarvam.ai"
SARVAM_LLM_BASE_URL_V1 = f"{SARVAM_API_BASE}/v1"
SARVAM_LLM_BASE_URL_V2 = f"{SARVAM_API_BASE}/v2"
USER_AGENT = f"Livekit/{livekit_version} Python/{platform.python_version()}"

# ---------------------------------------------------------------------------
# Model capability / endpoint tables
# ---------------------------------------------------------------------------

_SUPPORTED_MODELS: set[str] = {
    "gemma4",
    "sarvam-105b",
    "glm5.2",
    "sarvam-105b-conversations",
}

# Models that use the /v1 endpoint (everything else uses /v2)
_V1_MODELS: set[str] = {
    "sarvam-105b-conversations",
}

# Models that support vision (image input)
_VISION_MODELS: set[str] = {
    "gemma4",
}

# Models that support reasoning_effort
_REASONING_EFFORT_MODELS: set[str] = {
    "gemma4",
    "sarvam-105b",
    "glm5.2",
}

# Models that support wiki_grounding
_WIKI_GROUNDING_MODELS: set[str] = {
    "gemma4",
    "sarvam-105b",
    "glm5.2",
}

# OpenAI SDK fields the Sarvam API does not accept
_UNSUPPORTED_OAI_FIELDS: set[str] = {
    "stream_options",
    "max_completion_tokens",
    "service_tier",
}

# Fields allowed in extra_body (Sarvam-specific pass-through)
_ALLOWED_EXTRA_BODY_PARAMS: set[str] = {
    "frequency_penalty",
    "max_tokens",
    "n",
    "presence_penalty",
    "seed",
    "stop",
    "wiki_grounding",
}


def _resolve_base_url(model: str) -> str:
    """Return the Sarvam API base URL for *model*.

    ``sarvam-105b-conversations`` uses ``/v1``; all other models use ``/v2``.
    """
    if model in _V1_MODELS:
        return SARVAM_LLM_BASE_URL_V1
    return SARVAM_LLM_BASE_URL_V2


def _api_version(model: str) -> str:
    return "v1" if model in _V1_MODELS else "v2"


def _validate_model(model: str) -> str:
    if model not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported Sarvam model '{model}'. Supported models: {sorted(_SUPPORTED_MODELS)}"
        )
    return model


def _get_api_key(key: NotGivenOr[str]) -> str:
    sarvam_api_key = key if is_given(key) else os.environ.get("SARVAM_API_KEY")
    if not sarvam_api_key:
        raise ValueError(
            "SARVAM_API_KEY is required, either as argument or "
            "set SARVAM_API_KEY environment variable"
        )
    return sarvam_api_key


def _filter_extra_body(extra_body: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in extra_body.items() if k in _ALLOWED_EXTRA_BODY_PARAMS}


def _has_image_content(chat_ctx: ChatContext) -> bool:
    """Return True if any user message in *chat_ctx* carries an image."""
    for msg in chat_ctx.messages():
        if msg.role != "user":
            continue
        for item in msg.content:
            if isinstance(item, ImageContent):
                return True
    return False


class LLM(OpenAILLM):
    """Sarvam LLM service — OpenAI-compatible chat completions.

    Supports four models:

    * ``gemma4`` — vision-capable, ``/v2`` endpoint
    * ``sarvam-105b`` — text-only, ``/v2`` endpoint
    * ``glm5.2`` — text-only, ``/v2`` endpoint
    * ``sarvam-105b-conversations`` — multi-turn optimized, ``/v1`` endpoint

    The endpoint (``/v1`` vs ``/v2``) is resolved automatically from the model.
    An explicit ``base_url`` always overrides automatic resolution.
    """

    def __init__(
        self,
        *,
        model: str | SarvamLLMModels = "sarvam-105b",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        wiki_grounding: NotGivenOr[bool] = NOT_GIVEN,
        stop: NotGivenOr[str | list[str]] = NOT_GIVEN,
        n: NotGivenOr[int] = NOT_GIVEN,
        seed: NotGivenOr[int] = NOT_GIVEN,
        frequency_penalty: NotGivenOr[float] = NOT_GIVEN,
        presence_penalty: NotGivenOr[float] = NOT_GIVEN,
        extra_headers: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        extra_body: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        """
        Create a new instance of Sarvam LLM.

        ``api_key`` must be set to your Sarvam API key, either using the
        argument or by setting the ``SARVAM_API_KEY`` environment variable.
        """
        validated_model = _validate_model(model)
        sarvam_api_key = _get_api_key(api_key)

        # Resolve base URL: explicit > model-derived
        resolved_base_url = base_url if is_given(base_url) else _resolve_base_url(validated_model)

        # ---- Merge auth / telemetry headers (always enforced) ----
        merged_headers: dict[str, str] = {}
        if is_given(extra_headers):
            merged_headers.update(extra_headers)
        merged_headers["api-subscription-key"] = sarvam_api_key
        merged_headers["User-Agent"] = USER_AGENT

        # ---- Build extra_body with Sarvam-specific fields ----
        merged_body: dict[str, Any] = {}
        if is_given(extra_body):
            merged_body.update(extra_body)
        if is_given(max_tokens):
            merged_body["max_tokens"] = max_tokens
        if is_given(stop):
            merged_body["stop"] = stop
        if is_given(n):
            merged_body["n"] = n
        if is_given(seed):
            merged_body["seed"] = seed
        if is_given(frequency_penalty):
            merged_body["frequency_penalty"] = frequency_penalty
        if is_given(presence_penalty):
            merged_body["presence_penalty"] = presence_penalty

        # wiki_grounding — only for supported models
        if is_given(wiki_grounding):
            if validated_model in _WIKI_GROUNDING_MODELS:
                merged_body["wiki_grounding"] = wiki_grounding
            # silently drop for unsupported models

        filtered_body = _filter_extra_body(merged_body)

        # reasoning_effort — only for supported models
        effective_reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN
        if is_given(reasoning_effort):
            if validated_model in _REASONING_EFFORT_MODELS:
                effective_reasoning_effort = reasoning_effort
            # silently drop for unsupported models

        super().__init__(
            model=validated_model,
            api_key=sarvam_api_key,
            base_url=resolved_base_url,
            client=client,
            user=user,
            temperature=temperature,
            top_p=top_p,
            tool_choice=tool_choice,
            reasoning_effort=effective_reasoning_effort,
            extra_headers=merged_headers,
            extra_body=filtered_body if filtered_body else NOT_GIVEN,
            timeout=timeout,
        )

        # Track the API key and version for runtime model switching
        self._sarvam_api_key = sarvam_api_key
        self._sarvam_api_version = _api_version(validated_model)

    # ------------------------------------------------------------------
    # Public overrides
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Sarvam"

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Update the model at runtime.

        Validates the new model and recreates the underlying client when the
        API version changes (``/v1`` ↔ ``/v2``).  When both old and new models
        share the same endpoint, the client is **not** recreated.
        """
        if not is_given(model):
            return

        new_model = _validate_model(model)
        new_version = _api_version(new_model)

        if new_version != self._sarvam_api_version:
            # Endpoint changed — recreate the client pointing at the new URL
            new_base_url = _resolve_base_url(new_model)
            self._client = _create_sarvam_client(
                api_key=self._sarvam_api_key,
                base_url=new_base_url,
            )
            self._sarvam_api_version = new_version

        self._opts.model = new_model

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[Any] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> Any:
        """Build a chat-completion stream with Sarvam-specific param stripping.

        * Strips ``stream_options``, ``max_completion_tokens``, ``service_tier``
          (the Sarvam API does not accept these OpenAI SDK fields).
        * Strips ``reasoning_effort`` for models that don't support it.
        * Rejects images sent to non-vision models (client-side ``ValueError``).
        * Rejects ``tool_choice`` without a non-empty ``tools`` array.
        """
        model = self._opts.model

        # --- Image rejection on non-vision models ---
        if model not in _VISION_MODELS and _has_image_content(chat_ctx):
            raise ValueError(
                f"Image input is not supported for model '{model}'. "
                f"Use 'gemma4' for vision capabilities."
            )

        # --- tool_choice without tools ---
        # 'none' and 'auto' are always valid (they don't require tools).
        # 'required' and named-function tool_choice require a non-empty tools array.
        effective_tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice
        effective_tools = tools or []
        if is_given(effective_tool_choice) and not effective_tools:
            tc_str = effective_tool_choice if isinstance(effective_tool_choice, str) else "function"
            if tc_str not in ("none", "auto"):
                raise ValueError(
                    "tool_choice requires a non-empty tools array. "
                    "Provide tools or set tool_choice to 'none' or 'auto'."
                )

        # --- Strip unsupported fields from caller-provided extra_kwargs ---
        merged_extra: dict[str, Any] = {}
        if is_given(extra_kwargs):
            merged_extra.update(extra_kwargs)

        for field in _UNSUPPORTED_OAI_FIELDS:
            merged_extra.pop(field, None)

        # Strip reasoning_effort for unsupported models at chat-time too
        if model not in _REASONING_EFFORT_MODELS:
            merged_extra.pop("reasoning_effort", None)

        return super().chat(
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            response_format=response_format,
            extra_kwargs=merged_extra if merged_extra else NOT_GIVEN,
        )


def _create_sarvam_client(
    *,
    api_key: str,
    base_url: str,
) -> openai.AsyncClient:
    """Create an ``openai.AsyncClient`` with Sarvam auth headers."""
    return openai.AsyncClient(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
        default_headers={
            "api-subscription-key": api_key,
            "User-Agent": USER_AGENT,
        },
        http_client=httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=60.0, write=5.0, pool=5.0),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=50,
                keepalive_expiry=120,
            ),
        ),
    )
