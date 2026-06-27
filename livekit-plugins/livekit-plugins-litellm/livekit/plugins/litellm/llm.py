# Copyright 2024-2026, Daily
#
# SPDX-License-Identifier: Apache-2.0

"""LiteLLM LLM plugin for LiveKit Agents.

Routes completions through the LiteLLM SDK, which normalizes requests and
responses across 100+ LLM providers into a single OpenAI-compatible interface.
"""

from __future__ import annotations

from typing import Any

import httpx
import openai as _openai

from livekit.agents.llm import ToolChoice
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins.openai import LLM as OpenAILLM

_NOT_GIVEN_TYPE = type(_openai.NOT_GIVEN)


def _is_openai_sentinel(value: Any) -> bool:
    """Check if a value is an OpenAI SDK sentinel (omit, NOT_GIVEN, etc.)."""
    return isinstance(value, _NOT_GIVEN_TYPE) or type(value).__name__ in ("Omit", "_Omit")


class _LiteLLMCompletions:
    """Shim mapping ``chat.completions.create(**kw)`` to ``litellm.acompletion(**kw)``."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    async def create(self, **kwargs: Any) -> Any:
        try:
            import litellm
        except ImportError as err:
            raise ImportError(
                "litellm is required for livekit-plugins-litellm. "
                "Install with: pip install 'livekit-plugins-litellm'"
            ) from err

        # Strip OpenAI sentinel values (omit, NOT_GIVEN) that litellm doesn't understand
        cleaned = {k: v for k, v in kwargs.items() if not _is_openai_sentinel(v)}

        # Convert httpx.Timeout to float for litellm
        timeout = cleaned.get("timeout")
        if isinstance(timeout, httpx.Timeout):
            cleaned["timeout"] = timeout.read or timeout.connect or 30.0

        # Strip LiveKit-specific headers that litellm doesn't need
        cleaned.pop("extra_headers", None)

        # Forward api_key if provided at construction time
        if self._api_key:
            cleaned.setdefault("api_key", self._api_key)

        cleaned.setdefault("drop_params", True)
        return await litellm.acompletion(**cleaned)


class _LiteLLMChat:
    def __init__(self, api_key: str | None = None) -> None:
        self.completions = _LiteLLMCompletions(api_key=api_key)


class _LiteLLMClientShim:
    """AsyncOpenAI-shaped facade that routes through the LiteLLM SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        self.chat = _LiteLLMChat(api_key=api_key)

    class _BaseURL:
        netloc = b"litellm"

    _base_url = _BaseURL()

    async def close(self) -> None:
        pass


class LLM(OpenAILLM):
    """LLM service that routes through LiteLLM to any of 100+ providers.

    Uses the LiteLLM SDK in embedded mode (no proxy server required). Every
    call goes through ``litellm.acompletion(model=...)``, which handles
    provider-specific parameter translation and authentication.

    Specify the model with the standard LiteLLM provider-prefixed name::

        from livekit.plugins.litellm import LLM

        llm = LLM(model="anthropic/claude-sonnet-4-6")

    LiteLLM reads provider-specific env vars automatically
    (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, etc.).
    """

    def __init__(
        self,
        *,
        model: str = "openai/gpt-4o",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        max_retries: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """Create a new instance of LiteLLM LLM.

        Args:
            model: LiteLLM model identifier (e.g. "anthropic/claude-sonnet-4-6",
                "openai/gpt-4o", "vertex_ai/gemini-2.5-flash").
            api_key: Optional API key. Forwarded to litellm.acompletion() on every
                call. When omitted, LiteLLM resolves credentials from
                provider-specific env vars.
            temperature: Sampling temperature.
            top_p: Top-p (nucleus) sampling parameter.
            parallel_tool_calls: Whether to allow parallel tool calls.
            tool_choice: Tool choice configuration.
            max_completion_tokens: Maximum completion tokens to generate.
            timeout: HTTP timeout configuration.
            max_retries: Maximum number of retries on transient errors.
        """
        resolved_key = api_key if is_given(api_key) else None

        super().__init__(
            model=model,
            api_key="unused",
            client=_LiteLLMClientShim(api_key=resolved_key),  # type: ignore[arg-type]
            temperature=temperature,
            top_p=top_p,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            max_completion_tokens=max_completion_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
