from __future__ import annotations

import os

import httpx
import openai
from openai.types import ReasoningEffort

from livekit.agents.llm import ToolChoice
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins.openai import LLM as OpenAILLM, STT as OpenAISTT

from .models import LLMModels, STTModels


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str | LLMModels = "llama-3.3-70b-versatile",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        user: NotGivenOr[str] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        base_url: NotGivenOr[str] = "https://api.groq.com/openai/v1",
        metadata: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        service_tier: NotGivenOr[str] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        max_retries: NotGivenOr[int] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
    ):
        """
        Create a new instance of Groq LLM.

        ``api_key`` must be set to your Groq API key, either using the argument or by setting
        the ``GROQ_API_KEY`` environmental variable.
        """

        if not is_given(reasoning_effort):
            if model in ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
                reasoning_effort = "low"
            elif model in ["qwen/qwen3-32b"]:
                reasoning_effort = "none"

        super().__init__(
            model=model,
            api_key=_get_api_key(api_key),
            base_url=base_url,
            client=client,
            user=user,
            safety_identifier=safety_identifier,
            prompt_cache_key=prompt_cache_key,
            temperature=temperature,
            top_p=top_p,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            timeout=timeout,
            max_retries=max_retries,
            metadata=metadata,
            max_completion_tokens=max_completion_tokens,
        )


class STT(OpenAISTT):
    def __init__(
        self,
        *,
        model: STTModels | str = "whisper-large-v3-turbo",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = "https://api.groq.com/openai/v1",
        client: openai.AsyncClient | None = None,
        language: str = "en",
        prompt: NotGivenOr[str] = NOT_GIVEN,
        detect_language: bool = False,
    ):
        """
        Create a new instance of Groq STT.

        ``api_key`` must be set to your Groq API key, either using the argument or by setting
        the ``GROQ_API_KEY`` environmental variable.
        """
        super().__init__(
            model=model,
            api_key=_get_api_key(api_key),
            base_url=base_url,
            client=client,
            language=language,
            detect_language=detect_language,
            prompt=prompt,
            use_realtime=False,
        )


def _get_api_key(key: NotGivenOr[str]) -> str:
    groq_api_key = key if is_given(key) else os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is required, either as argument or set GROQ_API_KEY environmental variable"  # noqa: E501
        )
    return groq_api_key
