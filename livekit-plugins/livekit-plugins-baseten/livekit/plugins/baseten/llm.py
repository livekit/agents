from __future__ import annotations

import os

import httpx
import openai

from livekit.agents.llm import ToolChoice
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins.openai import LLM as OpenAILLM

from .models import LLMModels


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str | LLMModels = "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        base_url: NotGivenOr[str] = "https://inference.baseten.co/v1",
        client: openai.AsyncClient | None = None,
        timeout: httpx.Timeout | None = None,
    ):
        """
        Create a new instance of Baseten LLM.

        ``api_key`` must be set to your Baseten API key, either using the argument or by setting
        the ``BASETEN_API_KEY`` environmental variable.
        """
        api_key = api_key if is_given(api_key) else os.environ.get("BASETEN_API_KEY", "")
        if not api_key:
            raise ValueError(
                "BASETEN_API_KEY is required, either as argument or set BASETEN_API_KEY environmental variable"  # noqa: E501
            )
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            timeout=timeout,
        )
