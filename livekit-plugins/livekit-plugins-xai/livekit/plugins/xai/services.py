from __future__ import annotations

import os

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
        model: str | LLMModels = "grok-3",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = "https://api.x.ai/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
    ):
        """
        Create a new instance of xAI LLM.

        ``api_key`` must be set to your xAI API key, either using the argument or by setting
        the ``XAI_API_KEY`` environmental variable.
        """
        super().__init__(
            model=model,
            api_key=_get_api_key(api_key),
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            _strict_tool_call=False,
        )


def _get_api_key(key: NotGivenOr[str]) -> str:
    xai_api_key = key if is_given(key) else os.environ.get("XAI_API_KEY")
    if not xai_api_key:
        raise ValueError(
            "XAI_API_KEY is required, either as argument or set XAI_API_KEY environmental variable"  # noqa: E501
        )
    return xai_api_key
