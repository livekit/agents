import os

import httpx
from openai.types import Reasoning

from livekit.agents.llm import ToolChoice
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins import openai

XAI_BASE_URL = "https://api.x.ai/v1"


class LLM(openai.responses.LLM):
    def __init__(
        self,
        *,
        model: str = "grok-4-1-fast-non-reasoning",
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        reasoning: NotGivenOr[Reasoning] = NOT_GIVEN,
    ) -> None:
        api_key = api_key or os.environ.get("XAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "XAI API key is required, either as argument or set XAI_API_KEY environmental variable"  # noqa: E501
            )
        super().__init__(
            model=model,
            base_url=base_url if is_given(base_url) else XAI_BASE_URL,
            api_key=api_key,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            reasoning=reasoning,
            timeout=timeout,
        )
