from __future__ import annotations

import os

import httpx
import openai as openai_api
from openai.types import Reasoning

from livekit.agents.llm import ToolChoice
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins import openai

from ..models import PerplexityResponsesModels
from ..version import __version__

PERPLEXITY_RESPONSES_BASE_URL = "https://api.perplexity.ai/v1"
_ATTRIBUTION_HEADER = {"X-Pplx-Integration": f"livekit-agents/{__version__}"}


def _create_client(
    *,
    api_key: str,
    base_url: str,
    timeout: httpx.Timeout | None,
) -> openai_api.AsyncClient:
    return openai_api.AsyncClient(
        api_key=api_key,
        base_url=base_url,
        default_headers=_ATTRIBUTION_HEADER,
        max_retries=0,
        http_client=httpx.AsyncClient(
            timeout=timeout
            if timeout
            else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=50,
                keepalive_expiry=120,
            ),
        ),
    )


class LLM(openai.responses.LLM):
    def __init__(
        self,
        *,
        model: str | PerplexityResponsesModels = "perplexity/sonar",
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        reasoning: NotGivenOr[Reasoning] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Perplexity Responses LLM.

        ``api_key`` must be set to your Perplexity API key, either using the argument or by
        setting the ``PERPLEXITY_API_KEY`` environmental variable.
        """
        api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if api_key is None:
            raise ValueError(
                "Perplexity API key is required, either as argument or set "
                "PERPLEXITY_API_KEY environmental variable"
            )

        resolved_base_url = base_url if is_given(base_url) else PERPLEXITY_RESPONSES_BASE_URL
        client = _create_client(api_key=api_key, base_url=resolved_base_url, timeout=timeout)

        super().__init__(
            model=model,
            base_url=resolved_base_url,
            api_key=api_key,
            client=client,
            use_websocket=False,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            reasoning=reasoning,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
        )
        self._perplexity_client = client

    async def aclose(self) -> None:
        await super().aclose()
        await self._perplexity_client.close()
