import os
from typing import Literal, Union

import openai
from livekit.agents.llm import ToolChoice
from livekit.plugins.openai import LLM as OpenAILLM
from livekit.plugins.openai import STT as OpenAISTT

from .models import LLMModels, STTModels


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str | LLMModels = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        user: str | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
        max_tokens: int | None = None,
        base_url: str | None = "https://api.groq.com/openai/v1",
        client: openai.AsyncClient | None = None,
    ):
        """
        Create a new instance of Groq LLM.

        ``api_key`` must be set to your Groq API key, either using the argument or by setting
        the ``GROQ_API_KEY`` environmental variable.
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
            max_tokens=max_tokens,
        )


class STT(OpenAISTT):
    def __init__(
        self,
        *,
        model: STTModels | str = "whisper-large-v3-turbo",
        api_key: str | None = None,
        base_url: str | None = "https://api.groq.com/openai/v1",
        client: openai.AsyncClient | None = None,
        language: str = "en",
        prompt: str | None = None,
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
        )


def _get_api_key(key: str | None) -> str:
    key = key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "GROQ_API_KEY is required, either as argument or set GROQ_API_KEY environmental variable"
        )
    return key
