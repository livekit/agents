from __future__ import annotations

import os
from typing import Any

import httpx
import openai
from openai.types import ReasoningEffort
from openai.types.chat import completion_create_params

from livekit.agents import llm
from livekit.agents.llm import ChatContext, ToolChoice, utils as llm_utils
from livekit.agents.llm._provider_format.utils import convert_mid_conversation_instructions
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins.openai import LLM as OpenAILLM, LLMStream

from .models import LLMModels

# Model families with generations whose chat template rejects a system message that
# appears after the first turn: Gemma 2 and 3 raise on it (Gemma 3 also requires strict
# user/assistant alternation) and Qwen3.5 raises "System message must be at the
# beginning". Gemma 4 and Qwen3 render a later system turn without error, but the
# rewrite is applied to the whole family so a per-turn instruction is delivered the
# same way regardless of which generation a deployment serves. For these, the
# per-turn instructions LiveKit appends as trailing system messages
# (``generate_reply(instructions=...)``, expressive TTS guides) are inlined as
# ``<instructions>``-wrapped user messages instead. Matched case-insensitively against
# the model id so both Model API ids ("google/gemma-4-31B-it") and dedicated-deployment
# names work; pass ``inline_mid_conversation_instructions`` to override.
_INLINE_INSTRUCTIONS_MODEL_FAMILIES = ("gemma", "qwen")


def _needs_inline_instructions(model: str) -> bool:
    model = model.lower()
    return any(family in model for family in _INLINE_INSTRUCTIONS_MODEL_FAMILIES)


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str | LLMModels = "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        user: NotGivenOr[str] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        base_url: NotGivenOr[str] = "https://inference.baseten.co/v1",
        client: openai.AsyncClient | None = None,
        timeout: httpx.Timeout | None = None,
        inline_mid_conversation_instructions: NotGivenOr[bool] = NOT_GIVEN,
    ):
        """
        Create a new instance of Baseten LLM.

        ``api_key`` must be set to your Baseten API key, either using the argument or by setting
        the ``BASETEN_API_KEY`` environmental variable.

        ``inline_mid_conversation_instructions`` controls how system messages that appear
        after the conversation has started (e.g. ``generate_reply(instructions=...)``) are
        sent. When ``True`` they are rewritten as ``<instructions>``-wrapped user messages,
        which is required for models whose chat template only accepts a leading system
        message (Gemma 2, Gemma 3, Qwen3.5) and is applied to the Gemma and Qwen families
        as a whole. When ``False`` they are sent as-is, which is what models whose template
        renders system turns anywhere (GLM, Llama, Kimi, DeepSeek) expect. The default is
        inferred from ``model``; pass it explicitly for dedicated deployments whose model
        name does not reveal the model family.
        """
        api_key = api_key if is_given(api_key) else os.environ.get("BASETEN_API_KEY", "")
        if not api_key:
            raise ValueError(
                "BASETEN_API_KEY is required, either as argument or set BASETEN_API_KEY environmental variable"  # noqa: E501
            )

        if not is_given(reasoning_effort):
            if model == "openai/gpt-oss-120b":
                reasoning_effort = "low"

        if not is_given(inline_mid_conversation_instructions):
            inline_mid_conversation_instructions = _needs_inline_instructions(model)
        self._inline_mid_conversation_instructions = inline_mid_conversation_instructions

        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            user=user,
            safety_identifier=safety_identifier,
            prompt_cache_key=prompt_cache_key,
            temperature=temperature,
            top_p=top_p,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Baseten"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[
            completion_create_params.ResponseFormat | type[llm_utils.ResponseFormatT]
        ] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        if self._inline_mid_conversation_instructions:
            # Rewrite mid-conversation system messages before the OpenAI serializer sees
            # them. It passes system messages through as-is, so with none left after the
            # preamble the request is exactly what the OpenAI-compatible endpoint expects.
            # Returns a new ChatContext; the caller's context is left untouched.
            chat_ctx = convert_mid_conversation_instructions(chat_ctx)

        return super().chat(
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            response_format=response_format,
            extra_kwargs=extra_kwargs,
        )
