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

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal, MutableSet, Union

from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    llm,
    utils,
)
from livekit.agents.llm import ToolChoice, _create_ai_function_info
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from google import genai  # type: ignore
from google.genai import types  # type: ignore

from ._utils import _build_gemini_ctx, _build_tools
from .log import logger
from .models import (
    ChatModels,
)


@dataclass
class LLMOptions:
    model: str
    temperature: float | None
    parallel_tool_calls: bool | None
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto"
    vertexai: bool = False
    project: str | None = None
    location: str | None = None
    candidate_count: int = 1
    max_output_tokens: int | None = None
    top_p: float | None = None
    top_k: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str = "gemini-2.0-flash-exp",
        api_key: str | None = None,
        vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
        candidate_count: int = 1,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
    ) -> None:
        """
        Create a new instance of Google GenAI LLM.
        """
        super().__init__()
        self._capabilities = llm.LLMCapabilities(supports_choices_on_int=False)

        self._opts = LLMOptions(
            model=model,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            vertexai=vertexai,
            project=project,
            location=location,
            candidate_count=candidate_count,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        self._client = genai.Client(
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
        )
        self._running_fncs: MutableSet[asyncio.Task[Any]] = set()

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]]
        | None = None,
    ) -> "LLMStream":
        if parallel_tool_calls is None:
            parallel_tool_calls = self._opts.parallel_tool_calls

        if tool_choice is None:
            tool_choice = self._opts.tool_choice

        if temperature is None:
            temperature = self._opts.temperature

        return LLMStream(
            self,
            client=self._client,
            model=self._opts.model,
            max_output_tokens=self._opts.max_output_tokens,
            top_p=self._opts.top_p,
            top_k=self._opts.top_k,
            presence_penalty=self._opts.presence_penalty,
            frequency_penalty=self._opts.frequency_penalty,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            conn_options=conn_options,
            n=n,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        client: genai.Client,
        model: str | ChatModels,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions,
        fnc_ctx: llm.FunctionContext | None,
        temperature: float | None,
        n: int | None,
        max_output_tokens: int | None,
        top_p: float | None,
        top_k: float | None,
        presence_penalty: float | None,
        frequency_penalty: float | None,
        parallel_tool_calls: bool | None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]],
    ) -> None:
        super().__init__(
            llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx, conn_options=conn_options
        )
        self._client = client
        self._model = model
        self._llm: LLM = llm
        self._max_output_tokens = max_output_tokens
        self._top_p = top_p
        self._top_k = top_k
        self._presence_penalty = presence_penalty
        self._frequency_penalty = frequency_penalty

        self._temperature = temperature
        self._n = n
        self._parallel_tool_calls = parallel_tool_calls
        self._tool_choice = tool_choice

    async def _run(self) -> None:
        retryable = True

        try:
            opts: dict[str, Any] = dict()
            ctx = _build_gemini_ctx(self._chat_ctx, id(self))
            if ctx.get("system_instruction"):
                opts["system_instruction"] = ctx.get("system_instruction")

            if self._fnc_ctx and len(self._fnc_ctx.ai_functions) > 0:
                functions = _build_tools(self._fnc_ctx)
                opts["tools"] = [types.Tool(function_declarations=functions)]

                if self._tool_choice is not None:
                    if isinstance(self._tool_choice, ToolChoice):
                        # specific function
                        tool_config = types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="ANY",
                                allowed_function_names=[self._tool_choice.name],
                            )
                        )
                    elif self._tool_choice == "required":
                        # model must call any function
                        tool_config = types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="ANY",
                                allowed_function_names=[
                                    fnc.name
                                    for fnc in self._fnc_ctx.ai_functions.values()
                                ],
                            )
                        )
                    elif self._tool_choice == "auto":
                        # model can call any function
                        tool_config = types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="AUTO"
                            )
                        )
                    elif self._tool_choice == "none":
                        # model cannot call any function
                        tool_config = types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="NONE",
                            )
                        )
                    opts["tool_config"] = tool_config

            config = types.GenerateContentConfig(
                candidate_count=self._n,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
                top_p=self._top_p,
                top_k=self._top_k,
                presence_penalty=self._presence_penalty,
                frequency_penalty=self._frequency_penalty,
                **opts,
            )
            async for response in self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=ctx.get("turns"),
                config=config,
            ):
                response_id = utils.shortuuid()
                if response.prompt_feedback:
                    logger.warning(
                        "genai llm prompt feedback: %s",
                        response.prompt_feedback,
                    )
                    raise APIStatusError(response.prompt_feedback)

                if (
                    not response.candidates
                    or not response.candidates[0].content
                    or not response.candidates[0].content.parts
                ):
                    raise APIStatusError("No candidates in the response")

                if len(response.candidates) > 1:
                    logger.warning(
                        "Warning: there are multiple candidates in the response, returning"
                        " function calls from the first one."
                    )

                for index, part in enumerate(response.candidates[0].content.parts):
                    chat_chunk = self._parse_part(response_id, index, part)
                    if chat_chunk is not None:
                        retryable = False
                        self._event_ch.send_nowait(chat_chunk)

                if response.usage_metadata is not None:
                    usage = response.usage_metadata
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            request_id=response_id,
                            usage=llm.CompletionUsage(
                                completion_tokens=usage.candidates_token_count,
                                prompt_tokens=usage.prompt_token_count,
                                total_tokens=usage.total_token_count,
                            ),
                        )
                    )

        except Exception as e:
            print("\ne", e)
            raise APIConnectionError(
                "Error generating content", retryable=retryable
            ) from e

    def _parse_part(
        self, id: str, index: int, part: types.ContentPart
    ) -> llm.ChatChunk | None:
        if part.function_call:
            return self._try_build_function(id, index, part)

        return llm.ChatChunk(
            request_id=id,
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(content=part.text, role="assistant"),
                    index=index,
                )
            ],
        )

    def _try_build_function(
        self, id: str, index: int, part: types.ContentPart
    ) -> llm.ChatChunk | None:
        if part.function_call.id is None:
            part.function_call.id = utils.shortuuid()
        print(f"\nfunction_call: {part.function_call}")
        fnc_info = _create_ai_function_info(
            self._fnc_ctx,
            part.function_call.id,
            part.function_call.name,
            json.dumps(part.function_call.args),
        )

        self._function_calls_info.append(fnc_info)

        return llm.ChatChunk(
            request_id=id,
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        tool_calls=[fnc_info],
                        content=part.text,
                    ),
                    index=index,
                )
            ],
        )
