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

import json
import os
from dataclasses import dataclass
from typing import Any, Literal, cast

from google import genai
from google.auth._default_async import default_async
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
from livekit.agents import APIConnectionError, APIStatusError, llm, utils
from livekit.agents.llm import FunctionTool, ToolChoice
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .models import ChatModels
from .utils import to_chat_ctx, to_fnc_ctx


@dataclass
class _LLMOptions:
    model: ChatModels | str
    temperature: NotGivenOr[float]
    tool_choice: NotGivenOr[ToolChoice | Literal["auto", "required", "none"]]
    vertexai: NotGivenOr[bool]
    project: NotGivenOr[str]
    location: NotGivenOr[str]
    max_output_tokens: NotGivenOr[int]
    top_p: NotGivenOr[float]
    top_k: NotGivenOr[float]
    presence_penalty: NotGivenOr[float]
    frequency_penalty: NotGivenOr[float]


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: ChatModels | str = "gemini-2.0-flash-001",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        vertexai: NotGivenOr[bool] = False,
        project: NotGivenOr[str] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        top_k: NotGivenOr[float] = NOT_GIVEN,
        presence_penalty: NotGivenOr[float] = NOT_GIVEN,
        frequency_penalty: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice | Literal["auto", "required", "none"]] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Google GenAI LLM.

        Environment Requirements:
        - For VertexAI: Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the service account key file.
        The Google Cloud project and location can be set via `project` and `location` arguments or the environment variables
        `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`. By default, the project is inferred from the service account key file,
        and the location defaults to "us-central1".
        - For Google Gemini API: Set the `api_key` argument or the `GOOGLE_API_KEY` environment variable.

        Args:
            model (ChatModels | str, optional): The model name to use. Defaults to "gemini-2.0-flash-001".
            api_key (str, optional): The API key for Google Gemini. If not provided, it attempts to read from the `GOOGLE_API_KEY` environment variable.
            vertexai (bool, optional): Whether to use VertexAI. Defaults to False.
            project (str, optional): The Google Cloud project to use (only for VertexAI). Defaults to None.
            location (str, optional): The location to use for VertexAI API requests. Defaults value is "us-central1".
            temperature (float, optional): Sampling temperature for response generation. Defaults to 0.8.
            max_output_tokens (int, optional): Maximum number of tokens to generate in the output. Defaults to None.
            top_p (float, optional): The nucleus sampling probability for response generation. Defaults to None.
            top_k (int, optional): The top-k sampling value for response generation. Defaults to None.
            presence_penalty (float, optional): Penalizes the model for generating previously mentioned concepts. Defaults to None.
            frequency_penalty (float, optional): Penalizes the model for repeating words. Defaults to None.
            tool_choice (ToolChoice or Literal["auto", "required", "none"], optional): Specifies whether to use tools during response generation. Defaults to "auto".
        """  # noqa: E501
        super().__init__()
        gcp_project = project if is_given(project) else os.environ.get("GOOGLE_CLOUD_PROJECT")
        gcp_location = location if is_given(location) else os.environ.get("GOOGLE_CLOUD_LOCATION")
        gemini_api_key = api_key if is_given(api_key) else os.environ.get("GOOGLE_API_KEY")
        _gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if _gac is None:
            logger.warning(
                "`GOOGLE_APPLICATION_CREDENTIALS` environment variable is not set. please set it to the path of the service account key file. Otherwise, use any of the other Google Cloud auth methods."  # noqa: E501
            )

        if is_given(vertexai) and vertexai:
            if not gcp_project:
                _, gcp_project = default_async(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            gemini_api_key = None  # VertexAI does not require an API key

        else:
            gcp_project = None
            gcp_location = None
            if not gemini_api_key:
                raise ValueError(
                    "API key is required for Google API either via api_key or GOOGLE_API_KEY environment variable"  # noqa: E501
                )

        self._opts = _LLMOptions(
            model=model,
            temperature=temperature,
            tool_choice=tool_choice,
            vertexai=vertexai,
            project=project,
            location=location,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        self._client = genai.Client(
            api_key=gemini_api_key,
            vertexai=is_given(vertexai) and vertexai,
            project=gcp_project,
            location=gcp_location,
        )

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice | Literal["auto", "required", "none"]] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        extra = {}

        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice
        if is_given(tool_choice):
            gemini_tool_choice: types.ToolConfig
            if isinstance(tool_choice, dict):
                gemini_tool_choice = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=[tool_choice["function"]["name"]],
                    )
                )
                extra["tool_config"] = gemini_tool_choice
            elif tool_choice == "required":
                gemini_tool_choice = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=[fnc.name for fnc in tools],
                    )
                )
                extra["tool_config"] = gemini_tool_choice
            elif tool_choice == "auto":
                gemini_tool_choice = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="AUTO",
                    )
                )
                extra["tool_config"] = gemini_tool_choice
            elif tool_choice == "none":
                gemini_tool_choice = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="NONE",
                    )
                )
                extra["tool_config"] = gemini_tool_choice

        if is_given(self._opts.temperature):
            extra["temperature"] = self._opts.temperature
        if is_given(self._opts.max_output_tokens):
            extra["max_output_tokens"] = self._opts.max_output_tokens
        if is_given(self._opts.top_p):
            extra["top_p"] = self._opts.top_p
        if is_given(self._opts.top_k):
            extra["top_k"] = self._opts.top_k
        if is_given(self._opts.presence_penalty):
            extra["presence_penalty"] = self._opts.presence_penalty
        if is_given(self._opts.frequency_penalty):
            extra["frequency_penalty"] = self._opts.frequency_penalty

        return LLMStream(
            self,
            client=self._client,
            model=self._opts.model,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            extra_kwargs=extra,
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
        tools: list[FunctionTool] | None,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._client = client
        self._model = model
        self._llm: LLM = llm
        self._extra_kwargs = extra_kwargs

    async def _run(self) -> None:
        retryable = True
        request_id = utils.shortuuid()

        try:
            turns, system_instruction = to_chat_ctx(self._chat_ctx, id(self._llm))

            self._extra_kwargs["tools"] = [
                types.Tool(function_declarations=to_fnc_ctx(self._tools))
            ]
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                **self._extra_kwargs,
            )

            stream = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=cast(types.ContentListUnion, turns),
                config=config,
            )

            async for response in stream:
                if response.prompt_feedback:
                    raise APIStatusError(
                        response.prompt_feedback.json(),
                        retryable=False,
                        request_id=request_id,
                    )

                if (
                    not response.candidates
                    or not response.candidates[0].content
                    or not response.candidates[0].content.parts
                ):
                    raise APIStatusError(
                        "No candidates in the response",
                        retryable=True,
                        request_id=request_id,
                    )

                if len(response.candidates) > 1:
                    logger.warning(
                        "gemini llm: there are multiple candidates in the response, returning response from the first one."  # noqa: E501
                    )

                for part in response.candidates[0].content.parts:
                    chat_chunk = self._parse_part(request_id, part)
                    if chat_chunk is not None:
                        retryable = False
                        self._event_ch.send_nowait(chat_chunk)

                if response.usage_metadata is not None:
                    usage = response.usage_metadata
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            id=request_id,
                            usage=llm.CompletionUsage(
                                completion_tokens=usage.candidates_token_count or 0,
                                prompt_tokens=usage.prompt_token_count or 0,
                                total_tokens=usage.total_token_count or 0,
                            ),
                        )
                    )

        except ClientError as e:
            raise APIStatusError(
                "gemini llm: client error",
                status_code=e.code,
                body=e.message,
                request_id=request_id,
                retryable=False if e.code != 429 else True,
            ) from e
        except ServerError as e:
            raise APIStatusError(
                "gemini llm: server error",
                status_code=e.code,
                body=e.message,
                request_id=request_id,
                retryable=retryable,
            ) from e
        except APIError as e:
            raise APIStatusError(
                "gemini llm: api error",
                status_code=e.code,
                body=e.message,
                request_id=request_id,
                retryable=retryable,
            ) from e
        except Exception as e:
            raise APIConnectionError(
                "gemini llm: error generating content",
                retryable=retryable,
            ) from e

    def _parse_part(self, id: str, part: types.Part) -> llm.ChatChunk | None:
        if part.function_call:
            chat_chunk = llm.ChatChunk(
                id=id,
                delta=llm.ChoiceDelta(
                    role="assistant",
                    tool_calls=[
                        llm.FunctionToolCall(
                            arguments=json.dumps(part.function_call.args),
                            name=part.function_call.name,
                            call_id=part.function_call.id or utils.shortuuid("function_call_"),
                        )
                    ],
                    content=part.text,
                ),
            )
            return chat_chunk

        return llm.ChatChunk(
            id=id,
            delta=llm.ChoiceDelta(content=part.text, role="assistant"),
        )
