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
from typing import Any, cast

from google.auth._default_async import default_async
from google.genai import Client, types
from google.genai.errors import APIError, ClientError, ServerError
from livekit.agents import APIConnectionError, APIStatusError, llm, utils
from livekit.agents.llm import FunctionTool, RawFunctionTool, ToolChoice, utils as llm_utils
from livekit.agents.llm.tool_context import (
    get_function_info,
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .models import ChatModels
from .tools import _LLMTool
from .utils import create_tools_config, to_fnc_ctx, to_response_format


@dataclass
class _LLMOptions:
    model: ChatModels | str
    temperature: NotGivenOr[float]
    tool_choice: NotGivenOr[ToolChoice]
    vertexai: NotGivenOr[bool]
    project: NotGivenOr[str]
    location: NotGivenOr[str]
    max_output_tokens: NotGivenOr[int]
    top_p: NotGivenOr[float]
    top_k: NotGivenOr[float]
    presence_penalty: NotGivenOr[float]
    frequency_penalty: NotGivenOr[float]
    thinking_config: NotGivenOr[types.ThinkingConfigOrDict]
    automatic_function_calling_config: NotGivenOr[types.AutomaticFunctionCallingConfigOrDict]
    gemini_tools: NotGivenOr[list[_LLMTool]]
    http_options: NotGivenOr[types.HttpOptions]
    seed: NotGivenOr[int]


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: ChatModels | str = "gemini-2.0-flash-001",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        vertexai: NotGivenOr[bool] = NOT_GIVEN,
        project: NotGivenOr[str] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        top_k: NotGivenOr[float] = NOT_GIVEN,
        presence_penalty: NotGivenOr[float] = NOT_GIVEN,
        frequency_penalty: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        thinking_config: NotGivenOr[types.ThinkingConfigOrDict] = NOT_GIVEN,
        automatic_function_calling_config: NotGivenOr[
            types.AutomaticFunctionCallingConfigOrDict
        ] = NOT_GIVEN,
        gemini_tools: NotGivenOr[list[_LLMTool]] = NOT_GIVEN,
        http_options: NotGivenOr[types.HttpOptions] = NOT_GIVEN,
        seed: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Google GenAI LLM.

        Environment Requirements:
        - For VertexAI: Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the service account key file or use any of the other Google Cloud auth methods.
        The Google Cloud project and location can be set via `project` and `location` arguments or the environment variables
        `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`. By default, the project is inferred from the service account key file,
        and the location defaults to "us-central1".
        - For Google Gemini API: Set the `api_key` argument or the `GOOGLE_API_KEY` environment variable.

        Args:
            model (ChatModels | str, optional): The model name to use. Defaults to "gemini-2.0-flash-001".
            api_key (str, optional): The API key for Google Gemini. If not provided, it attempts to read from the `GOOGLE_API_KEY` environment variable.
            vertexai (bool, optional): Whether to use VertexAI. If not provided, it attempts to read from the `GOOGLE_GENAI_USE_VERTEXAI` environment variable. Defaults to False.
                project (str, optional): The Google Cloud project to use (only for VertexAI). Defaults to None.
                location (str, optional): The location to use for VertexAI API requests. Defaults value is "us-central1".
            temperature (float, optional): Sampling temperature for response generation. Defaults to 0.8.
            max_output_tokens (int, optional): Maximum number of tokens to generate in the output. Defaults to None.
            top_p (float, optional): The nucleus sampling probability for response generation. Defaults to None.
            top_k (int, optional): The top-k sampling value for response generation. Defaults to None.
            presence_penalty (float, optional): Penalizes the model for generating previously mentioned concepts. Defaults to None.
            frequency_penalty (float, optional): Penalizes the model for repeating words. Defaults to None.
            tool_choice (ToolChoice, optional): Specifies whether to use tools during response generation. Defaults to "auto".
            thinking_config (ThinkingConfigOrDict, optional): The thinking configuration for response generation. Defaults to None.
            automatic_function_calling_config (AutomaticFunctionCallingConfigOrDict, optional): The automatic function calling configuration for response generation. Defaults to None.
            gemini_tools (list[LLMTool], optional): The Gemini-specific tools to use for the session.
            http_options (HttpOptions, optional): The HTTP options to use for the session.
        """  # noqa: E501
        super().__init__()
        gcp_project = project if is_given(project) else os.environ.get("GOOGLE_CLOUD_PROJECT")
        gcp_location: str | None = (
            location
            if is_given(location)
            else os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
        )
        use_vertexai = (
            vertexai
            if is_given(vertexai)
            else os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "0").lower() in ["true", "1"]
        )
        gemini_api_key = api_key if is_given(api_key) else os.environ.get("GOOGLE_API_KEY")

        if use_vertexai:
            if not gcp_project:
                _, gcp_project = default_async(  # type: ignore
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

        # Validate thinking_config
        if is_given(thinking_config):
            _thinking_budget = None
            if isinstance(thinking_config, dict):
                _thinking_budget = thinking_config.get("thinking_budget")
            elif isinstance(thinking_config, types.ThinkingConfig):
                _thinking_budget = thinking_config.thinking_budget

            if _thinking_budget is not None:
                if not isinstance(_thinking_budget, int):
                    raise ValueError("thinking_budget inside thinking_config must be an integer")
                if not (0 <= _thinking_budget <= 24576):
                    raise ValueError(
                        "thinking_budget inside thinking_config must be between 0 and 24576"
                    )

        self._opts = _LLMOptions(
            model=model,
            temperature=temperature,
            tool_choice=tool_choice,
            vertexai=use_vertexai,
            project=project,
            location=location,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            thinking_config=thinking_config,
            automatic_function_calling_config=automatic_function_calling_config,
            gemini_tools=gemini_tools,
            http_options=http_options,
            seed=seed,
        )
        self._client = Client(
            api_key=gemini_api_key,
            vertexai=use_vertexai,
            project=gcp_project,
            location=gcp_location,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[
            types.SchemaUnion | type[llm_utils.ResponseFormatT]
        ] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        gemini_tools: NotGivenOr[list[_LLMTool]] = NOT_GIVEN,
    ) -> LLMStream:
        extra = {}

        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        tool_choice = (
            cast(ToolChoice, tool_choice) if is_given(tool_choice) else self._opts.tool_choice
        )
        if is_given(tool_choice):
            gemini_tool_choice: types.ToolConfig
            if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                gemini_tool_choice = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.ANY,
                        allowed_function_names=[tool_choice["function"]["name"]],
                    )
                )
                extra["tool_config"] = gemini_tool_choice
            elif tool_choice == "required":
                tool_names = []
                for tool in tools or []:
                    if is_function_tool(tool):
                        tool_names.append(get_function_info(tool).name)
                    elif is_raw_function_tool(tool):
                        tool_names.append(get_raw_function_info(tool).name)

                gemini_tool_choice = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.ANY,
                        allowed_function_names=tool_names or None,
                    )
                )
                extra["tool_config"] = gemini_tool_choice
            elif tool_choice == "auto":
                gemini_tool_choice = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.AUTO,
                    )
                )
                extra["tool_config"] = gemini_tool_choice
            elif tool_choice == "none":
                gemini_tool_choice = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.NONE,
                    )
                )
                extra["tool_config"] = gemini_tool_choice

        if is_given(response_format):
            extra["response_schema"] = to_response_format(response_format)  # type: ignore
            extra["response_mime_type"] = "application/json"

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
        if is_given(self._opts.seed):
            extra["seed"] = self._opts.seed

        # Add thinking config if thinking_budget is provided
        if is_given(self._opts.thinking_config):
            extra["thinking_config"] = self._opts.thinking_config

        if is_given(self._opts.automatic_function_calling_config):
            extra["automatic_function_calling"] = self._opts.automatic_function_calling_config

        gemini_tools = gemini_tools if is_given(gemini_tools) else self._opts.gemini_tools

        return LLMStream(
            self,
            client=self._client,
            model=self._opts.model,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            gemini_tools=gemini_tools,
            extra_kwargs=extra,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        client: Client,
        model: str | ChatModels,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions,
        tools: list[FunctionTool | RawFunctionTool],
        extra_kwargs: dict[str, Any],
        gemini_tools: NotGivenOr[list[_LLMTool]] = NOT_GIVEN,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._client = client
        self._model = model
        self._llm: LLM = llm
        self._extra_kwargs = extra_kwargs
        self._gemini_tools = gemini_tools

    async def _run(self) -> None:
        retryable = True
        request_id = utils.shortuuid()

        try:
            turns_dict, extra_data = self._chat_ctx.to_provider_format(format="google")
            turns = [types.Content.model_validate(turn) for turn in turns_dict]
            function_declarations = to_fnc_ctx(self._tools)
            tools_config = create_tools_config(
                function_tools=function_declarations,
                gemini_tools=self._gemini_tools if is_given(self._gemini_tools) else None,
            )
            if tools_config:
                self._extra_kwargs["tools"] = tools_config

            config = types.GenerateContentConfig(
                system_instruction=(
                    [types.Part(text=content) for content in extra_data.system_messages]
                    if extra_data.system_messages
                    else None
                ),
                http_options=(
                    self._llm._opts.http_options
                    or types.HttpOptions(timeout=int(self._conn_options.timeout * 1000))
                ),
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
                    logger.warning(f"no candidates in the response: {response}")
                    continue

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
                                prompt_cached_tokens=usage.cached_content_token_count or 0,
                                total_tokens=usage.total_token_count or 0,
                            ),
                        )
                    )

        except ClientError as e:
            raise APIStatusError(
                "gemini llm: client error",
                status_code=e.code,
                body=f"{e.message} {e.status}",
                request_id=request_id,
                retryable=False if e.code != 429 else True,
            ) from e
        except ServerError as e:
            raise APIStatusError(
                "gemini llm: server error",
                status_code=e.code,
                body=f"{e.message} {e.status}",
                request_id=request_id,
                retryable=retryable,
            ) from e
        except APIError as e:
            raise APIStatusError(
                "gemini llm: api error",
                status_code=e.code,
                body=f"{e.message} {e.status}",
                request_id=request_id,
                retryable=retryable,
            ) from e
        except Exception as e:
            raise APIConnectionError(
                f"gemini llm: error generating content {str(e)}",
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
                            name=part.function_call.name,  # type: ignore
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
