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
import os
from dataclasses import dataclass
from typing import Any, Literal, MutableSet, Union, cast

from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    llm,
    utils,
)
from livekit.agents.llm import LLMCapabilities, ToolChoice, _create_ai_function_info
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from google import genai
from google.auth._default_async import default_async
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError

from ._utils import _build_gemini_ctx, _build_tools
from .log import logger
from .models import ChatModels


@dataclass
class LLMOptions:
    model: ChatModels | str
    temperature: float | None
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
        model: ChatModels | str = "gemini-2.0-flash-exp",
        api_key: str | None = None,
        vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
        candidate_count: int = 1,
        temperature: float = 0.8,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
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
            model (ChatModels | str, optional): The model name to use. Defaults to "gemini-2.0-flash-exp".
            api_key (str, optional): The API key for Google Gemini. If not provided, it attempts to read from the `GOOGLE_API_KEY` environment variable.
            vertexai (bool, optional): Whether to use VertexAI. Defaults to False.
            project (str, optional): The Google Cloud project to use (only for VertexAI). Defaults to None.
            location (str, optional): The location to use for VertexAI API requests. Defaults value is "us-central1".
            candidate_count (int, optional): Number of candidate responses to generate. Defaults to 1.
            temperature (float, optional): Sampling temperature for response generation. Defaults to 0.8.
            max_output_tokens (int, optional): Maximum number of tokens to generate in the output. Defaults to None.
            top_p (float, optional): The nucleus sampling probability for response generation. Defaults to None.
            top_k (int, optional): The top-k sampling value for response generation. Defaults to None.
            presence_penalty (float, optional): Penalizes the model for generating previously mentioned concepts. Defaults to None.
            frequency_penalty (float, optional): Penalizes the model for repeating words. Defaults to None.
            tool_choice (ToolChoice or Literal["auto", "required", "none"], optional): Specifies whether to use tools during response generation. Defaults to "auto".
        """
        super().__init__(
            capabilities=LLMCapabilities(
                supports_choices_on_int=False,
                requires_persistent_functions=False,
            )
        )
        self._project_id = project or os.environ.get("GOOGLE_CLOUD_PROJECT", None)
        self._location = location or os.environ.get(
            "GOOGLE_CLOUD_LOCATION", "us-central1"
        )
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", None)
        _gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if _gac is None:
            logger.warning(
                "`GOOGLE_APPLICATION_CREDENTIALS` environment variable is not set. please set it to the path of the service account key file. Otherwise, use any of the other Google Cloud auth methods."
            )

        if vertexai:
            if not self._project_id:
                _, self._project_id = default_async(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            self._api_key = None  # VertexAI does not require an API key

        else:
            self._project_id = None
            self._location = None
            if not self._api_key:
                raise ValueError(
                    "API key is required for Google API either via api_key or GOOGLE_API_KEY environment variable"
                )

        self._opts = LLMOptions(
            model=model,
            temperature=temperature,
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
            api_key=self._api_key,
            vertexai=vertexai,
            project=self._project_id,
            location=self._location,
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
        self._tool_choice = tool_choice

    async def _run(self) -> None:
        retryable = True
        request_id = utils.shortuuid()

        try:
            opts: dict[str, Any] = dict()
            turns, system_instruction = _build_gemini_ctx(self._chat_ctx, id(self))

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
                system_instruction=system_instruction,
                **opts,
            )
            async for response in self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=cast(types.ContentListUnion, turns),
                config=config,
            ):
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
                        "gemini llm: there are multiple candidates in the response, returning response from the first one."
                    )

                for index, part in enumerate(response.candidates[0].content.parts):
                    chat_chunk = self._parse_part(request_id, index, part)
                    if chat_chunk is not None:
                        retryable = False
                        self._event_ch.send_nowait(chat_chunk)

                if response.usage_metadata is not None:
                    usage = response.usage_metadata
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            request_id=request_id,
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

    def _parse_part(
        self, id: str, index: int, part: types.Part
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
        self, id: str, index: int, part: types.Part
    ) -> llm.ChatChunk | None:
        if part.function_call is None:
            logger.warning("gemini llm: no function call in the response")
            return None

        if part.function_call.name is None:
            logger.warning("gemini llm: no function name in the response")
            return None

        if part.function_call.id is None:
            part.function_call.id = utils.shortuuid()

        if self._fnc_ctx is None:
            logger.warning(
                "google stream tried to run function without function context"
            )
            return None

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
