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
import os
from dataclasses import dataclass
from typing import Any, Literal

import boto3

from livekit.agents import APIConnectionError, APIStatusError, llm
from livekit.agents.llm import ChatContext, FunctionTool, FunctionToolCall, ToolChoice
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .utils import get_aws_credentials, to_chat_ctx, to_fnc_ctx

TEXT_MODEL = Literal["anthropic.claude-3-5-sonnet-20241022-v2:0"]


@dataclass
class _LLMOptions:
    model: str | TEXT_MODEL
    temperature: NotGivenOr[float]
    tool_choice: NotGivenOr[ToolChoice | Literal["auto", "required", "none"]]
    max_output_tokens: NotGivenOr[int]
    top_p: NotGivenOr[float]
    additional_request_fields: NotGivenOr[dict[str, Any]]


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: NotGivenOr[str | TEXT_MODEL] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        region: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice | Literal["auto", "required", "none"]] = NOT_GIVEN,
        additional_request_fields: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of AWS Bedrock LLM.

        ``api_key``  and ``api_secret`` must be set to your AWS Access key id and secret access key, either using the argument or by setting the
        ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY`` environmental variables.

        See https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse_stream.html for more details on the AWS Bedrock Runtime API.

        Args:
            model (TEXT_MODEL, optional): model or inference profile arn to use(https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-use.html). Defaults to 'anthropic.claude-3-5-sonnet-20240620-v1:0'.
            api_key(str, optional): AWS access key id.
            api_secret(str, optional): AWS secret access key
            region (str, optional): The region to use for AWS API requests. Defaults value is "us-east-1".
            temperature (float, optional): Sampling temperature for response generation. Defaults to 0.8.
            max_output_tokens (int, optional): Maximum number of tokens to generate in the output. Defaults to None.
            top_p (float, optional): The nucleus sampling probability for response generation. Defaults to None.
            tool_choice (ToolChoice or Literal["auto", "required", "none"], optional): Specifies whether to use tools during response generation. Defaults to "auto".
            additional_request_fields (dict[str, Any], optional): Additional request fields to send to the AWS Bedrock Converse API. Defaults to None.
        """  # noqa: E501
        super().__init__()
        self._api_key, self._api_secret, self._region = get_aws_credentials(
            api_key, api_secret, region
        )

        model = model if is_given(model) else os.environ.get("BEDROCK_INFERENCE_PROFILE_ARN")
        if not model:
            raise ValueError(
                "model or inference profile arn must be set using the argument or by setting the BEDROCK_INFERENCE_PROFILE_ARN environment variable."  # noqa: E501
            )
        self._opts = _LLMOptions(
            model=model,
            temperature=temperature,
            tool_choice=tool_choice,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            additional_request_fields=additional_request_fields,
        )

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice | Literal["auto", "required", "none"]] = NOT_GIVEN,
    ) -> LLMStream:
        opts = {}

        if is_given(self._opts.model):
            opts["modelId"] = self._opts.model

        def _get_tool_config() -> dict[str, Any] | None:
            nonlocal tool_choice

            if not tools:
                return None

            tool_config: dict[str, Any] = {"tools": to_fnc_ctx(tools)}
            tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice
            if is_given(tool_choice):
                if isinstance(tool_choice, ToolChoice):
                    tool_config["toolChoice"] = {"tool": {"name": tool_choice.name}}
                elif tool_choice == "required":
                    tool_config["toolChoice"] = {"any": {}}
                elif tool_choice == "auto":
                    tool_config["toolChoice"] = {"auto": {}}
                else:
                    return None

            return tool_config

        tool_config = _get_tool_config()
        if tool_config:
            opts["toolConfig"] = tool_config
        messages, system_message = to_chat_ctx(chat_ctx, id(self))
        opts["messages"] = messages
        if system_message:
            opts["system"] = [system_message]

        inference_config = {}
        if is_given(self._opts.max_output_tokens):
            inference_config["maxTokens"] = self._opts.max_output_tokens
        temperature = temperature if is_given(temperature) else self._opts.temperature
        if is_given(temperature):
            inference_config["temperature"] = temperature
        if is_given(self._opts.top_p):
            inference_config["topP"] = self._opts.top_p

        opts["inferenceConfig"] = inference_config
        if is_given(self._opts.additional_request_fields):
            opts["additionalModelRequestFields"] = self._opts.additional_request_fields

        return LLMStream(
            self,
            aws_access_key_id=self._api_key,
            aws_secret_access_key=self._api_secret,
            region_name=self._region,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            extra_kwargs=opts,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str,
        chat_ctx: ChatContext,
        conn_options: APIConnectOptions,
        tools: list[FunctionTool] | None,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        self._llm: LLM = llm
        self._opts = extra_kwargs

        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None
        self._text: str = ""

    async def _run(self) -> None:
        retryable = True
        try:
            response = self._client.converse_stream(**self._opts)  # type: ignore
            request_id = response["ResponseMetadata"]["RequestId"]
            if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                raise APIStatusError(
                    f"aws bedrock llm: error generating content: {response}",
                    retryable=False,
                    request_id=request_id,
                )

            for chunk in response["stream"]:
                chat_chunk = self._parse_chunk(request_id, chunk)
                if chat_chunk is not None:
                    retryable = False
                    self._event_ch.send_nowait(chat_chunk)

                # Let other coroutines run
                await asyncio.sleep(0)

        except Exception as e:
            raise APIConnectionError(
                f"aws bedrock llm: error generating content: {e}",
                retryable=retryable,
            ) from e

    def _parse_chunk(self, request_id: str, chunk: dict) -> llm.ChatChunk | None:
        if "contentBlockStart" in chunk:
            tool_use = chunk["contentBlockStart"]["start"]["toolUse"]
            self._tool_call_id = tool_use["toolUseId"]
            self._fnc_name = tool_use["name"]
            self._fnc_raw_arguments = ""

        elif "contentBlockDelta" in chunk:
            delta = chunk["contentBlockDelta"]["delta"]
            if "toolUse" in delta:
                self._fnc_raw_arguments += delta["toolUse"]["input"]
            elif "text" in delta:
                self._text += delta["text"]

        elif "metadata" in chunk:
            metadata = chunk["metadata"]
            return llm.ChatChunk(
                request_id=request_id,
                usage=llm.CompletionUsage(
                    completion_tokens=metadata["usage"]["outputTokens"],
                    prompt_tokens=metadata["usage"]["inputTokens"],
                    total_tokens=metadata["usage"]["totalTokens"],
                ),
            )

        elif "contentBlockStop" in chunk:
            if self._text:
                chat_chunk = llm.ChatChunk(
                    id=request_id,
                    delta=llm.ChoiceDelta(content=self._text, role="assistant"),
                )
                self._text = ""
                return chat_chunk
            elif self._tool_call_id:
                if self._tool_call_id is None:
                    logger.warning("aws bedrock llm: no tool call id in the response")
                    return None
                if self._fnc_name is None:
                    logger.warning("aws bedrock llm: no function name in the response")
                    return None
                if self._fnc_raw_arguments is None:
                    logger.warning("aws bedrock llm: no function arguments in the response")
                    return None
                chat_chunk = llm.ChatChunk(
                    id=request_id,
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            FunctionToolCall(
                                arguments=self._fnc_raw_arguments,
                                name=self._fnc_name,
                                call_id=self._tool_call_id,
                            ),
                        ],
                    ),
                )
                self._tool_call_id = self._fnc_name = self._fnc_raw_arguments = None
                return chat_chunk
        return None
