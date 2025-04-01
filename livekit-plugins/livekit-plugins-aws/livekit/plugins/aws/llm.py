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

import os
from dataclasses import dataclass
from typing import Any, Literal

import aioboto3

from livekit.agents import APIConnectionError, APIStatusError, llm, utils
from livekit.agents.llm import ChatContext, FunctionTool, FunctionToolCall, ToolChoice
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .utils import get_aws_async_session, get_aws_credentials, to_chat_ctx, to_fnc_ctx

TEXT_MODEL = Literal["anthropic.claude-3-5-sonnet-20241022-v2:0"]
REFRESH_INTERVAL = 1800
DEFAULT_REGION = "us-east-1"


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
        session: aioboto3.Session | None = None,
        refresh_interval: NotGivenOr[int] = NOT_GIVEN,
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
            session (aioboto3.Session, optional): Optional aioboto3 session to use.
            refresh_interval (int, optional): Refresh interval for the AWS session. Defaults to 1800 seconds (30 minutes).
        """  # noqa: E501
        super().__init__()
        self._session = session
        if not self._session:
            self._region = (
                region
                or os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
                or DEFAULT_REGION
            )
            self._creds = get_aws_credentials(api_key=api_key, api_secret=api_secret)

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
        self._client_cm = None
        self._pool = utils.ConnectionPool[aioboto3.Session.client](
            connect_cb=self._create_client,
            max_session_duration=refresh_interval
            if is_given(refresh_interval)
            else REFRESH_INTERVAL,
        )

    async def _create_client(self) -> aioboto3.Session.client:
        # Exit any existing client context manager
        if self._client_cm:
            await self._client_cm.__aexit__(None, None, None)
            self._client_cm = None

        session = self._session or await get_aws_async_session(
            region=self._region,
            api_key=self._creds.access_key,
            api_secret=self._creds.secret_key,
        )
        # context manager for the client
        self._client_cm = session.client("bedrock-runtime")
        client = await self._client_cm.__aenter__()
        return client

    async def aclose(self) -> None:
        if self._client_cm:
            await self._client_cm.__aexit__()
        await self._pool.aclose()
        await super().aclose()

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
                if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                    tool_config["toolChoice"] = {"tool": {"name": tool_choice["function"]["name"]}}
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
            chat_ctx=chat_ctx,
            tools=tools,
            pool=self._pool,
            conn_options=conn_options,
            extra_kwargs=opts,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        chat_ctx: ChatContext,
        pool: utils.ConnectionPool[aioboto3.Session.client],
        conn_options: APIConnectOptions,
        tools: list[FunctionTool] | None,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._llm: LLM = llm
        self._pool = pool
        self._opts = extra_kwargs

        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None
        self._text: str = ""

    async def _run(self) -> None:
        retryable = True
        try:
            async with self._pool.connection() as client:
                response = await client.converse_stream(**self._opts)  # type: ignore
                request_id = response["ResponseMetadata"]["RequestId"]
                if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                    raise APIStatusError(
                        f"aws bedrock llm: error generating content: {response}",
                        retryable=False,
                        request_id=request_id,
                    )

                async for chunk in response["stream"]:
                    chat_chunk = self._parse_chunk(request_id, chunk)
                    if chat_chunk is not None:
                        retryable = False
                        self._event_ch.send_nowait(chat_chunk)

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
                return llm.ChatChunk(
                    id=request_id,
                    delta=llm.ChoiceDelta(content=delta["text"], role="assistant"),
                )
            else:
                logger.warning(f"aws bedrock llm: unknown chunk type: {chunk}")

        elif "metadata" in chunk:
            metadata = chunk["metadata"]
            return llm.ChatChunk(
                id=request_id,
                usage=llm.CompletionUsage(
                    completion_tokens=metadata["usage"]["outputTokens"],
                    prompt_tokens=metadata["usage"]["inputTokens"],
                    total_tokens=metadata["usage"]["totalTokens"],
                ),
            )
        elif "contentBlockStop" in chunk:
            if self._tool_call_id:
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
