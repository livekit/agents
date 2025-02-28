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
from typing import Any, Literal, MutableSet, Union

import boto3
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    llm,
)
from livekit.agents.llm import LLMCapabilities, ToolChoice, _create_ai_function_info
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from ._utils import _build_aws_ctx, _build_tools, _get_aws_credentials
from .log import logger

TEXT_MODEL = Literal["anthropic.claude-3-5-sonnet-20241022-v2:0"]
DEFAULT_REGION = "us-east-1"


@dataclass
class LLMOptions:
    model: TEXT_MODEL | str
    temperature: float | None
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto"
    max_output_tokens: int | None = None
    top_p: float | None = None
    additional_request_fields: dict[str, Any] | None = None


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: TEXT_MODEL | str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        api_key: str | None = None,
        api_secret: str | None = None,
        region: str = "us-east-1",
        temperature: float = 0.8,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
        additional_request_fields: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a new instance of AWS Bedrock LLM.

        ``api_key``  and ``api_secret`` must be set to your AWS Access key id and secret access key, either using the argument or by setting the
        ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY`` environmental variables.

        See https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse_stream.html for more details on the the AWS Bedrock Runtime API.

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
        """
        super().__init__(
            capabilities=LLMCapabilities(
                supports_choices_on_int=True,
                requires_persistent_functions=True,
            )
        )
        self._api_key, self._api_secret = _get_aws_credentials(
            api_key, api_secret, region
        )

        self._model = model or os.environ.get("BEDROCK_INFERENCE_PROFILE_ARN")
        if not self._model:
            raise ValueError(
                "model or inference profile arn must be set using the argument or by setting the BEDROCK_INFERENCE_PROFILE_ARN environment variable."
            )
        self._opts = LLMOptions(
            model=self._model,
            temperature=temperature,
            tool_choice=tool_choice,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            additional_request_fields=additional_request_fields,
        )
        self._region = region
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
            model=self._opts.model,
            aws_access_key_id=self._api_key,
            aws_secret_access_key=self._api_secret,
            region_name=self._region,
            max_output_tokens=self._opts.max_output_tokens,
            top_p=self._opts.top_p,
            additional_request_fields=self._opts.additional_request_fields,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            conn_options=conn_options,
            temperature=temperature,
            tool_choice=tool_choice,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        model: str | TEXT_MODEL,
        aws_access_key_id: str | None,
        aws_secret_access_key: str | None,
        region_name: str,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions,
        fnc_ctx: llm.FunctionContext | None,
        temperature: float | None,
        max_output_tokens: int | None,
        top_p: float | None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]],
        additional_request_fields: dict[str, Any] | None,
    ) -> None:
        super().__init__(
            llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx, conn_options=conn_options
        )
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        self._model = model
        self._llm: LLM = llm
        self._max_output_tokens = max_output_tokens
        self._top_p = top_p
        self._temperature = temperature
        self._tool_choice = tool_choice
        self._additional_request_fields = additional_request_fields

    async def _run(self) -> None:
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None
        self._text: str = ""
        retryable = True

        try:
            opts: dict[str, Any] = {}
            messages, system_instruction = _build_aws_ctx(self._chat_ctx, id(self))
            messages = _merge_messages(messages)

            def _get_tool_config() -> dict[str, Any] | None:
                if not (self._fnc_ctx and self._fnc_ctx.ai_functions):
                    return None

                tools = _build_tools(self._fnc_ctx)
                config: dict[str, Any] = {"tools": tools}

                if isinstance(self._tool_choice, ToolChoice):
                    config["toolChoice"] = {"tool": {"name": self._tool_choice.name}}
                elif self._tool_choice == "required":
                    config["toolChoice"] = {"any": {}}
                elif self._tool_choice == "auto":
                    config["toolChoice"] = {"auto": {}}
                else:
                    return None

                return config

            tool_config = _get_tool_config()
            if tool_config:
                opts["toolConfig"] = tool_config

            if self._additional_request_fields:
                opts["additionalModelRequestFields"] = _strip_nones(
                    self._additional_request_fields
                )

            inference_config = _strip_nones(
                {
                    "maxTokens": self._max_output_tokens,
                    "temperature": self._temperature,
                    "topP": self._top_p,
                }
            )
            response = self._client.converse_stream(
                modelId=self._model,
                messages=messages,
                system=[system_instruction] if system_instruction else None,
                inferenceConfig=inference_config,
                **_strip_nones(opts),
            )  # type: ignore

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
        elif "contentBlockStop" in chunk:
            if self._text:
                chat_chunk = llm.ChatChunk(
                    request_id=request_id,
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(content=self._text, role="assistant"),
                            index=chunk["contentBlockStop"]["contentBlockIndex"],
                        )
                    ],
                )
                self._text = ""
                return chat_chunk
            elif self._tool_call_id:
                return self._try_build_function(request_id, chunk)

        return None

    def _try_build_function(self, request_id: str, chunk: dict) -> llm.ChatChunk | None:
        if self._tool_call_id is None:
            logger.warning("aws bedrock llm: no tool call id in the response")
            return None
        if self._fnc_name is None:
            logger.warning("aws bedrock llm: no function name in the response")
            return None
        if self._fnc_raw_arguments is None:
            logger.warning("aws bedrock llm: no function arguments in the response")
            return None
        if self._fnc_ctx is None:
            logger.warning(
                "aws bedrock llm: stream tried to run function without function context"
            )
            return None

        fnc_info = _create_ai_function_info(
            self._fnc_ctx,
            self._tool_call_id,
            self._fnc_name,
            self._fnc_raw_arguments,
        )

        self._tool_call_id = self._fnc_name = self._fnc_raw_arguments = None
        self._function_calls_info.append(fnc_info)

        return llm.ChatChunk(
            request_id=request_id,
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        tool_calls=[fnc_info],
                    ),
                    index=chunk["contentBlockStop"]["contentBlockIndex"],
                )
            ],
        )


def _merge_messages(
    messages: list[dict],
) -> list[dict]:
    # Anthropic enforces alternating messages
    combined_messages: list[dict] = []
    for m in messages:
        if len(combined_messages) == 0 or m["role"] != combined_messages[-1]["role"]:
            combined_messages.append(m)
            continue
        last_message = combined_messages[-1]
        if not isinstance(last_message["content"], list) or not isinstance(
            m["content"], list
        ):
            logger.error("message content is not a list")
            continue

        last_message["content"].extend(m["content"])

    if len(combined_messages) == 0 or combined_messages[0]["role"] != "user":
        combined_messages.insert(0, {"role": "user", "content": [{"text": "(empty)"}]})

    return combined_messages


def _strip_nones(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}
