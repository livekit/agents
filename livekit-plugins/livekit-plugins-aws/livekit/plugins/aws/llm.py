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
from dataclasses import dataclass
from typing import Any, Literal, MutableSet, Union

import boto3
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    llm,
)
from livekit.agents.llm import ToolChoice, _create_ai_function_info
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

        See https://docs.aws.amazon.com/polly/latest/dg/API_SynthesizeSpeech.html for more details on the the AWS Polly TTS.

        Args:
            model (TEXT_MODEL, optional): The model name to use. Defaults to "anthropic.claude-3-5-sonnet-20241022-v2:0".
            api_key(str, optional): AWS access key id.
            api_secret(str, optional): AWS secret access key
            region (str, optional): The region to use for AWS API requests. Defaults value is "us-east-1".
            candidate_count (int, optional): Number of candidate responses to generate. Defaults to 1.
            temperature (float, optional): Sampling temperature for response generation. Defaults to 0.8.
            max_output_tokens (int, optional): Maximum number of tokens to generate in the output. Defaults to None.
            top_p (float, optional): The nucleus sampling probability for response generation. Defaults to None.
            top_k (int, optional): The top-k sampling value for response generation. Defaults to None.
            presence_penalty (float, optional): Penalizes the model for generating previously mentioned concepts. Defaults to None.
            frequency_penalty (float, optional): Penalizes the model for repeating words. Defaults to None.
            tool_choice (ToolChoice or Literal["auto", "required", "none"], optional): Specifies whether to use tools during response generation. Defaults to "auto".
        """
        super().__init__()
        self._capabilities = llm.LLMCapabilities(supports_choices_on_int=True)

        self._api_key, self._api_secret = _get_aws_credentials(
            api_key, api_secret, region
        )
        self._opts = LLMOptions(
            model=model,
            temperature=temperature,
            tool_choice=tool_choice,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            additional_request_fields=additional_request_fields,
        )
        self._client = boto3.client("bedrock-runtime", region_name=region)
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
        client: boto3.client,
        model: str | TEXT_MODEL,
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
        self._client = client
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
            if messages[0]["role"] != "user":
                messages.insert(
                    0,
                    {"role": "user", "content": [{"text": "(empty)"}]},
                )

            if self._fnc_ctx and self._fnc_ctx.ai_functions:
                tools = _build_tools(self._fnc_ctx)
                tool_config = {"tools": tools}

                if isinstance(self._tool_choice, ToolChoice):
                    tool_config["toolChoice"] = {
                        "tool": {"name": self._tool_choice.name}
                    }
                elif self._tool_choice == "required":
                    tool_config["toolChoice"] = {"any": {}}
                elif self._tool_choice == "auto":
                    tool_config["toolChoice"] = {"auto": {}}
                else:
                    raise ValueError("aws bedrock llm: invalid tool choice")

                opts["toolConfig"] = tool_config

            if self._additional_request_fields:
                opts["additionalModelRequestFields"] = _strip_nones(
                    self._additional_request_fields
                )

            inference_config = _strip_nones({
                "maxTokens": self._max_output_tokens,
                "temperature": self._temperature,
                "topP": self._top_p,
            })
            response = self._client.converse_stream(
                modelId=self._model,
                messages=messages,
                system=[system_instruction],
                inferenceConfig=inference_config,
                **opts,
            )

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
        missing = []
        if not self._tool_call_id:
            missing.append("tool_call_id")
        if not self._fnc_name:
            missing.append("fnc_name")
        if self._fnc_raw_arguments is None:
            missing.append("fnc_raw_arguments")
        if not self._fnc_ctx:
            missing.append("fnc_ctx")

        if missing:
            logger.warning(
                f"aws bedrock llm: missing data for function call: {missing}"
            )
            self._tool_call_id = self._fnc_name = self._fnc_raw_arguments = None
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


def _strip_nones(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}
