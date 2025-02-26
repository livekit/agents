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

import base64
import inspect
import json
import os
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    List,
    Literal,
    Union,
    cast,
    get_args,
    get_origin,
)

import httpx
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    llm,
    utils,
)
from livekit.agents.llm import LLMCapabilities, ToolChoice
from livekit.agents.llm.function_context import (
    _create_ai_function_info,
    _is_optional_type,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

import anthropic

from .log import logger
from .models import (
    ChatModels,
)

CACHE_CONTROL_EPHEMERAL = anthropic.types.CacheControlEphemeralParam(type="ephemeral")


@dataclass
class LLMOptions:
    model: str | ChatModels
    user: str | None
    temperature: float | None
    parallel_tool_calls: bool | None
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] | None
    caching: Literal["ephemeral"] | None = None
    """If set to "ephemeral", the system prompt, tools, and chat history will be cached."""


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        base_url: str | None = None,
        user: str | None = None,
        client: anthropic.AsyncClient | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
        caching: Literal["ephemeral"] | None = None,
    ) -> None:
        """
        Create a new instance of Anthropic LLM.

        ``api_key`` must be set to your Anthropic API key, either using the argument or by setting
        the ``ANTHROPIC_API_KEY`` environmental variable.

        model (str | ChatModels): The model to use. Defaults to "claude-3-5-sonnet-20241022".
        api_key (str | None): The Anthropic API key. Defaults to the ANTHROPIC_API_KEY environment variable.
        base_url (str | None): The base URL for the Anthropic API. Defaults to None.
        user (str | None): The user for the Anthropic API. Defaults to None.
        client (anthropic.AsyncClient | None): The Anthropic client to use. Defaults to None.
        temperature (float | None): The temperature for the Anthropic API. Defaults to None.
        parallel_tool_calls (bool | None): Whether to parallelize tool calls. Defaults to None.
        tool_choice (Union[ToolChoice, Literal["auto", "required", "none"]] | None): The tool choice for the Anthropic API. Defaults to "auto".
        caching (Literal["ephemeral"] | None): If set to "ephemeral", caching will be enabled for the system prompt, tools, and chat history.
        """

        super().__init__(
            capabilities=LLMCapabilities(
                requires_persistent_functions=True,
                supports_choices_on_int=True,
            )
        )

        # throw an error on our end
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError("Anthropic API key is required")

        self._opts = LLMOptions(
            model=model,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            caching=caching,
        )
        self._client = client or anthropic.AsyncClient(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(
                timeout=5.0,
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=100,
                    keepalive_expiry=120,
                ),
            ),
        )

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
        if temperature is None:
            temperature = self._opts.temperature
        if parallel_tool_calls is None:
            parallel_tool_calls = self._opts.parallel_tool_calls
        if tool_choice is None:
            tool_choice = self._opts.tool_choice

        opts: dict[str, Any] = dict()
        if fnc_ctx and len(fnc_ctx.ai_functions) > 0:
            fncs_desc: list[anthropic.types.ToolParam] = []
            for i, fnc in enumerate(fnc_ctx.ai_functions.values()):
                # caching last tool will cache all the tools if caching is enabled
                cache_ctrl = (
                    CACHE_CONTROL_EPHEMERAL
                    if (i == len(fnc_ctx.ai_functions) - 1)
                    and self._opts.caching == "ephemeral"
                    else None
                )
                fncs_desc.append(
                    _build_function_description(
                        fnc,
                        cache_ctrl=cache_ctrl,
                    )
                )

            opts["tools"] = fncs_desc
            if tool_choice is not None:
                anthropic_tool_choice: dict[str, Any] | None = {"type": "auto"}
                if isinstance(tool_choice, ToolChoice):
                    if tool_choice.type == "function":
                        anthropic_tool_choice = {
                            "type": "tool",
                            "name": tool_choice.name,
                        }
                elif isinstance(tool_choice, str):
                    if tool_choice == "required":
                        anthropic_tool_choice = {"type": "any"}
                    elif tool_choice == "none":
                        opts["tools"] = []
                        anthropic_tool_choice = None
            if parallel_tool_calls is not None and parallel_tool_calls is False:
                anthropic_tool_choice["disable_parallel_tool_use"] = True
            if anthropic_tool_choice is not None:
                opts["tool_choice"] = anthropic_tool_choice

        latest_system_message: anthropic.types.TextBlockParam = _latest_system_message(
            chat_ctx, caching=self._opts.caching
        )
        anthropic_ctx = _build_anthropic_context(
            chat_ctx.messages,
            id(self),
            caching=self._opts.caching,
        )
        collaped_anthropic_ctx = _merge_messages(anthropic_ctx)

        stream = self._client.messages.create(
            max_tokens=opts.get("max_tokens", 1024),
            system=[latest_system_message],
            messages=collaped_anthropic_ctx,
            model=self._opts.model,
            temperature=temperature or anthropic.NOT_GIVEN,
            top_k=n or anthropic.NOT_GIVEN,
            stream=True,
            **opts,
        )

        return LLMStream(
            self,
            anthropic_stream=stream,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            conn_options=conn_options,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        anthropic_stream: Awaitable[
            anthropic.AsyncStream[anthropic.types.RawMessageStreamEvent]
        ],
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx, conn_options=conn_options
        )
        self._awaitable_anthropic_stream = anthropic_stream
        self._anthropic_stream: (
            anthropic.AsyncStream[anthropic.types.RawMessageStreamEvent] | None
        ) = None

        # current function call that we're waiting for full completion (args are streamed)
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None

        self._request_id: str = ""
        self._ignoring_cot = False  # ignore chain of thought
        self._input_tokens = 0
        self._cache_creation_tokens = 0
        self._cache_read_tokens = 0
        self._output_tokens = 0

    async def _run(self) -> None:
        retryable = True
        try:
            if not self._anthropic_stream:
                self._anthropic_stream = await self._awaitable_anthropic_stream

            async with self._anthropic_stream as stream:
                async for event in stream:
                    chat_chunk = self._parse_event(event)
                    if chat_chunk is not None:
                        self._event_ch.send_nowait(chat_chunk)
                        retryable = False

                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        request_id=self._request_id,
                        usage=llm.CompletionUsage(
                            completion_tokens=self._output_tokens,
                            prompt_tokens=self._input_tokens,
                            total_tokens=self._input_tokens
                            + self._output_tokens
                            + self._cache_creation_tokens
                            + self._cache_read_tokens,
                            cache_creation_input_tokens=self._cache_creation_tokens,
                            cache_read_input_tokens=self._cache_read_tokens,
                        ),
                    )
                )
        except anthropic.APITimeoutError:
            raise APITimeoutError(retryable=retryable)
        except anthropic.APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
            )
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e

    def _parse_event(
        self, event: anthropic.types.RawMessageStreamEvent
    ) -> llm.ChatChunk | None:
        if event.type == "message_start":
            self._request_id = event.message.id
            self._input_tokens = event.message.usage.input_tokens
            self._output_tokens = event.message.usage.output_tokens
            if event.message.usage.cache_creation_input_tokens:
                self._cache_creation_tokens = (
                    event.message.usage.cache_creation_input_tokens
                )
            if event.message.usage.cache_read_input_tokens:
                self._cache_read_tokens = event.message.usage.cache_read_input_tokens
        elif event.type == "message_delta":
            self._output_tokens += event.usage.output_tokens
        elif event.type == "content_block_start":
            if event.content_block.type == "tool_use":
                self._tool_call_id = event.content_block.id
                self._fnc_name = event.content_block.name
                self._fnc_raw_arguments = ""
        elif event.type == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                text = delta.text

                if self._fnc_ctx is not None:
                    # anthropic may inject COC when using functions
                    if text.startswith("<thinking>"):
                        self._ignoring_cot = True
                    elif self._ignoring_cot and "</thinking>" in text:
                        text = text.split("</thinking>")[-1]
                        self._ignoring_cot = False

                if self._ignoring_cot:
                    return None

                return llm.ChatChunk(
                    request_id=self._request_id,
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(content=text, role="assistant")
                        )
                    ],
                )
            elif delta.type == "input_json_delta":
                assert self._fnc_raw_arguments is not None
                self._fnc_raw_arguments += delta.partial_json

        elif event.type == "content_block_stop":
            if self._tool_call_id is not None and self._fnc_ctx:
                assert self._fnc_name is not None
                assert self._fnc_raw_arguments is not None

                fnc_info = _create_ai_function_info(
                    self._fnc_ctx,
                    self._tool_call_id,
                    self._fnc_name,
                    self._fnc_raw_arguments,
                )
                self._function_calls_info.append(fnc_info)

                chat_chunk = llm.ChatChunk(
                    request_id=self._request_id,
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(
                                role="assistant", tool_calls=[fnc_info]
                            ),
                        )
                    ],
                )
                self._tool_call_id = self._fnc_raw_arguments = self._fnc_name = None
                return chat_chunk

        return None


def _latest_system_message(
    chat_ctx: llm.ChatContext, caching: Literal["ephemeral"] | None = None
) -> anthropic.types.TextBlockParam:
    latest_system_message: llm.ChatMessage | None = None
    for m in chat_ctx.messages:
        if m.role == "system":
            latest_system_message = m
            continue

    latest_system_str = ""
    if latest_system_message:
        if isinstance(latest_system_message.content, str):
            latest_system_str = latest_system_message.content
        elif isinstance(latest_system_message.content, list):
            latest_system_str = " ".join(
                [c for c in latest_system_message.content if isinstance(c, str)]
            )
    system_text_block = anthropic.types.TextBlockParam(
        text=latest_system_str,
        type="text",
        cache_control=CACHE_CONTROL_EPHEMERAL if caching == "ephemeral" else None,
    )
    return system_text_block


def _merge_messages(
    messages: List[anthropic.types.MessageParam],
) -> List[anthropic.types.MessageParam]:
    # Anthropic enforces alternating messages
    combined_messages: list[anthropic.types.MessageParam] = []
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
        combined_messages.insert(
            0, {"role": "user", "content": [{"type": "text", "text": "(empty)"}]}
        )

    return combined_messages


def _build_anthropic_context(
    chat_ctx: List[llm.ChatMessage],
    cache_key: Any,
    caching: Literal["ephemeral"] | None,
) -> List[anthropic.types.MessageParam]:
    result: List[anthropic.types.MessageParam] = []
    for i, msg in enumerate(chat_ctx):
        # caching last message will cache whole chat history if caching is enabled
        cache_ctrl = (
            CACHE_CONTROL_EPHEMERAL
            if ((i == len(chat_ctx) - 1) and caching == "ephemeral")
            else None
        )
        a_msg = _build_anthropic_message(msg, cache_key, cache_ctrl=cache_ctrl)

        if a_msg:
            result.append(a_msg)
    return result


def _build_anthropic_message(
    msg: llm.ChatMessage,
    cache_key: Any,
    cache_ctrl: anthropic.types.CacheControlEphemeralParam | None,
) -> anthropic.types.MessageParam | None:
    if msg.role == "user" or msg.role == "assistant":
        a_msg: anthropic.types.MessageParam = {
            "role": msg.role,
            "content": [],
        }
        assert isinstance(a_msg["content"], list)
        a_content = a_msg["content"]

        # add content if provided
        if isinstance(msg.content, str) and msg.content:
            a_msg["content"].append(
                anthropic.types.TextBlockParam(
                    text=msg.content,
                    type="text",
                    cache_control=cache_ctrl,
                )
            )
        elif isinstance(msg.content, list):
            for cnt in msg.content:
                if isinstance(cnt, str) and cnt:
                    content: anthropic.types.TextBlockParam = (
                        anthropic.types.TextBlockParam(
                            text=cnt,
                            type="text",
                            cache_control=cache_ctrl,
                        )
                    )
                    a_content.append(content)
                elif isinstance(cnt, llm.ChatImage):
                    a_content.append(
                        _build_anthropic_image_content(cnt, cache_key, cache_ctrl)
                    )
        if msg.tool_calls is not None:
            for fnc in msg.tool_calls:
                tool_use = anthropic.types.ToolUseBlockParam(
                    id=fnc.tool_call_id,
                    type="tool_use",
                    name=fnc.function_info.name,
                    input=fnc.arguments,
                    cache_control=cache_ctrl,
                )
                a_content.append(tool_use)

        return a_msg
    elif msg.role == "tool":
        if isinstance(msg.content, dict):
            msg.content = json.dumps(msg.content)
        if not isinstance(msg.content, str):
            logger.warning("tool message content is not a string or dict")
            return None
        if not msg.tool_call_id:
            return None

        u_content = anthropic.types.ToolResultBlockParam(
            tool_use_id=msg.tool_call_id,
            type="tool_result",
            content=msg.content,
            is_error=msg.tool_exception is not None,
            cache_control=cache_ctrl,
        )
        return {
            "role": "user",
            "content": [u_content],
        }

    return None


def _build_anthropic_image_content(
    image: llm.ChatImage,
    cache_key: Any,
    cache_ctrl: anthropic.types.CacheControlEphemeralParam | None,
) -> anthropic.types.ImageBlockParam:
    if isinstance(image.image, str):  # image is a URL
        if not image.image.startswith("data:"):
            raise ValueError("LiveKit Anthropic Plugin: Image URLs must be data URLs")

        try:
            header, b64_data = image.image.split(",", 1)
            media_type = header.split(";")[0].split(":")[1]

            supported_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
            if media_type not in supported_types:
                raise ValueError(
                    f"LiveKit Anthropic Plugin: Unsupported media type {media_type}. Must be jpeg, png, webp, or gif"
                )

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "data": b64_data,
                    "media_type": cast(
                        Literal["image/jpeg", "image/png", "image/gif", "image/webp"],
                        media_type,
                    ),
                },
                "cache_control": cache_ctrl,
            }
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"LiveKit Anthropic Plugin: Invalid image data URL {str(e)}"
            )
    elif isinstance(image.image, rtc.VideoFrame):  # image is a VideoFrame
        if cache_key not in image._cache:
            # inside our internal implementation, we allow to put extra metadata to
            # each ChatImage (avoid to reencode each time we do a chatcompletion request)
            opts = utils.images.EncodeOptions()
            if image.inference_width and image.inference_height:
                opts.resize_options = utils.images.ResizeOptions(
                    width=image.inference_width,
                    height=image.inference_height,
                    strategy="scale_aspect_fit",
                )

            encoded_data = utils.images.encode(image.image, opts)
            image._cache[cache_key] = base64.b64encode(encoded_data).decode("utf-8")

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "data": image._cache[cache_key],
                "media_type": "image/jpeg",
            },
            "cache_control": cache_ctrl,
        }

    raise ValueError(
        "LiveKit Anthropic Plugin: ChatImage must be an rtc.VideoFrame or a data URL"
    )


def _build_function_description(
    fnc_info: llm.function_context.FunctionInfo,
    cache_ctrl: anthropic.types.CacheControlEphemeralParam | None,
) -> anthropic.types.ToolParam:
    def build_schema_field(arg_info: llm.function_context.FunctionArgInfo):
        def type2str(t: type) -> str:
            if t is str:
                return "string"
            elif t in (int, float):
                return "number"
            elif t is bool:
                return "boolean"

            raise ValueError(f"unsupported type {t} for ai_property")

        p: dict[str, Any] = {}
        if arg_info.default is inspect.Parameter.empty:
            p["required"] = True
        else:
            p["required"] = False

        if arg_info.description:
            p["description"] = arg_info.description

        _, inner_th = _is_optional_type(arg_info.type)

        if get_origin(inner_th) is list:
            inner_type = get_args(inner_th)[0]
            p["type"] = "array"
            p["items"] = {}
            p["items"]["type"] = type2str(inner_type)

            if arg_info.choices:
                p["items"]["enum"] = arg_info.choices
        else:
            p["type"] = type2str(inner_th)
            if arg_info.choices:
                p["enum"] = arg_info.choices

        return p

    input_schema: dict[str, object] = {"type": "object"}

    for arg_info in fnc_info.arguments.values():
        input_schema[arg_info.name] = build_schema_field(arg_info)

    return anthropic.types.ToolParam(
        name=fnc_info.name,
        description=fnc_info.description,
        input_schema=input_schema,
        cache_control=cache_ctrl,
    )
