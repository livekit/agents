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
from typing import Any, Awaitable, List, Tuple, get_args, get_origin

import httpx
from livekit import rtc
from livekit.agents import llm, utils

import anthropic

from .log import logger
from .models import (
    ChatModels,
)


@dataclass
class LLMOptions:
    model: str | ChatModels
    user: str | None
    temperature: float | None


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "claude-3-haiku-20240307",
        api_key: str | None = None,
        base_url: str | None = None,
        user: str | None = None,
        client: anthropic.AsyncClient | None = None,
        temperature: float | None = None,
    ) -> None:
        """
        Create a new instance of Anthropic LLM.

        ``api_key`` must be set to your Anthropic API key, either using the argument or by setting
        the ``ANTHROPIC_API_KEY`` environmental variable.
        """
        # throw an error on our end
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError("Anthropic API key is required")

        self._opts = LLMOptions(model=model, user=user, temperature=temperature)
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
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
        parallel_tool_calls: bool | None = None,
    ) -> "LLMStream":
        if temperature is None:
            temperature = self._opts.temperature

        opts: dict[str, Any] = dict()
        if fnc_ctx and len(fnc_ctx.ai_functions) > 0:
            fncs_desc: list[anthropic.types.ToolParam] = []
            for fnc in fnc_ctx.ai_functions.values():
                fncs_desc.append(_build_function_description(fnc))

            opts["tools"] = fncs_desc

            if fnc_ctx and parallel_tool_calls is not None:
                opts["parallel_tool_calls"] = parallel_tool_calls

        latest_system_message = _latest_system_message(chat_ctx)
        anthropic_ctx = _build_anthropic_context(chat_ctx.messages, id(self))
        collaped_anthropic_ctx = _merge_messages(anthropic_ctx)
        stream = self._client.messages.create(
            max_tokens=opts.get("max_tokens", 1000),
            system=latest_system_message,
            messages=collaped_anthropic_ctx,
            model=self._opts.model,
            temperature=temperature or anthropic.NOT_GIVEN,
            top_k=n or anthropic.NOT_GIVEN,
            stream=True,
            **opts,
        )

        return LLMStream(anthropic_stream=stream, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        *,
        anthropic_stream: Awaitable[
            anthropic.AsyncStream[anthropic.types.RawMessageStreamEvent]
        ],
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
    ) -> None:
        super().__init__(chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)
        self._awaitable_anthropic_stream = anthropic_stream
        self._anthropic_stream: (
            anthropic.AsyncStream[anthropic.types.RawMessageStreamEvent] | None
        ) = None

        # current function call that we're waiting for full completion (args are streamed)
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None

    async def aclose(self) -> None:
        if self._anthropic_stream:
            await self._anthropic_stream.close()

        return await super().aclose()

    async def __anext__(self):
        if not self._anthropic_stream:
            self._anthropic_stream = await self._awaitable_anthropic_stream

        fn_calling_enabled = self._fnc_ctx is not None
        ignore = False

        async for event in self._anthropic_stream:
            if event.type == "message_start":
                pass
            elif event.type == "message_delta":
                pass
            elif event.type == "message_stop":
                pass
            elif event.type == "content_block_start":
                if event.content_block.type == "tool_use":
                    self._tool_call_id = event.content_block.id
                    self._fnc_raw_arguments = ""
                    self._fnc_name = event.content_block.name
            elif event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    text = delta.text

                    # Anthropic seems to add a prompt when tool calling is enabled
                    # where responses always start with a "<thinking>" block containing
                    # the LLM's chain of thought. It's very verbose and not useful for voice
                    # applications.
                    if fn_calling_enabled:
                        if text.startswith("<thinking>"):
                            ignore = True

                        if "</thinking>" in text:
                            text = text.split("</thinking>")[-1]
                            ignore = False

                    if ignore:
                        continue

                    return llm.ChatChunk(
                        choices=[
                            llm.Choice(
                                delta=llm.ChoiceDelta(content=text, role="assistant")
                            )
                        ]
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
                    chunk = llm.ChatChunk(
                        choices=[
                            llm.Choice(
                                delta=llm.ChoiceDelta(
                                    role="assistant", tool_calls=[fnc_info]
                                ),
                                index=0,
                            )
                        ]
                    )
                    self._tool_call_id = None
                    self._fnc_raw_arguments = None
                    self._fnc_name = None
                    return chunk

        raise StopAsyncIteration


def _latest_system_message(chat_ctx: llm.ChatContext) -> str:
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
    return latest_system_str


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
    chat_ctx: List[llm.ChatMessage], cache_key: Any
) -> List[anthropic.types.MessageParam]:
    result: List[anthropic.types.MessageParam] = []
    for msg in chat_ctx:
        a_msg = _build_anthropic_message(msg, cache_key, chat_ctx)
        if a_msg:
            result.append(a_msg)
    return result


def _build_anthropic_message(
    msg: llm.ChatMessage, cache_key: Any, chat_ctx: List[llm.ChatMessage]
) -> anthropic.types.MessageParam | None:
    if msg.role == "user" or msg.role == "assistant":
        a_msg: anthropic.types.MessageParam = {
            "role": msg.role,
            "content": [],
        }
        assert isinstance(a_msg["content"], list)
        a_content = a_msg["content"]

        # add content if provided
        if isinstance(msg.content, str):
            a_msg["content"].append(
                anthropic.types.TextBlock(
                    text=msg.content,
                    type="text",
                )
            )
        elif isinstance(msg.content, list):
            for cnt in msg.content:
                if isinstance(cnt, str):
                    content: anthropic.types.TextBlock = anthropic.types.TextBlock(
                        text=cnt,
                        type="text",
                    )
                    a_content.append(content)
                elif isinstance(cnt, llm.ChatImage):
                    a_content.append(_build_anthropic_image_content(cnt, cache_key))

        if msg.tool_calls is not None:
            for fnc in msg.tool_calls:
                tool_use = anthropic.types.ToolUseBlockParam(
                    id=fnc.tool_call_id,
                    type="tool_use",
                    name=fnc.function_info.name,
                    input=fnc.arguments,
                )
                a_content.append(tool_use)

        return a_msg
    elif msg.role == "tool":
        if not isinstance(msg.content, str):
            logger.warning("tool message content is not a string")
            return None
        if not msg.tool_call_id:
            return None

        u_content = anthropic.types.ToolResultBlockParam(
            tool_use_id=msg.tool_call_id,
            type="tool_result",
            content=msg.content,
            is_error=msg.tool_exception is not None,
        )
        return {
            "role": "user",
            "content": [u_content],
        }

    return None


def _build_anthropic_image_content(
    image: llm.ChatImage, cache_key: Any
) -> anthropic.types.ImageBlockParam:
    if isinstance(image.image, str):  # image url
        logger.warning(
            "image url not supported by anthropic, skipping image '%s'", image.image
        )
    elif isinstance(image.image, rtc.VideoFrame):  # VideoFrame
        if cache_key not in image._cache:
            # inside our internal implementation, we allow to put extra metadata to
            # each ChatImage (avoid to reencode each time we do a chatcompletion request)
            opts = utils.images.EncodeOptions()
            if image.inference_width and image.inference_height:
                opts.resize_options = utils.images.ResizeOptions(
                    width=image.inference_width,
                    height=image.inference_height,
                    strategy="center_aspect_fit",
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
        }

    raise ValueError(f"unknown image type {type(image.image)}")


def _create_ai_function_info(
    fnc_ctx: llm.function_context.FunctionContext,
    tool_call_id: str,
    fnc_name: str,
    raw_arguments: str,  # JSON string
) -> llm.function_context.FunctionCallInfo:
    if fnc_name not in fnc_ctx.ai_functions:
        raise ValueError(f"AI function {fnc_name} not found")

    parsed_arguments: dict[str, Any] = {}
    try:
        if raw_arguments:  # ignore empty string
            parsed_arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        raise ValueError(
            f"AI function {fnc_name} received invalid JSON arguments - {raw_arguments}"
        )

    fnc_info = fnc_ctx.ai_functions[fnc_name]

    # Ensure all necessary arguments are present and of the correct type.
    sanitized_arguments: dict[str, Any] = {}
    for arg_info in fnc_info.arguments.values():
        if arg_info.name not in parsed_arguments:
            if arg_info.default is inspect.Parameter.empty:
                raise ValueError(
                    f"AI function {fnc_name} missing required argument {arg_info.name}"
                )
            continue

        arg_value = parsed_arguments[arg_info.name]
        if get_origin(arg_info.type) is not None:
            if not isinstance(arg_value, list):
                raise ValueError(
                    f"AI function {fnc_name} argument {arg_info.name} should be a list"
                )

            inner_type = get_args(arg_info.type)[0]
            sanitized_value = [
                _sanitize_primitive(
                    value=v, expected_type=inner_type, choices=arg_info.choices
                )
                for v in arg_value
            ]
        else:
            sanitized_value = _sanitize_primitive(
                value=arg_value, expected_type=arg_info.type, choices=arg_info.choices
            )

        sanitized_arguments[arg_info.name] = sanitized_value

    return llm.function_context.FunctionCallInfo(
        tool_call_id=tool_call_id,
        raw_arguments=raw_arguments,
        function_info=fnc_info,
        arguments=sanitized_arguments,
    )


def _build_function_description(
    fnc_info: llm.function_context.FunctionInfo,
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

        if get_origin(arg_info.type) is list:
            inner_type = get_args(arg_info.type)[0]
            p["type"] = "array"
            p["items"] = {}
            p["items"]["type"] = type2str(inner_type)

            if arg_info.choices:
                p["items"]["enum"] = arg_info.choices
        else:
            p["type"] = type2str(arg_info.type)
            if arg_info.choices:
                p["enum"] = arg_info.choices

        return p

    input_schema: dict[str, object] = {"type": "object"}

    for arg_info in fnc_info.arguments.values():
        input_schema[arg_info.name] = build_schema_field(arg_info)

    return {
        "name": fnc_info.name,
        "description": fnc_info.description,
        "input_schema": input_schema,
    }


def _sanitize_primitive(
    *, value: Any, expected_type: type, choices: Tuple[Any] | None
) -> Any:
    if expected_type is str:
        if not isinstance(value, str):
            raise ValueError(f"expected str, got {type(value)}")
    elif expected_type in (int, float):
        if not isinstance(value, (int, float)):
            raise ValueError(f"expected number, got {type(value)}")

        if expected_type is int:
            if value % 1 != 0:
                raise ValueError("expected int, got float")

            value = int(value)
        elif expected_type is float:
            value = float(value)

    elif expected_type is bool:
        if not isinstance(value, bool):
            raise ValueError(f"expected bool, got {type(value)}")

    if choices and value not in choices:
        raise ValueError(f"invalid value {value}, not in {choices}")

    return value
