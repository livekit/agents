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
import base64
import enum
import functools
import inspect
import json
import typing
from dataclasses import dataclass
from typing import Any, MutableSet

from livekit import rtc
from livekit.agents import llm
from livekit.agents.llm import function_context
from livekit.agents.utils import images

import openai
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from .log import logger
from .models import ChatModels
from .utils import get_base_url

_IMAGE_DETAIL_DIMENSIONS: list[tuple[int, str]] = [
    (512, "low"),
    (768, "medium"),
    (2048, "high"),
]


@dataclass
class LLMOptions:
    model: str | ChatModels


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "gpt-4o",
        base_url: str | None = None,
        client: openai.AsyncClient | None = None,
    ) -> None:
        self._opts = LLMOptions(model=model)
        self._client = client or openai.AsyncClient(base_url=get_base_url(base_url))
        self._running_fncs: MutableSet[asyncio.Task] = set()

    async def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
    ) -> "LLMStream":
        opts = dict()
        if fnc_ctx:
            opts["tools"] = _to_openai_tools(fnc_ctx)

        cache_key = id(self)
        messages = _to_openai_ctx(chat_ctx, cache_key)
        cmp = await self._client.chat.completions.create(
            messages=messages,
            model=self._opts.model,
            n=n,
            temperature=temperature,
            stream=True,
            **opts,
        )

        return LLMStream(cmp, fnc_ctx)


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        oai_stream: openai.AsyncStream[ChatCompletionChunk],
        fnc_ctx: llm.FunctionContext | None,
    ) -> None:
        super().__init__()
        self._oai_stream = oai_stream
        self._fnc_ctx = fnc_ctx
        self._running_tasks: MutableSet[asyncio.Task] = set()

    async def gather_function_results(self) -> list[llm.CalledFunction]:
        await asyncio.gather(*self._running_tasks, return_exceptions=True)
        return self._called_functions

    async def aclose(self) -> None:
        await self._oai_stream.close()

        for task in self._running_tasks:
            task.cancel()

        await asyncio.gather(*self._running_tasks, return_exceptions=True)

    async def __anext__(self) -> llm.ChatChunk:
        fnc_name, fnc_args, fnc_id = None, None, None
        tool_idx = None

        async for chunk in self._oai_stream:
            for i, choice in enumerate(chunk.choices):
                delta = choice.delta
                if delta.tool_calls:
                    for tool in delta.tool_calls:
                        assert tool.function is not None

                        if tool.index != tool_idx and tool_idx is not None:
                            tool_chunk = await self._call_function("assistant", i, fnc_id, fnc_name, fnc_args)
                            if tool_chunk is not None:
                                return tool_chunk

                            fnc_name = fnc_args = fnc_idx = None

                        if tool.function.name:
                            if tool_idx is not None:
                                logger.warning(
                                    "new fnc call while previous call is still running"
                                )

                            fnc_name = tool.function.name
                            fnc_args = tool.function.arguments
                            fnc_id = tool.id
                        else:
                            if tool.function.arguments:
                                fnc_args += tool.function.arguments # type: ignore

                    continue

                if choice.finish_reason == "tool_calls":
                    tool_chunk = await self._call_function("assistant", i, fnc_id, fnc_name, fnc_args)
                    if tool_chunk is not None:
                        return tool_chunk

                    continue

                return llm.ChatChunk(
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(
                                content=delta.content,
                                role="assistant",
                            ),
                            index=i,
                        )
                    ]
                )

        raise StopAsyncIteration

    async def _call_function(
        self,
        role: llm.ChatRole,
        choice_idx: int,
        id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
    ) -> llm.ChatChunk | None:
        assert self._fnc_ctx is not None

        if id is None:
            logger.error("received tool call but no function id")
            return

        if name is None:
            logger.error("received tool call but no function name")
            return

        fncs = self._fnc_ctx.ai_functions
        if name not in fncs:
            logger.warning(f"ai_function {name} not found, ignoring..")
            return

        if arguments is None:
            logger.warning(f"{name} no arguments, ignoring..")
            return

        args = {}
        if arguments:
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                logger.exception(
                    f"{name}: failed to decode json", extra={"arguments": arguments}
                )
                return

        fnc_info = fncs[name]
        for arg_info in fnc_info.arguments.values():
            if arg_info.name not in args:
                if arg_info.default is inspect.Parameter.empty:
                    logger.error(f"{name}: missing required arg {arg_info.name}")
                    return
                continue

            val = args[arg_info.name]
            try:
                val = _sanitize_primitive(arg_info.type, val)
            except ValueError:
                logger.exception(
                    f"{name}: invalid arg {arg_info.name}", extra={"value": val}
                )
                return

            if typing.get_origin(arg_info.type) is list:
                if not isinstance(val, list):
                    logger.error(
                        f"{name}: invalid arg {arg_info.name}", extra={"value": val}
                    )
                    return

                # validate all list items
                in_type = typing.get_args(arg_info.type)[0]
                for i in range(len(val)):
                    try:
                        val[i] = _sanitize_primitive(in_type, val[i])
                    except ValueError:
                        logger.exception(
                            f"{name}: invalid arg {arg_info.name}",
                            extra={"value": val},
                        )
                        return

            if issubclass(arg_info.type, enum.Enum):
                enum_values = set(item.value for item in arg_info.type)
                if val not in enum_values:
                    logger.error(
                        f"{name}: invalid arg {arg_info.name}", extra={"value": val}
                    )
                    return

            if arg_info.choices is not None:
                if val not in arg_info.choices:
                    logger.error(
                        f"{name}: invalid arg {arg_info.name}", extra={"value": val}
                    )
                    return

            args[arg_info.name] = val  # sanitized value

        logger.info(f"calling function {name} with arguments {args}")

        func = functools.partial(fnc_info.callable, **args)
        if asyncio.iscoroutinefunction(fnc_info.callable):
            task = asyncio.create_task(func())
        else:
            task = asyncio.create_task(asyncio.to_thread(func))

        called_fnc = llm.CalledFunction(
            id=id,
            raw_arguments=arguments,
            function_info=fnc_info,
            arguments=args,
            task=task,
        )
        self._called_functions.append(called_fnc)

        def _task_done(task: asyncio.Task) -> None:
            if not task.cancelled():
                if task.exception():
                    logger.error("ai_callable task failed", exc_info=task.exception())
                    called_fnc.exception = task.exception()
                else:
                    called_fnc.result = task.result()

            self._running_tasks.discard(task)

        task.add_done_callback(_task_done)
        self._running_tasks.add(task)

        return llm.ChatChunk(
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(
                        role=role,
                        tool_calls=[called_fnc]
                    ),
                    index=choice_idx,
                )
            ]
        )


def _image_detail_from_dimensions(width: int, height: int) -> str:
    for dim, detail in _IMAGE_DETAIL_DIMENSIONS:
        if width <= dim and height <= dim:
            return detail

    return "high"


def _to_openai_ctx(chat_ctx: llm.ChatContext, cache_key: Any):
    oai_context = []

    for msg in chat_ctx.messages:
        content = []
        if isinstance(msg.content, list):
            for cnt in msg.content:
                if isinstance(cnt, str):
                    content.append(_to_openai_text_content(cnt))
                elif isinstance(cnt, llm.ChatImage):
                    oai_image = _to_openai_image(cnt, cache_key)
                    if oai_image is not None:
                        content.append(oai_image)
        elif isinstance(msg.content, str):
            content.append(_to_openai_text_content(msg.content))

        tool_calls = []
        if msg.tool_calls is not None:
            for tool in msg.tool_calls:
                tool_calls.append(_to_openai_tool_calls(tool))
                pass

        oai_context.append(
            {
                "role": msg.role,
                "content": content,
                "tool_calls": tool_calls or None,
            }
        )

    return oai_context


def _to_openai_tool_calls(called_function: llm.CalledFunction):
    return {
        "id": called_function.id,
        "type": "function",
        "function": {
            "name": called_function.function_info.name,
            "arguments": called_function.raw_arguments,
        }
    }

def _to_openai_text_content(text: str):
    return {
        "type": "text",
        "content": text,
    }


def _to_openai_image(image: llm.ChatImage, cache_key: Any):
    if isinstance(image.image, str):  # image url
        return {
            "type": "image_url",
            "image_url": {
                "url": image.image,
                "detail": _image_detail_from_dimensions(
                    image.inference_width or 512,
                    image.inference_height or 512,
                ),
            },
        }
    elif isinstance(image.image, rtc.VideoFrame):  # VideoFrame
        if cache_key not in image._cache:
            encoded_data = images.encode(
                image.image,
                images.EncodeOptions(
                    resize_options=images.ResizeOptions(
                        width=image.inference_width or 512,
                        height=image.inference_height or 512,
                        strategy="center_aspect_fit",
                    ),
                ),
            )
            image._cache[cache_key] = base64.b64encode(encoded_data).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image._cache[cache_key]}"},
        }

    logger.warning(f"unknown image type {type(image.image)}")
    return None


def _to_openai_items(dst: dict, ty: type, choices: list | None):
    if ty is str:
        dst["type"] = "string"
    elif ty in (int, float):
        dst["type"] = "number"
    elif ty is bool:
        dst["type"] = "boolean"
    elif issubclass(ty, enum.Enum):
        dst["type"] = "string"
        dst["enum"] = [e.value for e in ty]
    else:
        raise ValueError(f"unsupported type {ty}")

    if choices:
        dst["enum"] = choices


def _to_openai_tools(fnc_ctx: llm.FunctionContext):
    tools_desc = []
    for fnc_info in fnc_ctx.ai_functions.values():
        properties = {}
        required_properties = []

        # build the properties for the function
        for arg_info in fnc_info.arguments.values():
            if arg_info.default is inspect.Parameter.empty:
                # property is required when there is no default value
                required_properties.append(arg_info.name)

            p = {}
            if arg_info.description:
                p["description"] = arg_info.description

            if typing.get_origin(arg_info.type) is list:
                in_type = typing.get_args(arg_info.type)[0]
                items = {}
                _to_openai_items(items, in_type, arg_info.choices)
                p["type"] = "array"
                p["items"] = items
            else:
                _to_openai_items(p, arg_info.type, arg_info.choices)

            properties[arg_info.name] = p

        tools_desc.append(
            {
                "type": "function",
                "function": {
                    "name": fnc_info.name,
                    "description": fnc_info.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required_properties,
                    },
                },
            }
        )

    return tools_desc


def _sanitize_primitive(ty: type, value: Any) -> Any:
    if ty is str:
        if not isinstance(value, str):
            raise ValueError(f"expected str, got {type(value)}")
        return value

    if ty in (int, float):
        if not isinstance(value, (int, float)):
            raise ValueError(f"expected number, got {type(value)}")

        if ty is int:
            if value % 1 != 0:
                raise ValueError("expected int, got float")

            return int(value)

        return float(value)

    if ty is bool:
        if not isinstance(value, bool):
            raise ValueError(f"expected bool, got {type(value)}")
        return value

    raise ValueError(f"unsupported type {ty}, not a primitive")
