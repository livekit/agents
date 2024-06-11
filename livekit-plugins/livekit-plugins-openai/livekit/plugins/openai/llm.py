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
from typing import Any, Dict, List, MutableSet, Tuple

from attrs import define
from livekit import rtc
from livekit.agents import llm
from livekit.agents.utils import images

import openai

from .log import logger
from .models import ChatModels
from .utils import get_base_url

IMAGE_DETAIL_DIMENSIONS: List[Tuple[int, str]] = [
    (512, "low"),
    (768, "medium"),
    (2048, "high"),
]


@define
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
        history: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
    ) -> "LLMStream":
        opts = dict()
        if fnc_ctx:
            opts["tools"] = to_openai_tools(fnc_ctx)

        messages = self._to_openai_ctx(history)
        cmp = await self._client.chat.completions.create(
            messages=messages,
            model=self._opts.model,
            n=n,
            temperature=temperature,
            stream=True,
            **opts,
        )

        return LLMStream(cmp, fnc_ctx)

    def _to_openai_ctx(self, history: llm.ChatContext):
        res = []

        for msg in history.messages:
            content: List[Dict[str, Any]] = [{"type": "text", "text": msg.text}]
            for img in msg.images:
                if isinstance(img.image, str):
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img.image,
                                "detail": image_detail_from_dimensions(
                                    img.inference_width or 512,
                                    img.inference_height or 512,
                                ),
                            },
                        }
                    )
                elif isinstance(img.image, rtc.VideoFrame):
                    if self not in img._cache:
                        w, h = img.inference_width, img.inference_height
                        encode_options = images.EncodeOptions(
                            format="JPEG",
                            resize_options=images.ResizeOptions(
                                width=w or 128,
                                height=h or 128,
                                strategy="center_aspect_fit",
                            ),
                        )
                        jpg_bytes = images.encode(img.image, encode_options)
                        b64 = base64.b64encode(jpg_bytes).decode("utf-8")
                        img._cache[self] = b64

                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img._cache[self]}"
                            },
                        }
                    )
                else:
                    logger.error(f"unknown image type {type(img)}")
                    continue

            res.append(
                {
                    "role": msg.role.value,
                    "content": content,
                }
            )

        return res


class LLMStream(llm.LLMStream):
    def __init__(
        self, oai_stream: openai.AsyncStream, fnc_ctx: llm.FunctionContext | None
    ) -> None:
        super().__init__()
        self._oai_stream = oai_stream
        self._fnc_ctx = fnc_ctx
        self._running_fncs: MutableSet[asyncio.Task] = set()

    def __aiter__(self) -> "LLMStream":
        return self

    async def __anext__(self) -> llm.ChatChunk:
        fnc_name = None
        fnc_args = None
        fnc_idx = None

        async for chunk in self._oai_stream:
            for i, choice in enumerate(chunk.choices):
                delta = choice.delta

                if delta.tool_calls:
                    for tool in delta.tool_calls:
                        finfo = tool.function
                        assert finfo is not None

                        if tool.index != fnc_idx and fnc_idx is not None:
                            await self._call_function(fnc_name, fnc_args)
                            fnc_name = fnc_args = fnc_idx = None

                        if finfo.name:
                            if fnc_idx is not None:
                                logger.warning(
                                    "new fnc call while previous call is still running"
                                )
                            fnc_name = finfo.name
                            fnc_args = finfo.arguments
                            fnc_idx = tool.index
                        else:
                            assert fnc_name is not None
                            assert fnc_args is not None
                            assert fnc_idx is not None

                            if finfo.arguments:
                                fnc_args += finfo.arguments
                    continue

                if choice.finish_reason == "tool_calls":
                    await self._call_function(fnc_name, fnc_args)
                    continue

                return llm.ChatChunk(
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(
                                content=delta.content,
                                role=delta.role,
                            ),
                            index=i,
                        )
                    ]
                )

        raise StopAsyncIteration

    async def _call_function(
        self,
        name: str | None = None,
        arguments: str | None = None,
    ) -> None:
        assert self._fnc_ctx

        if name is None:
            logger.error("received tool call but no function name")
            return

        fncs = self._fnc_ctx.ai_functions
        if name not in fncs:
            logger.warning(f"function {name} not found in function context")
            return

        if arguments is None:
            logger.warning(f"received tool call but no arguments for function {name}")
            return

        args = {}
        if arguments:
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                # TODO(theomonnom): Try to recover from invalid json
                logger.exception(f"failed to decode arguments for tool call {name}")
                return

        fnc = fncs[name]
        # validate args before calling fnc
        for arg in fnc.args.values():
            if arg.name not in args:
                if arg.default is inspect.Parameter.empty:
                    logger.error(
                        f"missing required arg {arg.name} for ai_callable {name}"
                    )
                    return
                continue

            if arg.type is bool and args[arg.name] not in (True, False):
                logger.error(f"invalid arg {arg.name} for ai_callable {name}")
                return

            if arg.type is int and not isinstance(args[arg.name], int):
                logger.error(f"invalid arg {arg.name} for ai_callable {name}")
                return

            if arg.type is float and not isinstance(args[arg.name], float):
                logger.error(f"invalid arg {arg.name} for ai_callable {name}")
                return

            if arg.type is str and not isinstance(args[arg.name], str):
                logger.error(f"invalid arg {arg.name} for ai_callable {name}")
                return

            if issubclass(arg.type, enum.Enum):
                values = set(item.value for item in arg.type)
                if args[arg.name] not in values:
                    logger.error(f"invalid arg {arg.name} for ai_callable {name}")
                    return

        logger.debug(f"calling function {name} with arguments {args}")
        self._called_functions.append(
            llm.CalledFunction(fnc_name=name, fnc=fnc.fnc, args=args)
        )
        func = functools.partial(fnc.fnc, **args)
        if asyncio.iscoroutinefunction(fnc.fnc):
            task = asyncio.create_task(func())
        else:
            task = asyncio.create_task(asyncio.to_thread(func))

        def _task_done(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logger.error("ai_callable task failed", exc_info=task.exception())
            self._running_fncs.discard(task)

        task.add_done_callback(_task_done)
        self._running_fncs.add(task)

    async def aclose(self, wait: bool = True) -> None:
        if not wait:
            for task in self._running_fncs:
                task.cancel()

        await asyncio.gather(*self._running_fncs, return_exceptions=True)


def image_detail_from_dimensions(width: int, height: int) -> str:
    deltas = [
        (min(abs(width - d[0]), abs(height - d[0])), d[1])
        for d in IMAGE_DETAIL_DIMENSIONS
    ]
    smallest_delta = min(deltas, key=lambda d: d[0])
    return smallest_delta[1]


def to_openai_tools(fnc_ctx: llm.FunctionContext):
    tools = []
    for fnc in fnc_ctx.ai_functions.values():
        plist = {}
        required = []
        for arg_name, arg in fnc.args.items():
            p: Dict[str, Any] = {}
            if arg.desc:
                p["description"] = arg.desc

            if arg.type is str:
                p["type"] = "string"
            elif arg.type is int:
                p["type"] = "int"
            elif arg.type is float:
                p["type"] = "float"
            elif arg.type is bool:
                p["type"] = "boolean"
            elif issubclass(arg.type, enum.Enum):
                p["type"] = "string"
                p["enum"] = [e.value for e in arg.type]

            plist[arg_name] = p

            if arg.default is inspect.Parameter.empty:
                required.append(arg_name)

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": fnc.metadata.name,
                    "description": fnc.metadata.desc,
                    "parameters": {
                        "type": "object",
                        "properties": plist,
                        "required": required,
                    },
                },
            }
        )

    return tools
