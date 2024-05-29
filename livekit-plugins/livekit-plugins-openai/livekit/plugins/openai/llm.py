from __future__ import annotations

import asyncio
import enum
import functools
import inspect
import json
import typing
from typing import Any, MutableSet

from attrs import define
from livekit.agents import llm

import openai

from .log import logger
from .models import ChatModels


@define
class LLMOptions:
    model: str | ChatModels


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "gpt-4o",
        client: openai.AsyncClient | None = None,
    ) -> None:
        self._opts = LLMOptions(model=model)
        self._client = client or openai.AsyncClient()
        self._running_fncs: MutableSet[asyncio.Task] = set()

    async def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = None,
    ) -> "LLMStream":
        opts = dict()
        if fnc_ctx:
            opts["tools"] = _to_openai_tools(fnc_ctx)

        cmp = await self._client.chat.completions.create(
            messages=_to_openai_ctx(chat_ctx),
            model=self._opts.model,
            n=n,
            temperature=temperature,
            stream=True,
            **opts,
        )

        return LLMStream(cmp, fnc_ctx)


class LLMStream(llm.LLMStream):
    def __init__(
        self, oai_stream: openai.AsyncStream, fnc_ctx: llm.FunctionContext | None
    ) -> None:
        super().__init__()
        self._oai_stream = oai_stream
        self._fnc_ctx = fnc_ctx
        self._running_tasks: MutableSet[asyncio.Task] = set()

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
                # TODO(theomonnom): try to recover from invalid json
                logger.exception(
                    f"{name}: failed to decode json", extra={"arguments": arguments}
                )
                return

        # validate/sanitize args before calling any ai function
        # TODO(theomonnom): try to recover from invalid args
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

        logger.debug(f"calling function {name} with arguments {args}")

        func = functools.partial(fnc_info.callable, **args)
        if asyncio.iscoroutinefunction(fnc_info.callable):
            task = asyncio.create_task(func())
        else:
            task = asyncio.create_task(asyncio.to_thread(func))

        self._called_functions.append(
            llm.CalledFunction(info=fnc_info, arguments=args, task=task)
        )

        def _task_done(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logger.error("ai_callable task failed", exc_info=task.exception())
            self._running_tasks.discard(task)

        task.add_done_callback(_task_done)
        self._running_tasks.add(task)

    async def gather_function_results(self) -> list[llm.CalledFunction]:
        await asyncio.gather(*self._running_tasks, return_exceptions=True)
        return self._called_functions

    async def aclose(self) -> None:
        await self._oai_stream.close()

        for task in self._running_tasks:
            task.cancel()

        await asyncio.gather(*self._running_tasks, return_exceptions=True)


def _to_openai_ctx(chat_ctx: llm.ChatContext) -> list:
    return [
        {
            "role": msg.role.value,
            "content": msg.text,
        }
        for msg in chat_ctx.messages
    ]


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
