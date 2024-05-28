from __future__ import annotations

import asyncio
import enum
import functools
import inspect
import json
import typing
from typing import Any, Dict, MutableSet

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
            opts["tools"] = to_openai_tools(fnc_ctx)

        cmp = await self._client.chat.completions.create(
            messages=to_openai_ctx(chat_ctx),
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
        fnc_info = fncs[name]
        for arg_info in fnc_info.args.values():
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
                        logger.error(
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
        self._called_functions.append(
            llm.CalledFunction(fnc_name=name, fnc=fnc_info.fnc, args=args)
        )
        func = functools.partial(fnc_info.fnc, **args)
        if asyncio.iscoroutinefunction(fnc_info.fnc):
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
        await self._oai_stream.close()

        if not wait:
            for task in self._running_fncs:
                task.cancel()

        await asyncio.gather(*self._running_fncs, return_exceptions=True)


def to_openai_ctx(chat_ctx: llm.ChatContext) -> list:
    return [
        {
            "role": msg.role.value,
            "content": msg.text,
        }
        for msg in chat_ctx.messages
    ]


def to_openai_tools(fnc_ctx: llm.FunctionContext):
    tools = []
    for fnc in fnc_ctx.ai_functions.values():
        plist = {}
        required = []
        for arg_name, arg in fnc.args.items():
            p: Dict[str, Any] = {}
            if arg.desc:
                p["description"] = arg.desc

            if typing.get_origin(arg.type) is list:
                in_type = typing.get_args(arg.type)[0]  # list type
                p["type"] = "array"

                items = {}
                if in_type is str:
                    items["type"] = "string"
                elif in_type is int or in_type is float:
                    items["type"] = "number"
                else:
                    raise ValueError(f"unsupported in_type {in_type}")

                if arg.choices:
                    items["enum"] = arg.choices

                p["items"] = items

            else:
                if arg.type is str:
                    p["type"] = "string"
                elif arg.type is int or arg.type is float:
                    p["type"] = "number"
                elif arg.type is bool:
                    p["type"] = "boolean"
                elif issubclass(arg.type, enum.Enum):
                    p["type"] = "string"
                    p["enum"] = [e.value for e in arg.type]
                else:
                    raise ValueError(f"unsupported type {arg.type}")

                if arg.choices:
                    p["enum"] = arg.choices

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
                raise ValueError(f"expected int, got float")

            return int(value)

        return float(value)

    if ty is bool:
        if not isinstance(value, bool):
            raise ValueError(f"expected bool, got {type(value)}")
        return value

    raise ValueError(f"unsupported type {ty}, not a primitive")
