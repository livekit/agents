import asyncio
import enum
import functools
import inspect
import json
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
        model: str | ChatModels = "gpt-4-1106-preview",
        client: openai.AsyncClient | None = None,
    ) -> None:
        self._opts = LLMOptions(model=model)
        self._client = client or openai.AsyncClient()
        self._running_fncs: MutableSet[asyncio.Task] = set()

    async def chat(
        self,
        history: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = None,
    ) -> "LLMStream":
        tools = to_openai_tools(fnc_ctx) if fnc_ctx else openai.NOT_GIVEN
        cmp = await self._client.chat.completions.create(
            messages=to_openai_ctx(history),
            model=self._opts.model,
            n=n,
            tools=tools,
            temperature=temperature,
            stream=True,
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
            if arg.default is inspect.Parameter.empty and arg.name not in args:
                logger.error(f"missing required arg {arg.name} for ai_callable {name}")
                return

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

            if issubclass(arg.type, enum.Enum) and args[arg.name] not in arg.type:
                logger.error(f"invalid arg {arg.name} for ai_callable {name}")
                return

        logger.debug(f"calling function {name} with arguments {args}")
        func = functools.partial(fnc.fnc, **args)
        if asyncio.iscoroutinefunction(fnc.fnc):
            task = asyncio.ensure_future(func())
        else:
            task = asyncio.ensure_future(asyncio.to_thread(func))

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

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": fnc.metadata.name,
                    "description": fnc.metadata.desc,
                    "parameters": {
                        "type": "object",
                        "properties": plist,
                    },
                },
            }
        )

    return tools
