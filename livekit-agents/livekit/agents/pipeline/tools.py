from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
)

from pydantic import ValidationError

from .. import debug, llm, utils
from ..llm import (
    AIError,
    FunctionContext,
    StopResponse,
)
from ..llm import (
    utils as llm_utils,
)
from ..log import logger
from .context import AgentContext
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from .agent_task import AgentTask


def perform_tool_executions(
    *,
    agent_ctx: AgentContext,
    speech_handle: SpeechHandle,
    fnc_ctx: FunctionContext,
    function_stream: AsyncIterable[llm.FunctionCall],
) -> tuple[asyncio.Task, list[tuple[llm.FunctionCallOutput | None, AgentTask | None]]]:
    out: list[tuple[llm.FunctionCallOutput | None, AgentTask | None]] = []
    task = asyncio.create_task(
        _execute_tools_task(
            agent_ctx=agent_ctx,
            speech_handle=speech_handle,
            fnc_ctx=fnc_ctx,
            function_stream=function_stream,
            out=out,
        ),
        name="execute_tools_task",
    )
    return task, out


@utils.log_exceptions(logger=logger)
async def _execute_tools_task(
    *,
    agent_ctx: AgentContext,
    speech_handle: SpeechHandle,
    fnc_ctx: FunctionContext,
    function_stream: AsyncIterable[llm.FunctionCall],
    out: list[tuple[llm.FunctionCallOutput | None, AgentTask | None]],
) -> None:
    """execute tools, when cancelled, stop executing new tools but wait for the pending ones"""
    tasks: list[asyncio.Task] = []
    try:
        async for fnc_call in function_stream:
            ai_function = fnc_ctx.ai_functions.get(fnc_call.name, None)
            if ai_function is None:
                logger.warning(
                    f"LLM called function `{fnc_call.name}` but it was not found in the current task",
                    extra={
                        "function": fnc_call.name,
                        "speech_id": speech_handle.id,
                    },
                )
                continue

            try:
                function_model = llm_utils.function_arguments_to_pydantic_model(
                    ai_function
                )
                parsed_args = function_model.model_validate_json(fnc_call.arguments)
            except ValidationError:
                logger.exception(
                    "LLM called function `{fnc.name}` with invalid arguments",
                    extra={
                        "function": fnc_call.name,
                        "arguments": fnc_call.arguments,
                        "speech_id": speech_handle.id,
                    },
                )
                continue

            logger.debug(
                "executing tool",
                extra={
                    "function": fnc_call.name,
                    "speech_id": speech_handle.id,
                },
            )
            debug.Tracing.log_event(
                "executing tool",
                {
                    "function": fnc_call.name,
                    "speech_id": speech_handle.id,
                },
            )

            fnc_args, fnc_kwargs = llm_utils.pydantic_model_to_function_arguments(
                ai_function=ai_function,
                model=parsed_args,
                agent_ctx=agent_ctx,
                speech_handle=speech_handle,
            )

            fnc_out = _FunctionCallOutput(
                name=fnc_call.name,
                arguments=fnc_call.arguments,
                call_id=fnc_call.call_id,
                output=None,
                exception=None,
            )

            if inspect.iscoroutinefunction(ai_function):
                task = asyncio.create_task(
                    ai_function(*fnc_args, **fnc_kwargs),
                    name=f"ai_function_{fnc_call.name}",
                )
                tasks.append(task)

                def _log_exceptions(task: asyncio.Task) -> None:
                    if task.exception() is not None:
                        logger.error(
                            "exception occurred while executing tool",
                            extra={
                                "function": fnc_call.name,
                                "speech_id": speech_handle.id,
                            },
                            exc_info=task.exception(),
                        )
                        fnc_out.exception = task.exception()
                        out.append(_sanitize_function_output(fnc_out))
                        return

                    fnc_out.output = task.result()
                    out.append(_sanitize_function_output(fnc_out))
                    tasks.remove(task)

                task.add_done_callback(_log_exceptions)
            else:
                start_time = time.monotonic()
                try:
                    output = ai_function(*fnc_args, **fnc_kwargs)
                    fnc_out.output = output
                    out.append(_sanitize_function_output(fnc_out))
                except Exception as e:
                    fnc_out.exception = e
                    out.append(_sanitize_function_output(fnc_out))

                elapsed = time.monotonic() - start_time
                if elapsed >= 1.5:
                    logger.warning(
                        f"function execution took too long ({elapsed:.2f}s), is `{fnc_call.name}` blocking?",
                        extra={
                            "function": fnc_call.name,
                            "speech_id": speech_handle.id,
                            "elapsed": elapsed,
                        },
                    )

    except asyncio.CancelledError:
        if len(tasks) > 0:
            names = [task.get_name() for task in tasks]
            logger.debug(
                "waiting for function call to finish before fully cancelling",
                extra={
                    "functions": names,
                    "speech_id": speech_handle.id,
                },
            )
            debug.Tracing.log_event(
                "waiting for function call to finish before fully cancelling",
                {
                    "functions": names,
                    "speech_id": speech_handle.id,
                },
            )
            await asyncio.gather(*tasks)
    finally:
        if len(out) > 0:
            logger.debug(
                "tools execution completed",
                extra={"speech_id": speech_handle.id},
            )
            debug.Tracing.log_event(
                "tools execution completed",
                {"speech_id": speech_handle.id},
            )


def _is_valid_function_output(value: Any) -> bool:
    VALID_TYPES = (str, int, float, bool, complex, type(None))

    if isinstance(value, VALID_TYPES):
        return True
    elif (
        isinstance(value, list)
        or isinstance(value, set)
        or isinstance(value, frozenset)
        or isinstance(value, tuple)
    ):
        return all(_is_valid_function_output(item) for item in value)
    elif isinstance(value, dict):
        return all(
            isinstance(key, VALID_TYPES) and _is_valid_function_output(val)
            for key, val in value.items()
        )
    return False


@dataclass
class _FunctionCallOutput:
    call_id: str
    name: str
    arguments: str
    output: Any
    exception: BaseException | None


def _sanitize_function_output(
    out: _FunctionCallOutput,
) -> tuple[llm.FunctionCallOutput | None, AgentTask | None]:
    from .agent_task import AgentTask

    if isinstance(out.exception, AIError):
        return llm.FunctionCallOutput(
            call_id=out.call_id,
            output=out.exception.message,
            is_error=True,
        ), None

    if isinstance(out.exception, StopResponse):
        return None, None

    if out.exception is not None:
        logger.error(
            "exception occurred while executing tool",
            extra={
                "call_id": out.call_id,
                "function": out.name,
            },
            exc_info=out.exception,
        )
        return llm.FunctionCallOutput(
            call_id=out.call_id,
            output="An internal error occurred",
            is_error=True,
        ), None

    fnc_out = out.output

    # find task if any
    task: AgentTask | None = None
    if isinstance(fnc_out, tuple):
        agent_tasks = [item for item in fnc_out if isinstance(item, AgentTask)]
        if len(agent_tasks) > 1:
            logger.error(
                "multiple AgentTask instances found in the function output tuple",
                extra={
                    "call_id": out.call_id,
                    "function": out.name,
                    "output": fnc_out,
                },
            )
            return None, None

        if agent_tasks:
            task = agent_tasks[0]

        fnc_out = [item for item in fnc_out if not isinstance(item, AgentTask)]

    if isinstance(fnc_out, AgentTask):
        task = fnc_out
        fnc_out = None

    # validate output without the task
    if not _is_valid_function_output(fnc_out):
        logger.error(
            "invalid function output type",
            extra={
                "call_id": out.call_id,
                "function": out.name,
                "output": fnc_out,
            },
        )
        return None, None

    return llm.FunctionCallOutput(
        call_id=out.call_id,
        output=str(out.output),
        is_error=False,
    ), task
