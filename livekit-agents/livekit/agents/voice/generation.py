from __future__ import annotations

import asyncio
import functools
import inspect
import json
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, runtime_checkable

from opentelemetry import trace
from pydantic import ValidationError

from livekit import rtc

from .. import llm, utils
from ..llm import (
    ChatChunk,
    ChatContext,
    StopResponse,
    ToolContext,
    ToolError,
    utils as llm_utils,
)
from ..llm.tool_context import (
    is_function_tool,
    is_raw_function_tool,
)
from ..log import logger
from ..telemetry import trace_types, tracer
from ..types import USERDATA_TIMED_TRANSCRIPT, NotGivenOr
from ..utils import aio
from . import io
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from .agent import Agent, ModelSettings
    from .agent_session import AgentSession


@runtime_checkable
class _ACloseable(Protocol):
    async def aclose(self) -> Any: ...


@dataclass
class _LLMGenerationData:
    text_ch: aio.Chan[str]
    function_ch: aio.Chan[llm.FunctionCall]
    generated_text: str = ""
    generated_functions: list[llm.FunctionCall] = field(default_factory=list)
    id: str = field(default_factory=lambda: utils.shortuuid("item_"))
    started_fut: asyncio.Future[None] = field(default_factory=asyncio.Future)


def perform_llm_inference(
    *,
    node: io.LLMNode,
    chat_ctx: ChatContext,
    tool_ctx: ToolContext,
    model_settings: ModelSettings,
) -> tuple[asyncio.Task[bool], _LLMGenerationData]:
    text_ch = aio.Chan[str]()
    function_ch = aio.Chan[llm.FunctionCall]()
    data = _LLMGenerationData(text_ch=text_ch, function_ch=function_ch)
    llm_task = asyncio.create_task(
        _llm_inference_task(node, chat_ctx, tool_ctx, model_settings, data)
    )
    llm_task.add_done_callback(lambda _: text_ch.close())
    llm_task.add_done_callback(lambda _: function_ch.close())

    def _cleanup(_: asyncio.Task[bool]) -> None:
        if not data.started_fut.done():
            data.started_fut.set_result(None)

    llm_task.add_done_callback(_cleanup)

    return llm_task, data


@utils.log_exceptions(logger=logger)
@tracer.start_as_current_span("llm_node")
async def _llm_inference_task(
    node: io.LLMNode,
    chat_ctx: ChatContext,
    tool_ctx: ToolContext,
    model_settings: ModelSettings,
    data: _LLMGenerationData,
) -> bool:
    current_span = trace.get_current_span()
    data.started_fut.set_result(None)

    text_ch, function_ch = data.text_ch, data.function_ch
    tools = list(tool_ctx.function_tools.values())

    current_span.set_attribute(
        trace_types.ATTR_CHAT_CTX,
        json.dumps(
            chat_ctx.to_dict(exclude_audio=True, exclude_image=True, exclude_timestamp=False)
        ),
    )
    current_span.set_attribute(
        trace_types.ATTR_FUNCTION_TOOLS, json.dumps(list(tool_ctx.function_tools.keys()))
    )

    llm_node = node(chat_ctx, tools, model_settings)
    if asyncio.iscoroutine(llm_node):
        llm_node = await llm_node

    # update the tool context after llm node
    tool_ctx.update_tools(tools)

    if isinstance(llm_node, str):
        data.generated_text = llm_node
        text_ch.send_nowait(llm_node)
        current_span.set_attribute(trace_types.ATTR_RESPONSE_TEXT, data.generated_text)
        return True

    if not isinstance(llm_node, AsyncIterable):
        return False

    # forward llm stream to output channels
    try:
        async for chunk in llm_node:
            # io.LLMNode can either return a string or a ChatChunk
            if isinstance(chunk, str):
                data.generated_text += chunk
                text_ch.send_nowait(chunk)

            elif isinstance(chunk, ChatChunk):
                if not chunk.delta:
                    continue

                if chunk.delta.tool_calls:
                    for tool in chunk.delta.tool_calls:
                        if tool.type != "function":
                            continue

                        fnc_call = llm.FunctionCall(
                            id=f"{data.id}/fnc_{len(data.generated_functions)}",
                            call_id=tool.call_id,
                            name=tool.name,
                            arguments=tool.arguments,
                        )
                        data.generated_functions.append(fnc_call)
                        function_ch.send_nowait(fnc_call)

                if chunk.delta.content:
                    data.generated_text += chunk.delta.content
                    text_ch.send_nowait(chunk.delta.content)
            else:
                logger.warning(
                    f"LLM node returned an unexpected type: {type(chunk)}",
                )
    finally:
        if isinstance(llm_node, _ACloseable):
            await llm_node.aclose()

    current_span.set_attribute(trace_types.ATTR_RESPONSE_TEXT, data.generated_text)
    current_span.set_attribute(
        trace_types.ATTR_RESPONSE_FUNCTION_CALLS,
        json.dumps(
            [fnc.model_dump(exclude={"type", "created_at"}) for fnc in data.generated_functions]
        ),
    )
    return True


@dataclass
class _TTSGenerationData:
    audio_ch: aio.Chan[rtc.AudioFrame]
    timed_texts_fut: asyncio.Future[aio.Chan[io.TimedString] | None]


def perform_tts_inference(
    *, node: io.TTSNode, input: AsyncIterable[str], model_settings: ModelSettings
) -> tuple[asyncio.Task[bool], _TTSGenerationData]:
    audio_ch = aio.Chan[rtc.AudioFrame]()
    timed_texts_fut = asyncio.Future[Optional[aio.Chan[io.TimedString]]]()
    data = _TTSGenerationData(audio_ch=audio_ch, timed_texts_fut=timed_texts_fut)

    tts_task = asyncio.create_task(_tts_inference_task(node, input, model_settings, data))

    def _inference_done(_: asyncio.Task[bool]) -> None:
        if timed_texts_fut.done() and (timed_text_ch := timed_texts_fut.result()):
            timed_text_ch.close()

        audio_ch.close()

    tts_task.add_done_callback(_inference_done)

    return tts_task, data


@utils.log_exceptions(logger=logger)
@tracer.start_as_current_span("tts_node")
async def _tts_inference_task(
    node: io.TTSNode,
    input: AsyncIterable[str],
    model_settings: ModelSettings,
    data: _TTSGenerationData,
) -> bool:
    audio_ch, timed_texts_fut = data.audio_ch, data.timed_texts_fut
    tts_node = node(input, model_settings)
    if asyncio.iscoroutine(tts_node):
        tts_node = await tts_node

    if isinstance(tts_node, AsyncIterable):
        timed_text_ch = aio.Chan[io.TimedString]()
        timed_texts_fut.set_result(timed_text_ch)

        async for audio_frame in tts_node:
            for text in audio_frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, []):
                timed_text_ch.send_nowait(text)

            audio_ch.send_nowait(audio_frame)
        return True

    timed_texts_fut.set_result(None)
    return False


@dataclass
class _TextOutput:
    text: str
    first_text_fut: asyncio.Future[None]


def perform_text_forwarding(
    *, text_output: io.TextOutput | None, source: AsyncIterable[str]
) -> tuple[asyncio.Task[None], _TextOutput]:
    out = _TextOutput(text="", first_text_fut=asyncio.Future())
    task = asyncio.create_task(_text_forwarding_task(text_output, source, out))
    return task, out


@utils.log_exceptions(logger=logger)
async def _text_forwarding_task(
    text_output: io.TextOutput | None,
    source: AsyncIterable[str],
    out: _TextOutput,
) -> None:
    try:
        async for delta in source:
            out.text += delta
            if text_output is not None:
                await text_output.capture_text(delta)

            if not out.first_text_fut.done():
                out.first_text_fut.set_result(None)
    finally:
        if isinstance(source, _ACloseable):
            await source.aclose()

        if text_output is not None:
            text_output.flush()


@dataclass
class _AudioOutput:
    audio: list[rtc.AudioFrame]
    first_frame_fut: asyncio.Future[None]


def perform_audio_forwarding(
    *,
    audio_output: io.AudioOutput,
    tts_output: AsyncIterable[rtc.AudioFrame],
) -> tuple[asyncio.Task[None], _AudioOutput]:
    out = _AudioOutput(audio=[], first_frame_fut=asyncio.Future())
    task = asyncio.create_task(_audio_forwarding_task(audio_output, tts_output, out))
    return task, out


@utils.log_exceptions(logger=logger)
async def _audio_forwarding_task(
    audio_output: io.AudioOutput,
    tts_output: AsyncIterable[rtc.AudioFrame],
    out: _AudioOutput,
) -> None:
    resampler: rtc.AudioResampler | None = None
    try:
        async for frame in tts_output:
            out.audio.append(frame)

            if (
                not out.first_frame_fut.done()
                and audio_output.sample_rate is not None
                and frame.sample_rate != audio_output.sample_rate
                and resampler is None
            ):
                resampler = rtc.AudioResampler(
                    input_rate=frame.sample_rate,
                    output_rate=audio_output.sample_rate,
                    num_channels=frame.num_channels,
                )

            if resampler:
                for f in resampler.push(frame):
                    await audio_output.capture_frame(f)
            else:
                await audio_output.capture_frame(frame)

            # set the first frame future if not already set
            # (after completing the first frame)
            if not out.first_frame_fut.done():
                out.first_frame_fut.set_result(None)
    finally:
        if isinstance(tts_output, _ACloseable):
            try:
                await tts_output.aclose()
            except Exception as e:
                logger.error("error while closing tts output", exc_info=e)

        if resampler:
            try:
                for frame in resampler.flush():
                    await audio_output.capture_frame(frame)
            except Exception as e:
                logger.error("error while flushing resampler", exc_info=e)

        audio_output.flush()


@dataclass
class _ToolOutput:
    output: list[ToolExecutionOutput]
    first_tool_started_fut: asyncio.Future[None]


def perform_tool_executions(
    *,
    session: AgentSession,
    speech_handle: SpeechHandle,
    tool_ctx: ToolContext,
    tool_choice: NotGivenOr[llm.ToolChoice],
    function_stream: AsyncIterable[llm.FunctionCall],
    tool_execution_started_cb: Callable[[llm.FunctionCall], Any],
    tool_execution_completed_cb: Callable[[ToolExecutionOutput], Any],
) -> tuple[asyncio.Task[None], _ToolOutput]:
    tool_output = _ToolOutput(output=[], first_tool_started_fut=asyncio.Future())
    task = asyncio.create_task(
        _execute_tools_task(
            session=session,
            speech_handle=speech_handle,
            tool_ctx=tool_ctx,
            tool_choice=tool_choice,
            function_stream=function_stream,
            tool_output=tool_output,
            tool_execution_started_cb=tool_execution_started_cb,
            tool_execution_completed_cb=tool_execution_completed_cb,
        ),
        name="execute_tools_task",
    )
    return task, tool_output


@utils.log_exceptions(logger=logger)
async def _execute_tools_task(
    *,
    session: AgentSession,
    speech_handle: SpeechHandle,
    tool_ctx: ToolContext,
    tool_choice: NotGivenOr[llm.ToolChoice],
    function_stream: AsyncIterable[llm.FunctionCall],
    tool_execution_started_cb: Callable[[llm.FunctionCall], Any],
    tool_execution_completed_cb: Callable[[ToolExecutionOutput], Any],
    tool_output: _ToolOutput,
) -> None:
    """execute tools, when cancelled, stop executing new tools but wait for the pending ones"""

    from .agent import _set_activity_task_info
    from .events import RunContext

    def _tool_completed(out: ToolExecutionOutput) -> None:
        tool_execution_completed_cb(out)
        tool_output.output.append(out)

    tasks: list[asyncio.Task[Any]] = []
    try:
        async for fnc_call in function_stream:
            if tool_choice == "none":
                logger.error(
                    "received a tool call with tool_choice set to 'none', ignoring",
                    extra={
                        "function": fnc_call.name,
                        "speech_id": speech_handle.id,
                    },
                )
                continue

            # TODO(theomonnom): assert other tool_choice values

            if (function_tool := tool_ctx.function_tools.get(fnc_call.name)) is None:
                logger.warning(
                    f"unknown AI function `{fnc_call.name}`",
                    extra={
                        "function": fnc_call.name,
                        "speech_id": speech_handle.id,
                    },
                )
                continue

            if not is_function_tool(function_tool) and not is_raw_function_tool(function_tool):
                logger.error(
                    f"unknown tool type: {type(function_tool)}",
                    extra={
                        "function": fnc_call.name,
                        "speech_id": speech_handle.id,
                    },
                )
                continue

            try:
                json_args = fnc_call.arguments or "{}"
                fnc_args, fnc_kwargs = llm_utils.prepare_function_arguments(
                    fnc=function_tool,
                    json_arguments=json_args,
                    call_ctx=RunContext(
                        session=session,
                        speech_handle=speech_handle,
                        function_call=fnc_call,
                    ),
                )

            except (ValidationError, ValueError) as e:
                logger.exception(
                    f"tried to call AI function `{fnc_call.name}` with invalid arguments",
                    extra={
                        "function": fnc_call.name,
                        "arguments": fnc_call.arguments,
                        "speech_id": speech_handle.id,
                    },
                )
                _tool_completed(make_tool_output(fnc_call=fnc_call, output=None, exception=e))
                continue

            if not tool_output.first_tool_started_fut.done():
                tool_output.first_tool_started_fut.set_result(None)

            tool_execution_started_cb(fnc_call)
            try:
                from .run_result import _MockToolsContextVar

                mock_tools: dict[str, Callable] = _MockToolsContextVar.get({}).get(
                    type(session.current_agent), {}
                )

                if mock := mock_tools.get(fnc_call.name):
                    logger.debug(
                        "executing mock tool",
                        extra={
                            "function": fnc_call.name,
                            "arguments": fnc_call.arguments,
                            "speech_id": speech_handle.id,
                        },
                    )

                    async def _run_mock(mock: Callable, *fnc_args: Any, **fnc_kwargs: Any) -> Any:
                        sig = inspect.signature(mock)

                        pos_param_names = [
                            name
                            for name, param in sig.parameters.items()
                            if param.kind
                            in (
                                inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            )
                        ]
                        max_positional = len(pos_param_names)
                        trimmed_args = fnc_args[:max_positional]
                        kw_param_names = [
                            name
                            for name, param in sig.parameters.items()
                            if param.kind
                            in (
                                inspect.Parameter.KEYWORD_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            )
                        ]
                        trimmed_kwargs = {
                            k: v for k, v in fnc_kwargs.items() if k in kw_param_names
                        }

                        bound = sig.bind_partial(*trimmed_args, **trimmed_kwargs)
                        bound.apply_defaults()

                        if asyncio.iscoroutinefunction(mock):
                            return await mock(*bound.args, **bound.kwargs)
                        else:
                            return mock(*bound.args, **bound.kwargs)

                    function_callable = functools.partial(_run_mock, mock, *fnc_args, **fnc_kwargs)
                else:
                    logger.debug(
                        "executing tool",
                        extra={
                            "function": fnc_call.name,
                            "arguments": fnc_call.arguments,
                            "speech_id": speech_handle.id,
                        },
                    )
                    function_callable = functools.partial(function_tool, *fnc_args, **fnc_kwargs)

                @tracer.start_as_current_span("function_tool")
                async def _traceable_fnc_tool(
                    function_callable: Callable, fnc_call: llm.FunctionCall
                ) -> None:
                    current_span = trace.get_current_span()
                    current_span.set_attribute(trace_types.ATTR_FUNCTION_TOOL_NAME, fnc_call.name)
                    current_span.set_attribute(
                        trace_types.ATTR_FUNCTION_TOOL_ARGS, fnc_call.arguments
                    )

                    try:
                        val = await function_callable()
                        output = make_tool_output(fnc_call=fnc_call, output=val, exception=None)
                    except BaseException as e:
                        logger.exception(
                            "exception occurred while executing tool",
                            extra={"function": fnc_call.name, "speech_id": speech_handle.id},
                        )

                        output = make_tool_output(fnc_call=fnc_call, output=None, exception=e)

                    if fnc_call_out := output.fnc_call_out:
                        current_span.set_attribute(
                            trace_types.ATTR_FUNCTION_TOOL_OUTPUT, fnc_call_out.output
                        )
                        current_span.set_attribute(
                            trace_types.ATTR_FUNCTION_TOOL_IS_ERROR, fnc_call_out.is_error
                        )

                    # TODO(theomonnom): Add the agent handoff inside the current_span
                    _tool_completed(output)

                task = asyncio.create_task(_traceable_fnc_tool(function_callable, fnc_call))
                _set_activity_task_info(
                    task, speech_handle=speech_handle, function_call=fnc_call, inline_task=True
                )
                tasks.append(task)
                task.add_done_callback(lambda task: tasks.remove(task))
            except Exception as e:
                # catching exceptions here because even though the function is asynchronous,
                # errors such as missing or incompatible arguments can still occur at
                # invocation time.
                logger.exception(
                    "exception occurred while executing tool",
                    extra={
                        "function": fnc_call.name,
                        "speech_id": speech_handle.id,
                    },
                )
                _tool_completed(make_tool_output(fnc_call=fnc_call, output=None, exception=e))
                continue

        await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))

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
            await asyncio.gather(*tasks)
    finally:
        await utils.aio.cancel_and_wait(*tasks)

        if len(tool_output.output) > 0:
            logger.debug(
                "tools execution completed",
                extra={"speech_id": speech_handle.id},
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
class ToolExecutionOutput:
    fnc_call: llm.FunctionCall
    fnc_call_out: llm.FunctionCallOutput | None
    agent_task: Agent | None
    raw_output: Any
    raw_exception: BaseException | None
    reply_required: bool = field(default=True)


def make_tool_output(
    *, fnc_call: llm.FunctionCall, output: Any, exception: BaseException | None
) -> ToolExecutionOutput:
    from .agent import Agent

    # support returning Exception instead of raising them (for devex purposes inside evals)
    if isinstance(output, BaseException):
        exception = output
        output = None

    if isinstance(exception, ToolError):
        return ToolExecutionOutput(
            fnc_call=fnc_call.model_copy(),
            fnc_call_out=llm.FunctionCallOutput(
                name=fnc_call.name,
                call_id=fnc_call.call_id,
                output=exception.message,
                is_error=True,
            ),
            agent_task=None,
            raw_output=output,
            raw_exception=exception,
        )

    if isinstance(exception, StopResponse):
        return ToolExecutionOutput(
            fnc_call=fnc_call.model_copy(),
            fnc_call_out=None,
            agent_task=None,
            raw_output=output,
            raw_exception=exception,
        )

    if exception is not None:
        return ToolExecutionOutput(
            fnc_call=fnc_call.model_copy(),
            fnc_call_out=llm.FunctionCallOutput(
                name=fnc_call.name,
                call_id=fnc_call.call_id,
                output="An internal error occurred",  # Don't send the actual error message, as it may contain sensitive information  # noqa: E501
                is_error=True,
            ),
            agent_task=None,
            raw_output=output,
            raw_exception=exception,
        )

    task: Agent | None = None
    fnc_out: Any = output
    if (
        isinstance(output, list)
        or isinstance(output, set)
        or isinstance(output, frozenset)
        or isinstance(output, tuple)
    ):
        agent_tasks = [item for item in output if isinstance(item, Agent)]
        other_outputs = [item for item in output if not isinstance(item, Agent)]
        if len(agent_tasks) > 1:
            logger.error(
                f"AI function `{fnc_call.name}` returned multiple AgentTask instances, ignoring the output",  # noqa: E501
                extra={
                    "call_id": fnc_call.call_id,
                    "output": output,
                },
            )

            return ToolExecutionOutput(
                fnc_call=fnc_call.model_copy(),
                fnc_call_out=None,
                agent_task=None,
                raw_output=output,
                raw_exception=exception,
            )

        task = next(iter(agent_tasks), None)

        # fmt: off
        fnc_out = (
            other_outputs if task is None
            else None if not other_outputs
            else other_outputs[0] if len(other_outputs) == 1
            else other_outputs
        )
        # fmt: on

    elif isinstance(fnc_out, Agent):
        task = fnc_out
        fnc_out = None

    if not _is_valid_function_output(fnc_out):
        logger.error(
            f"AI function `{fnc_call.name}` returned an invalid output",
            extra={
                "call_id": fnc_call.call_id,
                "output": output,
            },
        )
        return ToolExecutionOutput(
            fnc_call=fnc_call.model_copy(),
            fnc_call_out=None,
            agent_task=None,
            raw_output=output,
            raw_exception=exception,
        )

    return ToolExecutionOutput(
        fnc_call=fnc_call.model_copy(),
        fnc_call_out=(
            llm.FunctionCallOutput(
                name=fnc_call.name,
                call_id=fnc_call.call_id,
                output=str(fnc_out or ""),  # take the string representation of the output
                is_error=False,
            )
        ),
        reply_required=fnc_out is not None,  # require a reply if the tool returned an output
        agent_task=task,
        raw_output=output,
        raw_exception=exception,
    )


INSTRUCTIONS_MESSAGE_ID = "lk.agent_task.instructions"  #  value must not change
"""
The ID of the instructions message in the chat context. (only for stateless LLMs)
"""


def update_instructions(chat_ctx: ChatContext, *, instructions: str, add_if_missing: bool) -> None:
    """
    Update the instruction message in the chat context or insert a new one if missing.

    This function looks for an existing instruction message in the chat context using the identifier
    'INSTRUCTIONS_MESSAGE_ID'.

    Raises:
        ValueError: If an existing instruction message is not of type "message".
    """
    idx = chat_ctx.index_by_id(INSTRUCTIONS_MESSAGE_ID)
    if idx is not None:
        if chat_ctx.items[idx].type == "message":
            # create a new instance to avoid mutating the original
            chat_ctx.items[idx] = llm.ChatMessage(
                id=INSTRUCTIONS_MESSAGE_ID,
                role="system",
                content=[instructions],
                created_at=chat_ctx.items[idx].created_at,
            )
        else:
            raise ValueError(
                "expected the instructions inside the chat_ctx to be of type 'message'"
            )
    elif add_if_missing:
        # insert the instructions at the beginning of the chat context
        chat_ctx.items.insert(
            0,
            llm.ChatMessage(id=INSTRUCTIONS_MESSAGE_ID, role="system", content=[instructions]),
        )


def remove_instructions(chat_ctx: ChatContext) -> None:
    # loop in case there are items with the same id (shouldn't happen!)
    while True:
        if msg := chat_ctx.get_by_id(INSTRUCTIONS_MESSAGE_ID):
            chat_ctx.items.remove(msg)
        else:
            break
