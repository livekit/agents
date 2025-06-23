from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import ValidationError

from livekit import rtc

from .. import debug, llm, utils
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
from ..types import NotGivenOr
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

    @utils.log_exceptions(logger=logger)
    async def _inference_task() -> bool:
        tools = list(tool_ctx.function_tools.values())
        llm_node = node(
            chat_ctx,
            tools,
            model_settings,
        )
        if asyncio.iscoroutine(llm_node):
            llm_node = await llm_node

        # update the tool context after llm node
        tool_ctx.update_tools(tools)

        if isinstance(llm_node, str):
            data.generated_text = llm_node
            text_ch.send_nowait(llm_node)
            return True

        if isinstance(llm_node, AsyncIterable):
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

            return True

        return False

    llm_task = asyncio.create_task(_inference_task())
    llm_task.add_done_callback(lambda _: text_ch.close())
    llm_task.add_done_callback(lambda _: function_ch.close())
    return llm_task, data


@dataclass
class _TTSGenerationData:
    audio_ch: aio.Chan[rtc.AudioFrame]


def perform_tts_inference(
    *, node: io.TTSNode, input: AsyncIterable[str], model_settings: ModelSettings
) -> tuple[asyncio.Task[bool], _TTSGenerationData]:
    audio_ch = aio.Chan[rtc.AudioFrame]()

    @utils.log_exceptions(logger=logger)
    async def _inference_task() -> bool:
        tts_node = node(input, model_settings)
        if asyncio.iscoroutine(tts_node):
            tts_node = await tts_node

        if isinstance(tts_node, AsyncIterable):
            async for audio_frame in tts_node:
                audio_ch.send_nowait(audio_frame)

            return True

        return False

    tts_task = asyncio.create_task(_inference_task())
    tts_task.add_done_callback(lambda _: audio_ch.close())

    return tts_task, _TTSGenerationData(audio_ch=audio_ch)


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
            await tts_output.aclose()

        if resampler:
            for frame in resampler.flush():
                await audio_output.capture_frame(frame)

        audio_output.flush()


@dataclass
class _ToolOutput:
    output: list[_PythonOutput]
    first_tool_fut: asyncio.Future[None]


def perform_tool_executions(
    *,
    session: AgentSession,
    speech_handle: SpeechHandle,
    tool_ctx: ToolContext,
    tool_choice: NotGivenOr[llm.ToolChoice],
    function_stream: AsyncIterable[llm.FunctionCall],
) -> tuple[asyncio.Task[None], _ToolOutput]:
    tool_output = _ToolOutput(output=[], first_tool_fut=asyncio.Future())
    task = asyncio.create_task(
        _execute_tools_task(
            session=session,
            speech_handle=speech_handle,
            tool_ctx=tool_ctx,
            tool_choice=tool_choice,
            function_stream=function_stream,
            tool_output=tool_output,
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
    tool_output: _ToolOutput,
) -> None:
    """execute tools, when cancelled, stop executing new tools but wait for the pending ones"""

    from .agent import _authorize_inline_task
    from .events import RunContext

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

            py_out = _PythonOutput(fnc_call=fnc_call, output=None, exception=None)
            try:
                json_args = fnc_call.arguments or "{}"
                fnc_args, fnc_kwargs = llm_utils.prepare_function_arguments(
                    fnc=function_tool,
                    json_arguments=json_args,
                    call_ctx=RunContext(
                        session=session, speech_handle=speech_handle, function_call=fnc_call
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
                py_out.exception = e
                tool_output.output.append(py_out)
                continue

            if not tool_output.first_tool_fut.done():
                tool_output.first_tool_fut.set_result(None)

            logger.debug(
                "executing tool",
                extra={
                    "function": fnc_call.name,
                    "arguments": fnc_call.arguments,
                    "speech_id": speech_handle.id,
                },
            )

            try:
                task = asyncio.create_task(
                    function_tool(*fnc_args, **fnc_kwargs),
                    name=f"function_tool_{fnc_call.name}",
                )

                tasks.append(task)
                _authorize_inline_task(task, function_call=fnc_call)
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
                py_out.exception = e
                tool_output.output.append(py_out)
                continue

            def _log_exceptions(
                task: asyncio.Task[Any],
                *,
                py_out: _PythonOutput,
                fnc_call: llm.FunctionCall,
            ) -> None:
                if task.exception() is not None:
                    logger.error(
                        "exception occurred while executing tool",
                        extra={
                            "function": fnc_call.name,
                            "speech_id": speech_handle.id,
                        },
                        exc_info=task.exception(),
                    )
                    py_out.exception = task.exception()
                    tool_output.output.append(py_out)
                    return

                py_out.output = task.result()
                tool_output.output.append(py_out)
                tasks.remove(task)

            task.add_done_callback(partial(_log_exceptions, py_out=py_out, fnc_call=fnc_call))

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
            debug.Tracing.log_event(
                "waiting for function call to finish before fully cancelling",
                {
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
class _SanitizedOutput:
    fnc_call: llm.FunctionCall
    fnc_call_out: llm.FunctionCallOutput | None
    agent_task: Agent | None
    reply_required: bool = field(default=True)


@dataclass
class _PythonOutput:
    fnc_call: llm.FunctionCall
    output: Any
    exception: BaseException | None

    def sanitize(self) -> _SanitizedOutput:
        from .agent import Agent

        if isinstance(self.exception, ToolError):
            return _SanitizedOutput(
                fnc_call=self.fnc_call.model_copy(),
                fnc_call_out=llm.FunctionCallOutput(
                    name=self.fnc_call.name,
                    call_id=self.fnc_call.call_id,
                    output=self.exception.message,
                    is_error=True,
                ),
                agent_task=None,
            )

        if isinstance(self.exception, StopResponse):
            return _SanitizedOutput(
                fnc_call=self.fnc_call.model_copy(),
                fnc_call_out=None,
                agent_task=None,
            )

        if self.exception is not None:
            return _SanitizedOutput(
                fnc_call=self.fnc_call.model_copy(),
                fnc_call_out=llm.FunctionCallOutput(
                    name=self.fnc_call.name,
                    call_id=self.fnc_call.call_id,
                    output="An internal error occurred",  # Don't send the actual error message, as it may contain sensitive information  # noqa: E501
                    is_error=True,
                ),
                agent_task=None,
            )

        task: Agent | None = None
        fnc_out: Any = self.output
        if (
            isinstance(self.output, list)
            or isinstance(self.output, set)
            or isinstance(self.output, frozenset)
            or isinstance(self.output, tuple)
        ):
            agent_tasks = [item for item in self.output if isinstance(item, Agent)]
            other_outputs = [item for item in self.output if not isinstance(item, Agent)]
            if len(agent_tasks) > 1:
                logger.error(
                    f"AI function `{self.fnc_call.name}` returned multiple AgentTask instances, ignoring the output",  # noqa: E501
                    extra={
                        "call_id": self.fnc_call.call_id,
                        "output": self.output,
                    },
                )

                return _SanitizedOutput(
                    fnc_call=self.fnc_call.model_copy(),
                    fnc_call_out=None,
                    agent_task=None,
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
                f"AI function `{self.fnc_call.name}` returned an invalid output",
                extra={
                    "call_id": self.fnc_call.call_id,
                    "output": self.output,
                },
            )
            return _SanitizedOutput(
                fnc_call=self.fnc_call.model_copy(),
                fnc_call_out=None,
                agent_task=None,
            )

        return _SanitizedOutput(
            fnc_call=self.fnc_call.model_copy(),
            fnc_call_out=(
                llm.FunctionCallOutput(
                    name=self.fnc_call.name,
                    call_id=self.fnc_call.call_id,
                    output=str(fnc_out or ""),  # take the string representation of the output
                    is_error=False,
                )
            ),
            reply_required=fnc_out is not None,  # require a reply if the tool returned an output
            agent_task=task,
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
            0, llm.ChatMessage(id=INSTRUCTIONS_MESSAGE_ID, role="system", content=[instructions])
        )


def remove_instructions(chat_ctx: ChatContext) -> None:
    # loop in case there are items with the same id (shouldn't happen!)
    while True:
        if msg := chat_ctx.get_by_id(INSTRUCTIONS_MESSAGE_ID):
            chat_ctx.items.remove(msg)
        else:
            break
