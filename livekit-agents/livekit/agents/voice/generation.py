from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
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
    async def aclose(self): ...


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
    tool_ctx: ToolContext | None,
    model_settings: ModelSettings,
) -> tuple[asyncio.Task, _LLMGenerationData]:
    text_ch = aio.Chan()
    function_ch = aio.Chan()

    data = _LLMGenerationData(text_ch=text_ch, function_ch=function_ch)

    @utils.log_exceptions(logger=logger)
    async def _inference_task():
        llm_node = node(
            chat_ctx,
            list(tool_ctx.function_tools.values()) if tool_ctx is not None else [],
            model_settings,
        )
        if asyncio.iscoroutine(llm_node):
            llm_node = await llm_node

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
) -> tuple[asyncio.Task, _TTSGenerationData]:
    audio_ch = aio.Chan[rtc.AudioFrame]()

    @utils.log_exceptions(logger=logger)
    async def _inference_task():
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
    first_text_fut: asyncio.Future


def perform_text_forwarding(
    *, text_output: io.TextOutput | None, source: AsyncIterable[str]
) -> tuple[asyncio.Task, _TextOutput]:
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
    first_frame_fut: asyncio.Future


def perform_audio_forwarding(
    *,
    audio_output: io.AudioOutput,
    tts_output: AsyncIterable[rtc.AudioFrame],
) -> tuple[asyncio.Task, _AudioOutput]:
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

            if not out.first_frame_fut.done():
                out.first_frame_fut.set_result(None)
    finally:
        if isinstance(tts_output, _ACloseable):
            await tts_output.aclose()

        if resampler:
            for frame in resampler.flush():
                await audio_output.capture_frame(frame)

        audio_output.flush()


def perform_tool_executions(
    *,
    session: AgentSession,
    speech_handle: SpeechHandle,
    tool_ctx: ToolContext,
    tool_choice: NotGivenOr[llm.ToolChoice],
    function_stream: AsyncIterable[llm.FunctionCall],
) -> tuple[
    asyncio.Task,
    list[_PythonOutput],
]:
    out: list[_PythonOutput] = []
    task = asyncio.create_task(
        _execute_tools_task(
            session=session,
            speech_handle=speech_handle,
            tool_ctx=tool_ctx,
            tool_choice=tool_choice,
            function_stream=function_stream,
            out=out,
        ),
        name="execute_tools_task",
    )
    return task, out


@utils.log_exceptions(logger=logger)
async def _execute_tools_task(
    *,
    session: AgentSession,
    speech_handle: SpeechHandle,
    tool_ctx: ToolContext,
    tool_choice: NotGivenOr[llm.ToolChoice],
    function_stream: AsyncIterable[llm.FunctionCall],
    out: list[_PythonOutput],
) -> None:
    """execute tools, when cancelled, stop executing new tools but wait for the pending ones"""

    from .agent import _authorize_inline_task
    from .events import RunContext

    tasks: list[asyncio.Task] = []
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

            try:
                function_model = llm_utils.function_arguments_to_pydantic_model(function_tool)
                json_args = fnc_call.arguments or "{}"
                parsed_args = function_model.model_validate_json(json_args)

            except ValidationError:
                logger.exception(
                    f"tried to call AI function `{fnc_call.name}` with invalid arguments",
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
                    "arguments": fnc_call.arguments,
                    "speech_id": speech_handle.id,
                },
            )

            fnc_args, fnc_kwargs = llm_utils.pydantic_model_to_function_arguments(
                model=parsed_args,
                function_tool=function_tool,
                call_ctx=RunContext(
                    session=session, speech_handle=speech_handle, function_call=fnc_call
                ),
            )

            py_out = _PythonOutput(
                fnc_call=fnc_call,
                output=None,
                exception=None,
            )

            task = asyncio.create_task(
                function_tool(*fnc_args, **fnc_kwargs),
                name=f"function_tool_{fnc_call.name}",
            )
            tasks.append(task)
            _authorize_inline_task(task)

            def _log_exceptions(task: asyncio.Task) -> None:
                if task.exception() is not None:
                    logger.error(
                        "exception occurred while executing tool",
                        extra={
                            "function": fnc_call.name,  # noqa: B023
                            "speech_id": speech_handle.id,
                        },
                        exc_info=task.exception(),
                    )
                    py_out.exception = task.exception()  # noqa: B023
                    out.append(py_out)  # noqa: B023
                    return

                py_out.output = task.result()  # noqa: B023
                out.append(py_out)  # noqa: B023
                tasks.remove(task)

            task.add_done_callback(_log_exceptions)

        await asyncio.gather(*tasks)

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
class _SanitizedOutput:
    fnc_call: llm.FunctionCall
    fnc_call_out: llm.FunctionCallOutput | None
    agent_task: Agent | None


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
                    output=str(fnc_out),  # take the string representation of the output
                    is_error=False,
                )
                if fnc_out is not None
                else None
            ),
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
    if msg := chat_ctx.get_by_id(INSTRUCTIONS_MESSAGE_ID):
        if msg.type == "message":
            msg.content = [instructions]
        else:
            raise ValueError(
                "expected the instructions inside the chat_ctx to be of type 'message'"
            )
    elif add_if_missing:
        # insert the instructions at the beginning of the chat context
        chat_ctx.items.insert(
            0,
            llm.ChatMessage(
                id=INSTRUCTIONS_MESSAGE_ID,
                role="system",
                content=[instructions],
            ),
        )


STANDARD_SPEECH_RATE = 0.5  # words per second


def truncate_message(*, message: str, played_duration: float) -> str:
    # TODO(theomonnom): this is very naive
    from ..tokenize import _basic_word

    words = _basic_word.split_words(message, ignore_punctuation=False)
    total_duration = len(words) * STANDARD_SPEECH_RATE

    if total_duration <= played_duration:
        return message

    max_words = int(played_duration // STANDARD_SPEECH_RATE)
    if max_words < 1:
        return ""

    _, _, end_pos = words[max_words - 1]
    return message[:end_pos]
