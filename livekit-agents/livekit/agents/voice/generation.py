from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Protocol,
    Tuple,
    runtime_checkable,
)

from livekit import rtc
from pydantic import ValidationError

from .. import debug, llm, utils
from ..llm import (
    AIError,
    ChatChunk,
    ChatContext,
    FunctionContext,
    StopResponse,
)
from ..llm import (
    utils as llm_utils,
)
from ..log import logger
from ..utils import aio
from . import io
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from .agent_task import AgentTask
    from .voice_agent import VoiceAgent


@runtime_checkable
class _ACloseable(Protocol):
    async def aclose(self): ...


@dataclass
class _LLMGenerationData:
    text_ch: aio.Chan[str]
    function_ch: aio.Chan[llm.FunctionCall]
    generated_text: str = ""
    generated_functions: list[llm.FunctionCall] = field(default_factory=list)


def perform_llm_inference(
    *, node: io.LLMNode, chat_ctx: ChatContext, fnc_ctx: FunctionContext | None
) -> Tuple[asyncio.Task, _LLMGenerationData]:
    text_ch = aio.Chan()
    function_ch = aio.Chan()

    data = _LLMGenerationData(text_ch=text_ch, function_ch=function_ch)

    async def _inference_task():
        llm_node = node(
            chat_ctx, list(fnc_ctx.ai_functions.values()) if fnc_ctx is not None else []
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
    *, node: io.TTSNode, input: AsyncIterable[str]
) -> Tuple[asyncio.Task, _TTSGenerationData]:
    audio_ch = aio.Chan[rtc.AudioFrame]()

    async def _inference_task():
        tts_node = node(input)
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
    *, text_output: io.TextSink | None, source: AsyncIterable[str]
) -> tuple[asyncio.Task, _TextOutput]:
    out = _TextOutput(text="", first_text_fut=asyncio.Future())
    task = asyncio.create_task(_text_forwarding_task(text_output, source, out))
    return task, out


@utils.log_exceptions(logger=logger)
async def _text_forwarding_task(
    text_output: io.TextSink | None,
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
    first_frame_fut: asyncio.Future[None] = field(default_factory=asyncio.Future)


def perform_audio_forwarding(
    *,
    audio_output: io.AudioSink,
    tts_output: AsyncIterable[rtc.AudioFrame],
) -> tuple[asyncio.Task, _AudioOutput]:
    out = _AudioOutput(audio=[])
    task = asyncio.create_task(_audio_forwarding_task(audio_output, tts_output, out))
    return task, out


@utils.log_exceptions(logger=logger)
async def _audio_forwarding_task(
    audio_output: io.AudioSink,
    tts_output: AsyncIterable[rtc.AudioFrame],
    out: _AudioOutput,
) -> None:
    try:
        async for frame in tts_output:
            out.audio.append(frame)
            await audio_output.capture_frame(frame)
            if not out.first_frame_fut.done():
                out.first_frame_fut.set_result(None)
    finally:
        if isinstance(tts_output, _ACloseable):
            await tts_output.aclose()

        audio_output.flush()


def perform_tool_executions(
    *,
    agent: VoiceAgent,
    speech_handle: SpeechHandle,
    fnc_ctx: FunctionContext,
    function_stream: AsyncIterable[llm.FunctionCall],
) -> tuple[
    asyncio.Task,
    list[tuple[llm.FunctionCall, llm.FunctionCallOutput | None, AgentTask | None]],
]:
    out: list[tuple[llm.FunctionCall, llm.FunctionCallOutput | None, AgentTask | None]] = []
    task = asyncio.create_task(
        _execute_tools_task(
            agent=agent,
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
    agent: VoiceAgent,
    speech_handle: SpeechHandle,
    fnc_ctx: FunctionContext,
    function_stream: AsyncIterable[llm.FunctionCall],
    out: list[tuple[llm.FunctionCall, llm.FunctionCallOutput | None, AgentTask | None]],
) -> None:
    """execute tools, when cancelled, stop executing new tools but wait for the pending ones"""

    from .events import CallContext

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
                function_model = llm_utils.function_arguments_to_pydantic_model(ai_function)
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
                call_ctx=CallContext(
                    agent=agent, speech_handle=speech_handle, function_call=fnc_call
                ),
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
                        out.append(_sanitize_function_output(fnc_call, fnc_out))
                        return

                    fnc_out.output = task.result()
                    out.append(_sanitize_function_output(fnc_call, fnc_out))
                    tasks.remove(task)

                task.add_done_callback(_log_exceptions)
            else:
                start_time = time.monotonic()
                try:
                    output = ai_function(*fnc_args, **fnc_kwargs)
                    fnc_out.output = output
                    out.append(_sanitize_function_output(fnc_call, fnc_out))
                except Exception as e:
                    fnc_out.exception = e
                    out.append(_sanitize_function_output(fnc_call, fnc_out))

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
    fnc_call: llm.FunctionCall,
    out: _FunctionCallOutput,
) -> tuple[llm.FunctionCall, llm.FunctionCallOutput | None, AgentTask | None]:
    from .agent_task import AgentTask

    if isinstance(out.exception, AIError):
        return (
            fnc_call,
            llm.FunctionCallOutput(
                call_id=out.call_id,
                output=out.exception.message,
                is_error=True,
            ),
            None,
        )

    if isinstance(out.exception, StopResponse):
        return fnc_call, None, None

    if out.exception is not None:
        logger.error(
            "exception occurred while executing tool",
            extra={
                "call_id": out.call_id,
                "function": out.name,
            },
            exc_info=out.exception,
        )
        return (
            fnc_call,
            llm.FunctionCallOutput(
                call_id=out.call_id,
                output="An internal error occurred",
                is_error=True,
            ),
            None,
        )

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
            return fnc_call, None, None

        if agent_tasks:
            task = agent_tasks[0]

        fnc_out = [item for item in fnc_out if not isinstance(item, AgentTask)]
        if len(fnc_out) == 1:
            fnc_out = fnc_out[0]

        if len(fnc_out) == 0:
            fnc_out = None

    elif isinstance(fnc_out, AgentTask):
        task = fnc_out
        fnc_out = None

    if not _is_valid_function_output(fnc_out):
        logger.error(
            "invalid function output type",
            extra={
                "call_id": out.call_id,
                "function": out.name,
                "output": fnc_out,
            },
        )
        return fnc_call, None, None

    if fnc_out is None:
        return fnc_call, None, task

    return (
        fnc_call,
        llm.FunctionCallOutput(
            call_id=out.call_id,
            output=str(fnc_out),
            is_error=False,
        ),
        task,
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
