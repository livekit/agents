from __future__ import annotations

import asyncio
import functools
import json
import time
from collections.abc import AsyncIterable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from opentelemetry import trace

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
from ..llm.chat_context import Instructions
from ..log import logger
from ..telemetry import trace_types, tracer
from ..types import (
    USERDATA_TIMED_TRANSCRIPT,
    USERDATA_TTS_STARTED_TIME,
    FlushSentinel,
    NotGivenOr,
)
from ..utils import aio
from ..utils.aio import itertools
from . import io
from .speech_handle import SpeechHandle
from .tool_executor import _build_executor_map
from .transcription.text_transforms import _apply_text_transforms

if TYPE_CHECKING:
    from .agent import Agent, ModelSettings
    from .agent_session import AgentSession
    from .transcription.text_transforms import TextTransforms


@runtime_checkable
class _ACloseable(Protocol):
    async def aclose(self) -> Any: ...


@dataclass
class _LLMGenerationData:
    text_ch: aio.Chan[str | FlushSentinel]
    function_ch: aio.Chan[llm.FunctionCall]
    generated_text: str = ""
    generated_functions: list[llm.FunctionCall] = field(default_factory=list)
    generated_extra: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: utils.shortuuid("item_"))
    started_fut: asyncio.Future[None] = field(default_factory=asyncio.Future)
    ttft: float | None = None


# output for an injected in-progress tool call, phrased so the model waits instead of
# re-issuing the call.
_RUNNING_TOOL_PLACEHOLDER = "The tool call is still in progress."
# extra flag marking an injected pair so it can be stripped before the ctx is forwarded.
_RUNNING_PLACEHOLDER_KEY = "__lk_running_placeholder__"


def _inject_running_tool_calls(
    chat_ctx: ChatContext,
    running_calls: Iterable[llm.FunctionCall],
    *,
    placeholder: str = _RUNNING_TOOL_PLACEHOLDER,
) -> None:
    """Add a flagged in-progress pair for each running tool call missing from ``chat_ctx``
    so the model won't re-issue an in-flight call. Mutates in place; strip the pairs with
    :func:`_strip_running_tool_calls` before the ctx is persisted or forwarded."""
    existing = {
        item.call_id
        for item in chat_ctx.items
        if item.type in ("function_call", "function_call_output")
    }
    for fnc_call in running_calls:
        if fnc_call.call_id in existing:
            continue
        existing.add(fnc_call.call_id)
        # copy so the executor's live FunctionCall stays unflagged
        call = fnc_call.model_copy(
            update={"extra": {**fnc_call.extra, _RUNNING_PLACEHOLDER_KEY: True}}
        )
        chat_ctx.insert(
            [
                call,
                llm.FunctionCallOutput(
                    call_id=fnc_call.call_id,
                    name=fnc_call.name,
                    output=placeholder,
                    is_error=False,
                    created_at=fnc_call.created_at,
                ),
            ]
        )


def _strip_running_tool_calls(chat_ctx: ChatContext) -> None:
    """Remove the pairs added by :func:`_inject_running_tool_calls`, keeping everything
    else (e.g. items a custom ``llm_node`` added)."""
    flagged = {
        item.call_id
        for item in chat_ctx.items
        if item.type == "function_call" and item.extra.get(_RUNNING_PLACEHOLDER_KEY)
    }
    if not flagged:
        return
    chat_ctx.items[:] = [
        item
        for item in chat_ctx.items
        if not (item.type in ("function_call", "function_call_output") and item.call_id in flagged)
    ]


def perform_llm_inference(
    *,
    node: io.LLMNode,
    chat_ctx: ChatContext,
    tool_ctx: ToolContext,
    model_settings: ModelSettings,
    model: str | None = None,
    provider: str | None = None,
) -> tuple[asyncio.Task[bool], _LLMGenerationData]:
    text_ch = aio.Chan[str | FlushSentinel]()
    function_ch = aio.Chan[llm.FunctionCall]()
    data = _LLMGenerationData(text_ch=text_ch, function_ch=function_ch)
    llm_task = asyncio.create_task(
        _llm_inference_task(node, chat_ctx, tool_ctx, model_settings, data, model, provider)
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
    model: str | None = None,
    provider: str | None = None,
) -> bool:
    start_time = time.perf_counter()
    current_span = trace.get_current_span()
    data.started_fut.set_result(None)

    text_ch, function_ch = data.text_ch, data.function_ch
    tools = tool_ctx.flatten()

    attrs: dict[str, Any] = {
        trace_types.ATTR_CHAT_CTX: json.dumps(
            chat_ctx.to_dict(
                exclude_audio=True,
                exclude_image=True,
                exclude_timestamp=True,
                exclude_metrics=True,
            )
        ),
        trace_types.ATTR_FUNCTION_TOOLS: list(tool_ctx.function_tools.keys()),
        trace_types.ATTR_PROVIDER_TOOLS: [type(tool).__name__ for tool in tool_ctx.provider_tools],
        trace_types.ATTR_TOOL_SETS: [type(tool_set).__name__ for tool_set in tool_ctx.toolsets],
    }
    if model:
        attrs[trace_types.ATTR_GEN_AI_REQUEST_MODEL] = model
    if provider:
        attrs[trace_types.ATTR_GEN_AI_PROVIDER_NAME] = provider
    current_span.set_attributes(attrs)

    llm_node = node(chat_ctx, tools, model_settings)
    if asyncio.iscoroutine(llm_node):
        llm_node = await llm_node

    # store any updated tools, to ensure subsequent tool calls in the same turn (nested calls)
    # are using the newer tools.
    # tool_ctx here is ephemeral for this turn, and we allow manipulations.
    # _sync_flattened writes back flat edits while preserving Toolset grouping
    # (e.g. tool_ctx.toolsets stays intact for executor routing on handoff).
    tool_ctx._sync_flattened(tools)
    tools_snapshot = tools.copy()

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
            if data.ttft is None:
                data.ttft = time.perf_counter() - start_time

            # extract text content from either str or ChatChunk
            content: str | None = None

            if isinstance(chunk, str):
                content = chunk

            elif isinstance(chunk, ChatChunk):
                if not chunk.delta:
                    continue

                # A tool call is starting: flush any buffered text preamble to TTS now,
                # so it plays while the tool arguments are still streaming in.
                if chunk.delta.tool_call_started:
                    text_ch.send_nowait(FlushSentinel())

                if chunk.delta.tool_calls:
                    for tool in chunk.delta.tool_calls:
                        if tool.type != "function":
                            continue

                        if (
                            tool_ctx.get_function_tool(tool.name) is None
                            and tools != tools_snapshot
                        ):
                            tool_ctx._sync_flattened(tools)
                            tools_snapshot = tools.copy()

                        fnc_call = llm.FunctionCall(
                            id=f"{data.id}/fnc_{len(data.generated_functions)}",
                            call_id=tool.call_id,
                            name=tool.name,
                            arguments=tool.arguments,
                            extra=tool.extra or {},
                        )
                        data.generated_functions.append(fnc_call)
                        function_ch.send_nowait(fnc_call)

                if chunk.delta.extra:
                    data.generated_extra.update(chunk.delta.extra)

                content = chunk.delta.content

            elif isinstance(chunk, FlushSentinel):
                text_ch.send_nowait(chunk)
                content = None
            else:
                logger.warning(
                    f"LLM node returned an unexpected type: {type(chunk)}",
                )
                content = None

            # route text content to output channels
            if content:
                data.generated_text += content
                text_ch.send_nowait(content)
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
    if data.ttft is not None:
        current_span.set_attribute(trace_types.ATTR_RESPONSE_TTFT, data.ttft)
    return True


@dataclass
class _TTSGenerationData:
    audio_ch: aio.Chan[rtc.AudioFrame]
    timed_texts_fut: asyncio.Future[aio.Chan[io.TimedString] | None]
    ttfb: float | None = None


def perform_tts_inference(
    *,
    node: io.TTSNode,
    input: AsyncIterable[str],
    model_settings: ModelSettings,
    text_transforms: Sequence[TextTransforms] | None,
    model: str | None = None,
    provider: str | None = None,
) -> tuple[asyncio.Task[bool], _TTSGenerationData]:
    audio_ch = aio.Chan[rtc.AudioFrame]()
    timed_texts_fut = asyncio.Future[aio.Chan[io.TimedString] | None]()
    data = _TTSGenerationData(audio_ch=audio_ch, timed_texts_fut=timed_texts_fut)

    tts_task = asyncio.create_task(
        _tts_inference_task(node, input, model_settings, data, text_transforms, model, provider)
    )

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
    text_transforms: Sequence[TextTransforms] | None,
    model: str | None = None,
    provider: str | None = None,
) -> bool:
    current_span = trace.get_current_span()
    if model:
        current_span.set_attribute(trace_types.ATTR_GEN_AI_REQUEST_MODEL, model)
    if provider:
        current_span.set_attribute(trace_types.ATTR_GEN_AI_PROVIDER_NAME, provider)

    audio_ch, timed_texts_fut = data.audio_ch, data.timed_texts_fut
    if text_transforms:
        input = _apply_text_transforms(input, text_transforms)

    start_time: float | None = None
    input_tee = itertools.tee(input, 2)

    async def _get_start_time() -> None:
        nonlocal start_time
        async for _ in input_tee[0]:
            start_time = time.perf_counter()
            break

    _start_time_task = asyncio.create_task(_get_start_time())
    try:
        tts_node = node(input_tee[1], model_settings)
        if asyncio.iscoroutine(tts_node):
            tts_node = await tts_node

        if not isinstance(tts_node, AsyncIterable):
            timed_texts_fut.set_result(None)
            return False

        timed_text_ch = aio.Chan[io.TimedString]()
        timed_texts_fut.set_result(timed_text_ch)

        audio_duration = 0.0
        async for audio_frame in tts_node:
            if data.ttfb is None:
                # the framework TTS streams attach the time the text was first sent to the
                # provider; without it (custom tts_node), fall back to the arrival of the
                # first input token, which also counts any text buffering (e.g. sentence
                # tokenization) as TTFB
                anchor: float | None = audio_frame.userdata.get(
                    USERDATA_TTS_STARTED_TIME, start_time
                )
                if anchor is not None:
                    data.ttfb = time.perf_counter() - anchor
                    current_span.set_attribute(trace_types.ATTR_RESPONSE_TTFB, data.ttfb)

            for text in audio_frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, []):
                if isinstance(text, io.TimedString):
                    timed_text_ch.send_nowait(text)

            audio_ch.send_nowait(audio_frame)
            audio_duration += audio_frame.duration
        return audio_duration > 0
    finally:
        await aio.gracefully_cancel(_start_time_task)
        await input_tee.aclose()


@dataclass
class _TextOutput:
    text: str
    first_text_fut: asyncio.Future[None]


def perform_text_forwarding(
    *,
    text_output: io.TextOutput | None,
    source: AsyncIterable[str],
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
    # The raw LLM text (expressive markup intact) is forwarded verbatim: it flows into
    # chat history via out.text and on to the transcript sinks. The markup is a TTS audio
    # directive, not spoken text, so the sinks strip it downstream (and surface the leading
    # expression as the segment's lk.expression attribute) — see TranscriptMarkupStripper.
    try:
        async for delta in source:
            out.text += delta
            if not out.first_text_fut.done():
                out.first_text_fut.set_result(None)
            if text_output is not None and delta:
                await text_output.capture_text(delta)
    finally:
        if isinstance(source, _ACloseable):
            await source.aclose()

        if text_output is not None:
            text_output.flush()


@dataclass
class _AudioOutput:
    audio: list[rtc.AudioFrame]
    first_frame_fut: asyncio.Future[float]
    """Future that will be set with the timestamp of the first frame's capture"""

    started_forwarding_at: float | None = None

    def _resolve_first_frame_fut(self, ev: io.PlaybackStartedEvent) -> None:
        if not self.first_frame_fut.done():
            self.first_frame_fut.set_result(ev.created_at)


def perform_audio_forwarding(
    *,
    audio_output: io.AudioOutput,
    tts_output: AsyncIterable[rtc.AudioFrame],
) -> tuple[asyncio.Task[None], _AudioOutput]:
    out = _AudioOutput(audio=[], first_frame_fut=asyncio.Future())
    # out.first_frame_fut should be cancelled in the caller after the playout is finished or interrupted
    audio_output.on("playback_started", out._resolve_first_frame_fut)
    out.first_frame_fut.add_done_callback(
        lambda _: audio_output.off("playback_started", out._resolve_first_frame_fut)
    )
    task = asyncio.create_task(_audio_forwarding_task(audio_output, tts_output, out))
    return task, out


@utils.log_exceptions(logger=logger)
async def _audio_forwarding_task(
    audio_output: io.AudioOutput,
    tts_output: AsyncIterable[rtc.AudioFrame],
    out: _AudioOutput,
) -> None:
    resampler: rtc.AudioResampler | None = None

    cancelled = False
    try:
        audio_output.resume()

        async for frame in tts_output:
            out.audio.append(frame)
            if out.started_forwarding_at is None:
                out.started_forwarding_at = time.time()

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

        if resampler:
            for frame in resampler.flush():
                await audio_output.capture_frame(frame)

    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        if isinstance(tts_output, _ACloseable):
            try:
                await tts_output.aclose()
            except Exception as e:
                logger.warning("error while closing tts output: %s", e)

        audio_output.flush()
        if cancelled:
            audio_output.clear_buffer()


@dataclass
class _ForwardOutput:
    """Result of forwarding one generation segment's audio and text to the outputs."""

    text_out: _TextOutput | None = None
    audio_out: _AudioOutput | None = None
    played: Literal["full", "partial", "skipped"] = "skipped"
    playback_position: float = 0.0
    synchronized_transcript: str | None = None

    @property
    def forwarded_text(self) -> str:
        """The text that actually reached the user, accounting for interruptions."""
        if self.played == "skipped":
            return ""
        if self.played == "partial" and self.synchronized_transcript is not None:
            return self.synchronized_transcript
        return self.text_out.text if self.text_out else ""


async def forward_generation(
    *,
    speech_handle: SpeechHandle,
    audio_output: io.AudioOutput | None,
    text_output: io.TextOutput | None,
    audio_source: AsyncIterable[rtc.AudioFrame] | None,
    text_source: AsyncIterable[str] | None,
    on_first_frame: Callable[[asyncio.Future[Any], _AudioOutput | None], None],
) -> _ForwardOutput:
    """Forward one segment's audio/text to the outputs, then wait for its playout.

    Returns when the segment has fully played, been interrupted, or never started
    (e.g. interrupted before the first frame). Callers resolve the audio/text sources
    and own message creation; this is the shared core between the pipeline and realtime
    generation paths.
    """
    out = _ForwardOutput()
    forward_tasks: list[asyncio.Task[Any]] = []
    try:
        audio_out: _AudioOutput | None = None
        if audio_output is not None and audio_source is not None:
            forward_audio_task, audio_out = perform_audio_forwarding(
                audio_output=audio_output, tts_output=audio_source
            )
            forward_tasks.append(forward_audio_task)
            audio_out.first_frame_fut.add_done_callback(lambda fut: on_first_frame(fut, audio_out))
            out.audio_out = audio_out

        text_out: _TextOutput | None = None
        if text_source is not None:
            forward_text_task, text_out = perform_text_forwarding(
                text_output=text_output, source=text_source
            )
            forward_tasks.append(forward_text_task)
            out.text_out = text_out

        if audio_out is None and text_out is not None:
            text_out.first_text_fut.add_done_callback(lambda fut: on_first_frame(fut, None))

        playout_fut: asyncio.Future[Any] | None = None
        await speech_handle.wait_if_not_interrupted(list(forward_tasks))
        if not speech_handle.interrupted and audio_output is not None:
            playout_fut = asyncio.ensure_future(audio_output.wait_for_playout())
            await speech_handle.wait_if_not_interrupted([playout_fut])

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*forward_tasks)
            if audio_output is not None:
                audio_output.clear_buffer()
                playback_ev = await audio_output.wait_for_playout()
                if (
                    audio_out is not None
                    and audio_out.first_frame_fut.done()
                    and not audio_out.first_frame_fut.cancelled()
                ):
                    out.played = "partial"
                    out.playback_position = playback_ev.playback_position
                    out.synchronized_transcript = playback_ev.synchronized_transcript
                # else: audio never reached the speakers, stays "skipped"
            elif text_out is not None and text_out.text:
                out.played = "partial"
            return out

        if audio_output is not None:
            assert playout_fut is not None
            playback_ev = playout_fut.result()
            out.played = "full"
            out.playback_position = playback_ev.playback_position
            out.synchronized_transcript = playback_ev.synchronized_transcript
        elif text_out is not None and text_out.text:
            out.played = "full"
        return out
    finally:
        await utils.aio.cancel_and_wait(*forward_tasks)


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
    """Dispatch tools through the activity's _ToolExecutor.

    Tools that never call ``ctx.update()`` behave like classic sync tools. Those
    that do release control to the LLM with the first update as their synthetic
    output, and later updates / the final return are coalesced into deferred replies.
    """

    from .agent import _set_activity_task_info
    from .events import RunContext
    from .run_result import _MockToolsContextVar

    def _tool_completed(out: ToolExecutionOutput) -> None:
        tool_execution_completed_cb(out)
        tool_output.output.append(out)

    activity = session._activity
    if activity is None:
        logger.error(
            "no active AgentActivity to execute tools",
            extra={"speech_id": speech_handle.id},
        )
        return

    # Route AsyncToolset members to their own executor so session-scoped async
    # tools survive handoff; everything else falls back to the activity executor.
    executor_by_name = _build_executor_map(
        toolsets=tool_ctx.toolsets, default=activity._tool_executor
    )

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
                _tool_completed(
                    make_tool_output(
                        fnc_call=fnc_call,
                        output=None,
                        # Name the available tools so the model can self-correct
                        exception=ToolError(
                            f"Unknown function: {fnc_call.name} - available tools: "
                            f"{', '.join(tool_ctx.function_tools.keys())}"
                        ),
                    )
                )
                continue

            if not isinstance(function_tool, llm.FunctionTool | llm.RawFunctionTool):
                logger.error(
                    f"unknown tool type: {type(function_tool)}",
                    extra={
                        "function": fnc_call.name,
                        "speech_id": speech_handle.id,
                    },
                )
                _tool_completed(
                    make_tool_output(
                        fnc_call=fnc_call,
                        output=None,
                        exception=ToolError(f"Unknown tool type for function: {fnc_call.name}"),
                    )
                )
                continue

            # parse up front so the executor doesn't repeat the work, and so
            # invalid JSON surfaces as a tool error instead of inside the lock.
            # parse_function_arguments adds json_repair fallback + chat-template
            # token cleanup for misbehaving open-weight models.
            json_args = fnc_call.arguments or "{}"
            try:
                raw_args = llm_utils.parse_function_arguments(json_args)
            except ValueError as e:
                logger.warning(
                    f"invalid arguments for AI function `{fnc_call.name}`: {e}",
                    extra={
                        "function": fnc_call.name,
                        "arguments": fnc_call.arguments,
                        "speech_id": speech_handle.id,
                    },
                )
                _tool_completed(
                    make_tool_output(
                        fnc_call=fnc_call,
                        output=None,
                        exception=ToolError(f"Error parsing arguments for `{fnc_call.name}`: {e}"),
                    )
                )
                continue

            # write canonical JSON back so subsequent LLM turns see valid JSON
            # even if the original was repaired
            canonical = json.dumps(raw_args, default=str)
            if canonical != json_args:
                fnc_call.arguments = canonical

            if not tool_output.first_tool_started_fut.done():
                tool_output.first_tool_started_fut.set_result(None)

            tool_execution_started_cb(fnc_call)
            try:
                mock_tools: dict[str, Callable] = _MockToolsContextVar.get({}).get(
                    type(session.current_agent), {}
                )
                mock = mock_tools.get(fnc_call.name)
                mocked = mock is not None

                run_ctx = RunContext(
                    session=session, speech_handle=speech_handle, function_call=fnc_call
                )

                logger.debug(
                    "executing mock tool" if mocked else "executing tool",
                    extra={
                        "function": fnc_call.name,
                        "arguments": fnc_call.arguments,
                        "speech_id": speech_handle.id,
                    },
                )

                executor = executor_by_name.get(fnc_call.name, activity._tool_executor)
                function_callable = functools.partial(
                    executor.execute,
                    tool=function_tool,
                    run_ctx=run_ctx,
                    raw_arguments=raw_args,
                    mock=mock,
                )

                @tracer.start_as_current_span("function_tool")
                async def _traceable_fnc_tool(
                    function_callable: Callable, fnc_call: llm.FunctionCall
                ) -> None:
                    current_span = trace.get_current_span()
                    current_span.set_attributes(
                        {
                            trace_types.ATTR_FUNCTION_TOOL_ID: fnc_call.call_id,
                            trace_types.ATTR_FUNCTION_TOOL_NAME: fnc_call.name,
                            trace_types.ATTR_FUNCTION_TOOL_ARGS: fnc_call.arguments,
                        }
                    )

                    try:
                        val = await function_callable()
                        output = make_tool_output(fnc_call=fnc_call, output=val, exception=None)
                    except BaseException as e:
                        if isinstance(e, ToolError):
                            logger.warning(
                                "ToolError while executing tool: %s",
                                e.message,
                                extra={
                                    "function": fnc_call.name,
                                    "speech_id": speech_handle.id,
                                },
                            )
                        elif not isinstance(e, StopResponse):
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

                task = asyncio.create_task(
                    _traceable_fnc_tool(function_callable, fnc_call),
                    name=f"func_exec_{fnc_call.name}",  # task name is used for logging when the task is cancelled
                )
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

    if isinstance(output, BaseException):
        exception = output
        output = None

    if exception is not None:
        base_result = llm_utils.make_function_call_output(
            fnc_call=fnc_call, output=None, exception=exception
        )
        return ToolExecutionOutput(
            fnc_call=fnc_call.model_copy(),
            fnc_call_out=base_result.fnc_call_out,
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
                extra={"call_id": fnc_call.call_id, "output": output},
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

    base_result = llm_utils.make_function_call_output(
        fnc_call=fnc_call, output=fnc_out, exception=None
    )

    return ToolExecutionOutput(
        fnc_call=fnc_call.model_copy(),
        fnc_call_out=base_result.fnc_call_out,
        reply_required=fnc_out is not None,  # require a reply if the tool returned an output
        agent_task=task,
        raw_output=output,
        raw_exception=exception,
    )


INSTRUCTIONS_MESSAGE_ID = "lk.agent_task.instructions"  #  value must not change
"""
The ID of the instructions message in the chat context. (only for stateless LLMs)
"""


def update_instructions(
    chat_ctx: ChatContext,
    *,
    instructions: str | Instructions,
    add_if_missing: bool,
    modality: Literal["audio", "text"] = "audio",
) -> None:
    """
    Update the instruction message in the chat context or insert a new one if missing.

    Instructions are resolved to a plain string using the given modality before storage.
    """
    text = (
        instructions.render(modality=modality)
        if isinstance(instructions, Instructions)
        else instructions
    )

    idx = chat_ctx.index_by_id(INSTRUCTIONS_MESSAGE_ID)
    if idx is not None:
        if chat_ctx.items[idx].type == "message":
            chat_ctx.items[idx] = llm.ChatMessage(
                id=INSTRUCTIONS_MESSAGE_ID,
                role="system",
                content=[text],
                created_at=chat_ctx.items[idx].created_at,
            )
        else:
            raise ValueError(
                "expected the instructions inside the chat_ctx to be of type 'message'"
            )
    elif add_if_missing:
        chat_ctx.items.insert(
            0,
            llm.ChatMessage(id=INSTRUCTIONS_MESSAGE_ID, role="system", content=[text]),
        )


def remove_instructions(chat_ctx: ChatContext) -> None:
    # loop in case there are items with the same id (shouldn't happen!)
    while True:
        if msg := chat_ctx.get_by_id(INSTRUCTIONS_MESSAGE_ID):
            chat_ctx.items.remove(msg)
        else:
            break
