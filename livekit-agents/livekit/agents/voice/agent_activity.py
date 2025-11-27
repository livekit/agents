from __future__ import annotations

import asyncio
import contextvars
import heapq
import json
import time
from collections.abc import AsyncIterable, Coroutine, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union, cast

from opentelemetry import context as otel_context, trace

from livekit import rtc
from livekit.agents.llm.realtime import MessageGeneration
from livekit.agents.metrics.base import Metadata

from .. import llm, stt, tts, utils, vad
from ..llm.tool_context import (
    StopResponse,
    ToolFlag,
    _FunctionToolInfo,
    _RawFunctionToolInfo,
    get_function_info,
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)
from ..log import logger
from ..metrics import (
    EOUMetrics,
    LLMMetrics,
    RealtimeModelMetrics,
    STTMetrics,
    TTSMetrics,
    VADMetrics,
)
from ..telemetry import trace_types, tracer, utils as trace_utils
from ..tokenize.basic import split_words
from ..types import NOT_GIVEN, FlushSentinel, NotGivenOr
from ..utils.misc import is_given
from ._utils import _set_participant_attributes
from .agent import (
    Agent,
    ModelSettings,
    _get_activity_task_info,
    _set_activity_task_info,
)
from .audio_recognition import (
    AudioRecognition,
    RecognitionHooks,
    TurnDetectionMode,
    _EndOfTurnInfo,
    _PreemptiveGenerationInfo,
)
from .events import (
    AgentFalseInterruptionEvent,
    ErrorEvent,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
)
from .generation import (
    ToolExecutionOutput,
    _AudioOutput,
    _TextOutput,
    _TTSGenerationData,
    perform_audio_forwarding,
    perform_llm_inference,
    perform_text_forwarding,
    perform_tool_executions,
    perform_tts_inference,
    remove_instructions,
    update_instructions,
)
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from ..llm import mcp
    from .agent_session import AgentSession

_AgentActivityContextVar = contextvars.ContextVar["AgentActivity"]("agents_activity")
_SpeechHandleContextVar = contextvars.ContextVar["SpeechHandle"]("agents_speech_handle")


@dataclass
class _OnEnterData:
    session: AgentSession
    agent: Agent


_OnEnterContextVar = contextvars.ContextVar["_OnEnterData"]("agents_activity_on_enter")


@dataclass
class _PreemptiveGeneration:
    speech_handle: SpeechHandle
    user_message: llm.ChatMessage
    info: _PreemptiveGenerationInfo
    chat_ctx: llm.ChatContext
    tools: list[llm.FunctionTool | llm.RawFunctionTool]
    tool_choice: llm.ToolChoice | None
    created_at: float


# NOTE: AgentActivity isn't exposed to the public API
class AgentActivity(RecognitionHooks):
    def __init__(self, agent: Agent, sess: AgentSession) -> None:
        self._agent, self._session = agent, sess
        self._rt_session: llm.RealtimeSession | None = None
        self._realtime_spans: utils.BoundedDict[str, trace.Span] | None = None
        self._audio_recognition: AudioRecognition | None = None
        self._lock = asyncio.Lock()
        self._tool_choice: llm.ToolChoice | None = None

        self._started = False
        self._closed = False
        self._scheduling_paused = True

        self._current_speech: SpeechHandle | None = None
        self._speech_q: list[tuple[int, float, SpeechHandle]] = []

        # for false interruption handling
        self._paused_speech: SpeechHandle | None = None
        self._false_interruption_timer: asyncio.TimerHandle | None = None
        self._interrupt_paused_speech_task: asyncio.Task[None] | None = None

        # fired when a speech_task finishes or when a new speech_handle is scheduled
        # this is used to wake up the main task when the scheduling state changes
        self._q_updated = asyncio.Event()

        self._scheduling_atask: asyncio.Task[None] | None = None
        self._user_turn_completed_atask: asyncio.Task[None] | None = None
        self._speech_tasks: list[asyncio.Task[Any]] = []

        self._preemptive_generation: _PreemptiveGeneration | None = None

        self._drain_blocked_tasks: list[asyncio.Task[Any]] = []
        self._mcp_tools: list[mcp.MCPTool] = []

        self._on_enter_task: asyncio.Task | None = None
        self._on_exit_task: asyncio.Task | None = None

        if (
            isinstance(self.llm, llm.RealtimeModel)
            and self.llm.capabilities.turn_detection
            and not self.allow_interruptions
        ):
            raise ValueError(
                "the RealtimeModel uses a server-side turn detection, "
                "allow_interruptions cannot be False, disable turn_detection in "
                "the RealtimeModel and use VAD on the AgentSession instead"
            )

        # validate turn detection mode and turn detector
        turn_detection = (
            cast(Optional[TurnDetectionMode], self._agent.turn_detection)
            if is_given(self._agent.turn_detection)
            else self._session.turn_detection
        )
        self._turn_detection = self._validate_turn_detection(turn_detection)

        # speeches that audio playout finished but not done because of tool calls
        self._background_speeches: set[SpeechHandle] = set()

    def _validate_turn_detection(
        self, turn_detection: TurnDetectionMode | None
    ) -> TurnDetectionMode | None:
        if turn_detection is not None and not isinstance(turn_detection, str):
            # return directly if turn_detection is _TurnDetector
            return turn_detection

        mode = turn_detection if isinstance(turn_detection, str) else None
        vad_model = self.vad
        stt_model = self.stt
        llm_model = self.llm

        if mode == "vad" and not vad_model:
            logger.warning("turn_detection is set to 'vad', but no VAD model is provided")
            mode = None

        if mode == "stt" and not stt_model:
            logger.warning(
                "turn_detection is set to 'stt', but no STT model is provided, "
                "ignoring the turn_detection setting"
            )
            mode = None

        if isinstance(llm_model, llm.RealtimeModel):
            if mode == "realtime_llm" and not llm_model.capabilities.turn_detection:
                logger.warning(
                    "turn_detection is set to 'realtime_llm', but the LLM is not a RealtimeModel "
                    "or the server-side turn detection is not supported/enabled, "
                    "ignoring the turn_detection setting"
                )
                mode = None

            if mode == "stt":
                logger.warning(
                    "turn_detection is set to 'stt', but the LLM is a RealtimeModel, "
                    "ignoring the turn_detection setting"
                )
                mode = None

            elif mode and mode != "realtime_llm" and llm_model.capabilities.turn_detection:
                logger.warning(
                    f"turn_detection is set to '{mode}', but the LLM "
                    "is a RealtimeModel and server-side turn detection enabled, "
                    "ignoring the turn_detection setting"
                )
                mode = None

            # fallback to VAD if server side turn detection is disabled and VAD is available
            if not llm_model.capabilities.turn_detection and vad_model and mode is None:
                mode = "vad"

        elif mode == "realtime_llm":
            logger.warning(
                "turn_detection is set to 'realtime_llm', but the LLM is not a RealtimeModel"
            )
            mode = None

        if (
            not vad_model
            and stt_model
            and not stt_model.capabilities.streaming
            and isinstance(llm_model, llm.LLM)
            and self.allow_interruptions
            and mode is None
        ):
            logger.warning(
                "VAD is not set. Enabling VAD is recommended when using LLM and non-streaming STT "
                "for more responsive interruption handling."
            )

        return mode

    @property
    def scheduling_paused(self) -> bool:
        return self._scheduling_paused

    @property
    def session(self) -> AgentSession:
        return self._session

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def mcp_servers(self) -> list[mcp.MCPServer] | None:
        return (
            self._agent.mcp_servers
            if is_given(self._agent.mcp_servers)
            else self._session.mcp_servers
        )

    @property
    def allow_interruptions(self) -> bool:
        return (
            self._agent.allow_interruptions
            if is_given(self._agent.allow_interruptions)
            else self._session.options.allow_interruptions
        )

    @property
    def min_endpointing_delay(self) -> float:
        return (
            self._agent.min_endpointing_delay
            if is_given(self._agent.min_endpointing_delay)
            else self._session.options.min_endpointing_delay
        )

    @property
    def max_endpointing_delay(self) -> float:
        return (
            self._agent.max_endpointing_delay
            if is_given(self._agent.max_endpointing_delay)
            else self._session.options.max_endpointing_delay
        )

    @property
    def realtime_llm_session(self) -> llm.RealtimeSession | None:
        return self._rt_session

    @property
    def current_speech(self) -> SpeechHandle | None:
        return self._current_speech

    @property
    def tools(self) -> list[llm.FunctionTool | llm.RawFunctionTool | mcp.MCPTool]:
        return self._session.tools + self._agent.tools + self._mcp_tools  # type: ignore

    @property
    def min_consecutive_speech_delay(self) -> float:
        return (
            self._agent.min_consecutive_speech_delay
            if is_given(self._agent.min_consecutive_speech_delay)
            else self._session.options.min_consecutive_speech_delay
        )

    @property
    def use_tts_aligned_transcript(self) -> bool:
        use_aligned_transcript = (
            self._agent.use_tts_aligned_transcript
            if is_given(self._agent.use_tts_aligned_transcript)
            else self._session.options.use_tts_aligned_transcript
        )

        return use_aligned_transcript is True

    async def update_instructions(self, instructions: str) -> None:
        self._agent._instructions = instructions

        if self._rt_session is not None:
            await self._rt_session.update_instructions(instructions)
        else:
            update_instructions(
                self._agent._chat_ctx, instructions=instructions, add_if_missing=True
            )

    async def update_tools(self, tools: list[llm.FunctionTool | llm.RawFunctionTool]) -> None:
        tools = list(set(tools))
        self._agent._tools = tools

        if self._rt_session is not None:
            await self._rt_session.update_tools(tools)

        if isinstance(self.llm, llm.LLM):
            # for realtime LLM, we assume the server will remove unvalid tool messages
            await self.update_chat_ctx(self._agent._chat_ctx.copy(tools=tools))

    async def update_chat_ctx(
        self, chat_ctx: llm.ChatContext, *, exclude_invalid_function_calls: bool = True
    ) -> None:
        chat_ctx = chat_ctx.copy(tools=self.tools if exclude_invalid_function_calls else NOT_GIVEN)
        self._agent._chat_ctx = chat_ctx

        if self._rt_session is not None:
            remove_instructions(chat_ctx)
            await self._rt_session.update_chat_ctx(chat_ctx)
        else:
            update_instructions(
                chat_ctx, instructions=self._agent.instructions, add_if_missing=True
            )

    def update_options(
        self,
        *,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        min_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
        max_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
    ) -> None:
        if utils.is_given(tool_choice):
            self._tool_choice = cast(Optional[llm.ToolChoice], tool_choice)

        if self._rt_session is not None:
            self._rt_session.update_options(tool_choice=self._tool_choice)

        if utils.is_given(turn_detection):
            turn_detection = self._validate_turn_detection(
                cast(Optional[TurnDetectionMode], turn_detection)
            )

            if (
                self._turn_detection == "manual" or turn_detection == "manual"
            ) and self._false_interruption_timer is not None:
                self._false_interruption_timer.cancel()
                self._false_interruption_timer = None

        if self._audio_recognition:
            self._audio_recognition.update_options(
                min_endpointing_delay=min_endpointing_delay,
                max_endpointing_delay=max_endpointing_delay,
                turn_detection=turn_detection,
            )

    def _create_speech_task(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        speech_handle: SpeechHandle | None = None,
        name: str | None = None,
    ) -> asyncio.Task[Any]:
        """
        This method must only be used for tasks that "could" create a new SpeechHandle.
        When draining, every task created with this method will be awaited.
        """
        # https://github.com/python/cpython/pull/31837 alternative impl
        tk = _AgentActivityContextVar.set(self)
        tk1 = None
        if speech_handle is not None:
            tk1 = _SpeechHandleContextVar.set(speech_handle)

        # Capture the current OpenTelemetry context to ensure proper span nesting
        current_context = otel_context.get_current()

        # Create a wrapper coroutine that runs in the captured context
        async def _context_aware_coro() -> Any:
            # Attach the captured context before running the original coroutine
            token = otel_context.attach(current_context)
            try:
                return await coro
            finally:
                otel_context.detach(token)

        task = asyncio.create_task(_context_aware_coro(), name=name)
        self._speech_tasks.append(task)
        task.add_done_callback(lambda _: self._speech_tasks.remove(task))

        _set_activity_task_info(task, speech_handle=speech_handle)

        if speech_handle is not None:
            # mark a speech_handle as done, if every "linked" tasks are done
            speech_handle._tasks.append(task)

            def _mark_done_if_needed(_: asyncio.Task) -> None:
                if all(task.done() for task in speech_handle._tasks):
                    speech_handle._mark_done()

            task.add_done_callback(_mark_done_if_needed)

        task.add_done_callback(lambda _: self._wake_up_scheduling_task())
        _AgentActivityContextVar.reset(tk)

        if tk1:
            _SpeechHandleContextVar.reset(tk1)

        return task

    async def start(self) -> None:
        # `start` must only be called by AgentSession

        async with self._lock:
            if self._started:
                return

            start_span = tracer.start_span(
                "start_agent_activity",
                attributes={trace_types.ATTR_AGENT_LABEL: self.agent.label},
            )
            try:
                self._agent._activity = self

                with trace.use_span(start_span, end_on_exit=False):
                    if isinstance(self.llm, llm.LLM):
                        self.llm.prewarm()

                    if isinstance(self.stt, stt.STT):
                        self.stt.prewarm()

                    if isinstance(self.tts, tts.TTS):
                        self.tts.prewarm()

                # don't use start_span for _start_session, avoid nested user/assistant turns
                await self._start_session()
                self._started = True

                @tracer.start_as_current_span(
                    "on_enter",
                    context=trace.set_span_in_context(start_span),
                    attributes={trace_types.ATTR_AGENT_LABEL: self._agent.label},
                )
                @utils.log_exceptions(logger=logger)
                async def _traceable_on_enter() -> None:
                    data = _OnEnterData(session=self._session, agent=self._agent)
                    try:
                        tk = _OnEnterContextVar.set(data)
                        await self._agent.on_enter()
                    finally:
                        _OnEnterContextVar.reset(tk)

                self._on_enter_task = task = self._create_speech_task(
                    _traceable_on_enter(), name="AgentTask_on_enter"
                )
                _set_activity_task_info(task, inline_task=True)
            finally:
                start_span.end()

    async def _start_session(self) -> None:
        assert self._lock.locked(), "_start_session should only be used when locked."

        if isinstance(self.llm, llm.LLM):
            self.llm.on("metrics_collected", self._on_metrics_collected)
            self.llm.on("error", self._on_error)

        if isinstance(self.stt, stt.STT):
            self.stt.on("metrics_collected", self._on_metrics_collected)
            self.stt.on("error", self._on_error)

        if isinstance(self.tts, tts.TTS):
            self.tts.on("metrics_collected", self._on_metrics_collected)
            self.tts.on("error", self._on_error)

        if isinstance(self.vad, vad.VAD):
            self.vad.on("metrics_collected", self._on_metrics_collected)

        if self.mcp_servers:

            @utils.log_exceptions(logger=logger)
            async def _list_mcp_tools_task(
                mcp_server: mcp.MCPServer,
            ) -> list[mcp.MCPTool]:
                if not mcp_server.initialized:
                    await mcp_server.initialize()

                return await mcp_server.list_tools()

            gathered = await asyncio.gather(
                *(_list_mcp_tools_task(s) for s in self.mcp_servers),
                return_exceptions=True,
            )
            tools: list[mcp.MCPTool] = []
            for mcp_server, res in zip(self.mcp_servers, gathered):
                if isinstance(res, BaseException):
                    logger.error(
                        f"failed to list tools from MCP server {mcp_server}",
                        exc_info=res,
                    )
                    continue

                tools.extend(res)

            self._mcp_tools = tools

        if isinstance(self.llm, llm.RealtimeModel):
            self._rt_session = self.llm.session()
            self._rt_session.on("generation_created", self._on_generation_created)
            self._rt_session.on("input_speech_started", self._on_input_speech_started)
            self._rt_session.on("input_speech_stopped", self._on_input_speech_stopped)
            self._rt_session.on(
                "input_audio_transcription_completed",
                self._on_input_audio_transcription_completed,
            )
            self._rt_session.on("metrics_collected", self._on_metrics_collected)
            self._rt_session.on("error", self._on_error)

            remove_instructions(self._agent._chat_ctx)

            try:
                await self._rt_session.update_instructions(self._agent.instructions)
            except llm.RealtimeError:
                logger.exception("failed to update the instructions")

            try:
                await self._rt_session.update_chat_ctx(self._agent.chat_ctx)
            except llm.RealtimeError:
                logger.exception("failed to update the chat_ctx")

            try:
                await self._rt_session.update_tools(self.tools)
            except llm.RealtimeError:
                logger.exception("failed to update the tools")

            self._realtime_spans = utils.BoundedDict[str, trace.Span](maxsize=100)
            if (
                not self.llm.capabilities.audio_output
                and not self.tts
                and self._session.output.audio
            ):
                logger.error(
                    "audio output is enabled but RealtimeModel has no audio modality "
                    "and no TTS is set. Either enable audio modality in the RealtimeModel "
                    "or set a TTS model."
                )

        elif isinstance(self.llm, llm.LLM):
            try:
                update_instructions(
                    self._agent._chat_ctx,
                    instructions=self._agent.instructions,
                    add_if_missing=True,
                )
            except ValueError:
                logger.exception("failed to update the instructions")

        await self._resume_scheduling_task()
        self._audio_recognition = AudioRecognition(
            self._session,
            hooks=self,
            stt=self._agent.stt_node if self.stt else None,
            vad=self.vad,
            min_endpointing_delay=self.min_endpointing_delay,
            max_endpointing_delay=self.max_endpointing_delay,
            turn_detection=self._turn_detection,
        )
        self._audio_recognition.start()

    @tracer.start_as_current_span("drain_agent_activity")
    async def drain(self) -> None:
        # `drain` must only be called by AgentSession
        # AgentSession makes sure there is always one agent available to the users.
        current_span = trace.get_current_span()
        current_span.set_attribute(trace_types.ATTR_AGENT_LABEL, self._agent.label)

        @tracer.start_as_current_span(
            "on_exit", attributes={trace_types.ATTR_AGENT_LABEL: self._agent.label}
        )
        @utils.log_exceptions(logger=logger)
        async def _traceable_on_exit() -> None:
            await self._agent.on_exit()

        async with self._lock:
            self._on_exit_task = task = self._create_speech_task(
                _traceable_on_exit(), name="AgentTask_on_exit"
            )
            _set_activity_task_info(task, inline_task=True)

            self._cancel_preemptive_generation()

            await self._on_exit_task
            await self._pause_scheduling_task()

    async def _pause_scheduling_task(
        self, *, blocked_tasks: list[asyncio.Task] | None = None
    ) -> None:
        assert self._lock.locked(), "_finalize_main_task should only be used when locked."

        if self._scheduling_paused:
            return

        self._scheduling_paused = True
        self._drain_blocked_tasks = blocked_tasks or []
        self._wake_up_scheduling_task()

        if self._scheduling_atask is not None:
            # When pausing/draining, we ensure that all speech_tasks complete fully.
            # This means that even if the SpeechHandle themselves have finished,
            # we still wait for the entire execution (e.g function_tools)
            await asyncio.shield(self._scheduling_atask)

    async def _resume_scheduling_task(self) -> None:
        assert self._lock.locked(), "_finalize_main_task should only be used when locked."

        if not self._scheduling_paused:
            return

        self._scheduling_paused = False
        self._scheduling_atask = asyncio.create_task(
            self._scheduling_task(), name="_scheduling_task"
        )

    async def resume(self) -> None:
        # `resume` must only be called by AgentSession

        async with self._lock:
            span = tracer.start_span(
                "resume_agent_activity",
                attributes={trace_types.ATTR_AGENT_LABEL: self.agent.label},
            )
            try:
                await self._start_session()
            finally:
                span.end()

    def _wake_up_scheduling_task(self) -> None:
        self._q_updated.set()

    async def pause(self, *, blocked_tasks: list[asyncio.Task]) -> None:
        # `pause` must only be called by AgentSession

        # When draining, the tasks that have done the "premption" must be ignored.
        # They will most likely block until the Agent transition is done. So we must not
        # wait for them to avoid deadlocks.

        # When resuming, the AgentSession.update_agent must use the same AgentActivity instance!
        async with self._lock:
            span = tracer.start_span(
                "pause_agent_activity",
                attributes={trace_types.ATTR_AGENT_LABEL: self._agent.label},
            )
            try:
                await self._pause_scheduling_task(blocked_tasks=blocked_tasks)
                await self._close_session()
            finally:
                span.end()

    async def _close_session(self) -> None:
        assert self._lock.locked(), "_close_session should only be used when locked."

        if isinstance(self.llm, llm.LLM):
            self.llm.off("metrics_collected", self._on_metrics_collected)
            self.llm.off("error", self._on_error)

        if isinstance(self.llm, llm.RealtimeModel) and self._rt_session is not None:
            self._rt_session.off("generation_created", self._on_generation_created)
            self._rt_session.off("input_speech_started", self._on_input_speech_started)
            self._rt_session.off("input_speech_stopped", self._on_input_speech_stopped)
            self._rt_session.off(
                "input_audio_transcription_completed",
                self._on_input_audio_transcription_completed,
            )
            self._rt_session.off("metrics_collected", self._on_metrics_collected)
            self._rt_session.off("error", self._on_error)

        if isinstance(self.stt, stt.STT):
            self.stt.off("metrics_collected", self._on_metrics_collected)
            self.stt.off("error", self._on_error)

        if isinstance(self.tts, tts.TTS):
            self.tts.off("metrics_collected", self._on_metrics_collected)
            self.tts.off("error", self._on_error)

        if isinstance(self.vad, vad.VAD):
            self.vad.off("metrics_collected", self._on_metrics_collected)

        if self._rt_session is not None:
            await self._rt_session.aclose()

        if self._realtime_spans is not None:
            self._realtime_spans.clear()

        if self._audio_recognition is not None:
            await self._audio_recognition.aclose()

        await self._interrupt_paused_speech(old_task=self._interrupt_paused_speech_task)
        self._interrupt_paused_speech_task = None

    async def aclose(self) -> None:
        # `aclose` must only be called by AgentSession

        async with self._lock:
            if self._closed:
                return

            self._closed = True
            self._cancel_preemptive_generation()

            await self._close_session()
            await asyncio.gather(*self._interrupt_background_speeches(force=False))

            if self._scheduling_atask is not None:
                await utils.aio.cancel_and_wait(self._scheduling_atask)

            self._agent._activity = None

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if not self._started:
            return

        if (
            self._current_speech
            and not self._current_speech.allow_interruptions
            and self._session.options.discard_audio_if_uninterruptible
        ):
            # discard the audio if the current speech is not interruptable
            return

        if self._rt_session is not None:
            self._rt_session.push_audio(frame)

        if self._audio_recognition is not None:
            self._audio_recognition.push_audio(frame)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        if not self._started:
            return

        if self._rt_session is not None:
            self._rt_session.push_video(frame)

    def say(
        self,
        text: str | AsyncIterable[str],
        *,
        audio: NotGivenOr[AsyncIterable[rtc.AudioFrame]] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        add_to_chat_ctx: bool = True,
    ) -> SpeechHandle:
        if (
            not is_given(audio)
            and not self.tts
            and self._session.output.audio
            and self._session.output.audio_enabled
        ):
            raise RuntimeError("trying to generate speech from text without a TTS model")

        if (
            isinstance(self.llm, llm.RealtimeModel)
            and self.llm.capabilities.turn_detection
            and allow_interruptions is False
        ):
            logger.warning(
                "the RealtimeModel uses a server-side turn detection, allow_interruptions cannot be False when using VoiceAgent.say(), "  # noqa: E501
                "disable turn_detection in the RealtimeModel and use VAD on the AgentTask/VoiceAgent instead"  # noqa: E501
            )
            allow_interruptions = NOT_GIVEN

        handle = SpeechHandle.create(
            allow_interruptions=allow_interruptions
            if is_given(allow_interruptions)
            else self.allow_interruptions
        )
        self._session.emit(
            "speech_created",
            SpeechCreatedEvent(speech_handle=handle, user_initiated=True, source="say"),
        )

        task = self._create_speech_task(
            self._tts_task(
                speech_handle=handle,
                text=text,
                audio=audio or None,
                add_to_chat_ctx=add_to_chat_ctx,
                model_settings=ModelSettings(),
            ),
            speech_handle=handle,
            name="AgentActivity.tts_say",
        )
        task.add_done_callback(self._on_pipeline_reply_done)
        self._schedule_speech(handle, SpeechHandle.SPEECH_PRIORITY_NORMAL)
        return handle

    def _generate_reply(
        self,
        *,
        user_message: NotGivenOr[llm.ChatMessage | None] = NOT_GIVEN,
        chat_ctx: NotGivenOr[llm.ChatContext | None] = NOT_GIVEN,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        schedule_speech: bool = True,
    ) -> SpeechHandle:
        if (
            isinstance(self.llm, llm.RealtimeModel)
            and self.llm.capabilities.turn_detection
            and allow_interruptions is False
        ):
            logger.warning(
                "the RealtimeModel uses a server-side turn detection, allow_interruptions cannot be False when using VoiceAgent.generate_reply(), "  # noqa: E501
                "disable turn_detection in the RealtimeModel and use VAD on the AgentTask/VoiceAgent instead"  # noqa: E501
            )
            allow_interruptions = NOT_GIVEN

        if self.llm is None:
            raise RuntimeError("trying to generate reply without an LLM model")

        task = asyncio.current_task()
        if not is_given(tool_choice) and task is not None:
            if task_info := _get_activity_task_info(task):
                if task_info.function_call is not None:
                    # when generate_reply is called inside a function_tool, set tool_choice to None by default  # noqa: E501
                    tool_choice = "none"

        tools = self.tools

        # if tool has the IGNORE_ON_ENTER flag, every generate_reply inside on_enter will ignore it
        if on_enter_data := _OnEnterContextVar.get(None):
            if on_enter_data.agent == self._agent and on_enter_data.session == self._session:
                filtered_tools = []
                for tool in tools:
                    info: _RawFunctionToolInfo | _FunctionToolInfo
                    if is_raw_function_tool(tool):
                        info = get_raw_function_info(tool)
                    elif is_function_tool(tool):
                        info = get_function_info(tool)

                    if not info or (info.flags & ToolFlag.IGNORE_ON_ENTER):
                        continue

                    filtered_tools.append(tool)

                tools = filtered_tools

        handle = SpeechHandle.create(
            allow_interruptions=allow_interruptions
            if is_given(allow_interruptions)
            else self.allow_interruptions,
        )
        self._session.emit(
            "speech_created",
            SpeechCreatedEvent(speech_handle=handle, user_initiated=True, source="generate_reply"),
        )

        if isinstance(self.llm, llm.RealtimeModel):
            self._create_speech_task(
                self._realtime_reply_task(
                    speech_handle=handle,
                    # TODO(theomonnom): support llm.ChatMessage for the realtime model
                    user_input=user_message.text_content if user_message else None,
                    instructions=instructions or None,
                    # TODO(theomonnom): the list of tools should always be passed here
                    model_settings=ModelSettings(tool_choice=tool_choice),
                ),
                speech_handle=handle,
                name="AgentActivity.realtime_reply",
            )

        elif isinstance(self.llm, llm.LLM):
            # instructions used inside generate_reply are "extra" instructions.
            # this matches the behavior of the Realtime API:
            # https://platform.openai.com/docs/api-reference/realtime-client-events/response/create
            if instructions:
                instructions = "\n".join([self._agent.instructions, instructions])

            task = self._create_speech_task(
                self._pipeline_reply_task(
                    speech_handle=handle,
                    chat_ctx=chat_ctx or self._agent._chat_ctx,
                    tools=tools,
                    new_message=user_message if is_given(user_message) else None,
                    instructions=instructions or None,
                    model_settings=ModelSettings(
                        tool_choice=tool_choice
                        if utils.is_given(tool_choice) or self._tool_choice is None
                        else self._tool_choice
                    ),
                ),
                speech_handle=handle,
                name="AgentActivity.pipeline_reply",
            )
            task.add_done_callback(self._on_pipeline_reply_done)

        if schedule_speech:
            self._schedule_speech(handle, SpeechHandle.SPEECH_PRIORITY_NORMAL)

        return handle

    def _cancel_preemptive_generation(self) -> None:
        if self._preemptive_generation is not None:
            self._preemptive_generation.speech_handle._cancel()
            self._preemptive_generation = None

    def _interrupt_background_speeches(self, force: bool = False) -> list[SpeechHandle]:
        interrupted_speeches: list[SpeechHandle] = []
        for speech in self._background_speeches:
            if force or speech.allow_interruptions:
                interrupted_speeches.append(speech.interrupt(force=force))

        return interrupted_speeches

    def interrupt(self, *, force: bool = False) -> asyncio.Future[None]:
        """Interrupt the current speech generation and any queued speeches.

        Returns:
            An asyncio.Future that completes when the interruption is fully processed
            and chat context has been updated
        """
        self._cancel_preemptive_generation()

        future = asyncio.Future[None]()

        interrupted_speeches = self._interrupt_background_speeches(force=force)

        if self._current_speech is not None:
            self._current_speech.interrupt(force=force)
            interrupted_speeches.append(self._current_speech)

        for _, _, speech in self._speech_q:
            speech.interrupt(force=force)
            interrupted_speeches.append(speech)

        if self._rt_session is not None:
            self._rt_session.interrupt()

        if not interrupted_speeches:
            future.set_result(None)
        else:

            def on_playout_done(_: SpeechHandle) -> None:
                if not future.done() and all(speech.done() for speech in interrupted_speeches):
                    future.set_result(None)

            for speech in interrupted_speeches:
                speech.add_done_callback(on_playout_done)

        return future

    def clear_user_turn(self) -> None:
        if self._audio_recognition:
            self._audio_recognition.clear_user_turn()

        if self._rt_session is not None:
            self._rt_session.clear_audio()

    def commit_user_turn(self, *, transcript_timeout: float, stt_flush_duration: float) -> None:
        assert self._audio_recognition is not None
        self._audio_recognition.commit_user_turn(
            audio_detached=not self._session.input.audio_enabled,
            transcript_timeout=transcript_timeout,
            stt_flush_duration=stt_flush_duration,
        )

    def _schedule_speech(self, speech: SpeechHandle, priority: int, force: bool = False) -> None:
        # when force=True, we still allow to schedule a new speech even if
        # `pause_speech_scheduling` is waiting for the schedule_task to drain.
        # This allows for tool responses to be generated before the AgentActivity is finalized.

        if self._scheduling_paused and not force:
            raise RuntimeError(
                "cannot schedule new speech, the speech scheduling is draining/pausing"
            )

        if self._scheduling_atask and self._scheduling_atask.done():
            logger.warning(
                "attempting to schedule a new SpeechHandle, but the scheduling_task is not running."
            )
            return

        while True:
            try:
                # negate the priority to make it a max heap
                heapq.heappush(self._speech_q, (-priority, time.perf_counter_ns(), speech))
                break
            except TypeError:
                # handle TypeError when identical timestamps cause speech comparison failure
                # with perf_counter_ns(), collisions should be rare
                pass

        speech._mark_scheduled()
        self._wake_up_scheduling_task()

    @utils.log_exceptions(logger=logger)
    async def _scheduling_task(self) -> None:
        last_playout_ts = 0.0
        while True:
            await self._q_updated.wait()
            while self._speech_q:
                _, _, speech = heapq.heappop(self._speech_q)
                if speech.done():
                    # skip done speech (interrupted when it's in the queue)
                    self._current_speech = None
                    continue
                self._current_speech = speech
                if self.min_consecutive_speech_delay > 0.0:
                    await asyncio.sleep(
                        self.min_consecutive_speech_delay - (time.time() - last_playout_ts)
                    )
                    # check again if speech is done after sleep delay
                    if speech.done():
                        # skip done speech (interrupted during delay)
                        self._current_speech = None
                        continue
                speech._authorize_generation()
                await speech._wait_for_generation()
                self._current_speech = None
                last_playout_ts = time.time()

            # if we're draining/pasuing and there are no more speech tasks, we can exit.
            # only speech tasks can bypass draining to create a tool response (see `_schedule_speech`)  # noqa: E501

            blocked_handles: list[SpeechHandle] = []
            for task in self._drain_blocked_tasks:
                info = _get_activity_task_info(task)
                if not info:
                    logger.error("blocked task without activity info; skipping.")
                    continue

                if not info.speech_handle:
                    continue  # on_enter/on_exit

                blocked_handles.append(info.speech_handle)

            to_wait: list[asyncio.Task] = []
            for task in self._speech_tasks:
                if task in self._drain_blocked_tasks:
                    continue

                info = _get_activity_task_info(task)
                if info and info.speech_handle in blocked_handles:
                    continue

                to_wait.append(task)

            if self._scheduling_paused and len(to_wait) == 0:
                break

            self._q_updated.clear()

    # -- Realtime Session events --

    def _on_metrics_collected(
        self,
        ev: STTMetrics | TTSMetrics | VADMetrics | LLMMetrics | RealtimeModelMetrics,
    ) -> None:
        if (speech_handle := _SpeechHandleContextVar.get(None)) and (
            isinstance(ev, LLMMetrics) or isinstance(ev, TTSMetrics)
        ):
            ev.speech_id = speech_handle.id
        if (
            isinstance(ev, RealtimeModelMetrics)
            and self._realtime_spans is not None
            and (realtime_span := self._realtime_spans.pop(ev.request_id, None))
        ):
            trace_utils.record_realtime_metrics(realtime_span, ev)
        self._session.emit("metrics_collected", MetricsCollectedEvent(metrics=ev))

    def _on_error(
        self, error: llm.LLMError | stt.STTError | tts.TTSError | llm.RealtimeModelError
    ) -> None:
        if isinstance(error, llm.LLMError):
            error_event = ErrorEvent(error=error, source=self.llm)
            self._session.emit("error", error_event)
        elif isinstance(error, llm.RealtimeModelError):
            error_event = ErrorEvent(error=error, source=self.llm)
            self._session.emit("error", error_event)
        elif isinstance(error, stt.STTError):
            error_event = ErrorEvent(error=error, source=self.stt)
            self._session.emit("error", error_event)
        elif isinstance(error, tts.TTSError):
            error_event = ErrorEvent(error=error, source=self.tts)
            self._session.emit("error", error_event)

        self._session._on_error(error)

    def _on_input_speech_started(self, _: llm.InputSpeechStartedEvent) -> None:
        if self.vad is None:
            self._session._update_user_state("speaking")

        # self.interrupt() is going to raise when allow_interruptions is False, llm.InputSpeechStartedEvent is only fired by the server when the turn_detection is enabled.  # noqa: E501
        # When using the server-side turn_detection, we don't allow allow_interruptions to be False.
        try:
            self.interrupt()  # input_speech_started is also interrupting on the serverside realtime session  # noqa: E501
        except RuntimeError:
            logger.exception(
                "RealtimeAPI input_speech_started, but current speech is not interruptable, this should never happen!"  # noqa: E501
            )

    def _on_input_speech_stopped(self, ev: llm.InputSpeechStoppedEvent) -> None:
        if self.vad is None:
            self._session._update_user_state("listening")

        if ev.user_transcription_enabled:
            self._session._user_input_transcribed(
                UserInputTranscribedEvent(transcript="", is_final=False)
            )

    def _on_input_audio_transcription_completed(self, ev: llm.InputTranscriptionCompleted) -> None:
        self._session._user_input_transcribed(
            UserInputTranscribedEvent(transcript=ev.transcript, is_final=ev.is_final)
        )

        if ev.is_final:
            # TODO: for realtime models, the created_at field is off. it should be set to when the user started speaking.
            # but we don't have that information here.
            msg = llm.ChatMessage(role="user", content=[ev.transcript], id=ev.item_id)
            self._agent._chat_ctx.items.append(msg)
            self._session._conversation_item_added(msg)

    def _on_generation_created(self, ev: llm.GenerationCreatedEvent) -> None:
        if ev.user_initiated:
            # user_initiated generations are directly handled inside _realtime_reply_task
            return

        if self._scheduling_paused:
            # TODO(theomonnom): should we "forward" this new turn to the next agent?
            logger.warning("skipping new realtime generation, the speech scheduling is not running")
            return

        handle = SpeechHandle.create(allow_interruptions=self.allow_interruptions)
        self._session.emit(
            "speech_created",
            SpeechCreatedEvent(speech_handle=handle, user_initiated=False, source="generate_reply"),
        )

        self._create_speech_task(
            self._realtime_generation_task(
                speech_handle=handle,
                generation_ev=ev,
                model_settings=ModelSettings(),
            ),
            speech_handle=handle,
            name="AgentActivity.realtime_generation",
        )
        self._schedule_speech(handle, SpeechHandle.SPEECH_PRIORITY_NORMAL)

    def _interrupt_by_audio_activity(self) -> None:
        opt = self._session.options
        use_pause = opt.resume_false_interruption and opt.false_interruption_timeout is not None

        if isinstance(self.llm, llm.RealtimeModel) and self.llm.capabilities.turn_detection:
            # ignore if realtime model has turn detection enabled
            return

        if (
            self.stt is not None
            and opt.min_interruption_words > 0
            and self._audio_recognition is not None
        ):
            text = self._audio_recognition.current_transcript

            # TODO(long): better word splitting for multi-language
            if len(split_words(text, split_character=True)) < opt.min_interruption_words:
                return

        if self._rt_session is not None:
            self._rt_session.start_user_activity()

        if (
            self._current_speech is not None
            and not self._current_speech.interrupted
            and self._current_speech.allow_interruptions
        ):
            self._paused_speech = self._current_speech

            # reset the false interruption timer
            if self._false_interruption_timer:
                self._false_interruption_timer.cancel()
                self._false_interruption_timer = None

            if use_pause and self._session.output.audio and self._session.output.audio.can_pause:
                self._session.output.audio.pause()
                self._session._update_agent_state("listening")
            else:
                if self._rt_session is not None:
                    self._rt_session.interrupt()

                self._current_speech.interrupt()

    # region recognition hooks

    def on_start_of_speech(self, ev: vad.VADEvent | None) -> None:
        self._session._update_user_state("speaking")

        if self._false_interruption_timer:
            # cancel the timer when user starts speaking but leave the paused state unchanged
            self._false_interruption_timer.cancel()
            self._false_interruption_timer = None

    def on_end_of_speech(self, ev: vad.VADEvent | None) -> None:
        speech_end_time = time.time()
        if ev:
            speech_end_time = speech_end_time - ev.silence_duration
        self._session._update_user_state(
            "listening",
            last_speaking_time=speech_end_time,
        )

        if (
            self._paused_speech
            and (timeout := self._session.options.false_interruption_timeout) is not None
        ):
            # schedule a resume timer when user stops speaking
            self._start_false_interruption_timer(timeout)

    def on_vad_inference_done(self, ev: vad.VADEvent) -> None:
        if self._turn_detection in ("manual", "realtime_llm"):
            # ignore vad inference done event if turn_detection is manual or realtime_llm
            return

        if ev.speech_duration >= self._session.options.min_interruption_duration:
            self._interrupt_by_audio_activity()

    def on_interim_transcript(self, ev: stt.SpeechEvent, *, speaking: bool | None) -> None:
        if isinstance(self.llm, llm.RealtimeModel) and self.llm.capabilities.user_transcription:
            # skip stt transcription if user_transcription is enabled on the realtime model
            return

        self._session._user_input_transcribed(
            UserInputTranscribedEvent(
                language=ev.alternatives[0].language,
                transcript=ev.alternatives[0].text,
                is_final=False,
                speaker_id=ev.alternatives[0].speaker_id,
            ),
        )

        if ev.alternatives[0].text and self._turn_detection not in (
            "manual",
            "realtime_llm",
        ):
            self._interrupt_by_audio_activity()

            if (
                speaking is False
                and self._paused_speech
                and (timeout := self._session.options.false_interruption_timeout) is not None
            ):
                # schedule a resume timer if interrupted after end_of_speech
                self._start_false_interruption_timer(timeout)

    def on_final_transcript(self, ev: stt.SpeechEvent, *, speaking: bool | None = None) -> None:
        if isinstance(self.llm, llm.RealtimeModel) and self.llm.capabilities.user_transcription:
            # skip stt transcription if user_transcription is enabled on the realtime model
            return

        self._session._user_input_transcribed(
            UserInputTranscribedEvent(
                language=ev.alternatives[0].language,
                transcript=ev.alternatives[0].text,
                is_final=True,
                speaker_id=ev.alternatives[0].speaker_id,
            ),
        )
        # agent speech might not be interrupted if VAD failed and a final transcript is received
        # we call _interrupt_by_audio_activity (idempotent) to pause the speech, if possible
        # which will also be immediately interrupted

        if self._audio_recognition and self._turn_detection not in (
            "manual",
            "realtime_llm",
        ):
            self._interrupt_by_audio_activity()

            if (
                speaking is False
                and self._paused_speech
                and (timeout := self._session.options.false_interruption_timeout) is not None
            ):
                # schedule a resume timer if interrupted after end_of_speech
                self._start_false_interruption_timer(timeout)

        self._interrupt_paused_speech_task = asyncio.create_task(
            self._interrupt_paused_speech(old_task=self._interrupt_paused_speech_task)
        )

    def on_preemptive_generation(self, info: _PreemptiveGenerationInfo) -> None:
        if (
            not self._session.options.preemptive_generation
            or self._scheduling_paused
            or (self._current_speech is not None and not self._current_speech.interrupted)
            or not isinstance(self.llm, llm.LLM)
        ):
            return

        self._cancel_preemptive_generation()

        user_message = llm.ChatMessage(
            role="user",
            content=[info.new_transcript],
            transcript_confidence=info.transcript_confidence,
        )

        chat_ctx = self._agent.chat_ctx.copy()
        speech_handle = self._generate_reply(
            # we need to send in the original user_message because metrics are injected later on
            user_message=user_message,
            chat_ctx=chat_ctx,
            schedule_speech=False,
        )

        self._preemptive_generation = _PreemptiveGeneration(
            speech_handle=speech_handle,
            user_message=user_message,
            info=info,
            chat_ctx=chat_ctx.copy(),
            tools=self.tools.copy(),
            tool_choice=self._tool_choice,
            created_at=time.time(),
        )

    def on_end_of_turn(self, info: _EndOfTurnInfo) -> bool:
        # IMPORTANT: This method is sync to avoid it being cancelled by the AudioRecognition
        # We explicitly create a new task here

        if self._scheduling_paused:
            self._cancel_preemptive_generation()
            logger.warning(
                "skipping user input, speech scheduling is paused",
                extra={"user_input": info.new_transcript},
            )

            if self._session._closing:
                # add user input to chat context
                user_message = llm.ChatMessage(
                    role="user",
                    content=[info.new_transcript],
                    transcript_confidence=info.transcript_confidence,
                )
                self._agent._chat_ctx.items.append(user_message)
                self._session._conversation_item_added(user_message)

            # TODO(theomonnom): should we "forward" this new turn to the next agent/activity?
            return True

        if (
            self.stt is not None
            and self._turn_detection != "manual"
            and self._current_speech is not None
            and self._current_speech.allow_interruptions
            and not self._current_speech.interrupted
            and self._session.options.min_interruption_words > 0
            and len(split_words(info.new_transcript, split_character=True))
            < self._session.options.min_interruption_words
        ):
            self._cancel_preemptive_generation()
            # avoid interruption if the new_transcript is too short
            return False

        old_task = self._user_turn_completed_atask
        self._user_turn_completed_atask = self._create_speech_task(
            self._user_turn_completed_task(old_task, info),
            name="AgentActivity._user_turn_completed_task",
        )
        return True

    @utils.log_exceptions(logger=logger)
    async def _user_turn_completed_task(
        self, old_task: asyncio.Task[None] | None, info: _EndOfTurnInfo
    ) -> None:
        if old_task is not None:
            # We never cancel user code as this is very confusing.
            # So we wait for the old execution of on_user_turn_completed to finish.
            # In practice this is OK because most speeches will be interrupted if a new turn
            # is detected. So the previous execution should complete quickly.
            await old_task

        # When the audio recognition detects the end of a user turn:
        #  - check if realtime model server-side turn detection is enabled
        #  - check if there is no current generation happening
        #  - cancel the current generation if it allows interruptions (otherwise skip this current
        #  turn)
        #  - generate a reply to the user input

        # interrupt all background speeches and wait for them to finish to update the chat context
        await asyncio.gather(*self._interrupt_background_speeches(force=False))

        if isinstance(self.llm, llm.RealtimeModel):
            if self.llm.capabilities.turn_detection:
                return

            if self._rt_session is not None:
                self._rt_session.commit_audio()

        if self._current_speech is not None:
            if not self._current_speech.allow_interruptions:
                logger.warning(
                    "skipping reply to user input, current speech generation cannot be interrupted",
                    extra={"user_input": info.new_transcript},
                )
                return
            await self._interrupt_paused_speech(self._interrupt_paused_speech_task)

            if self._current_speech:
                await self._current_speech.interrupt()

            if self._rt_session is not None:
                self._rt_session.interrupt()

        user_message = llm.ChatMessage(
            role="user",
            content=[info.new_transcript],
            transcript_confidence=info.transcript_confidence,
        )

        if self._scheduling_paused:
            logger.warning(
                "skipping on_user_turn_completed, speech scheduling is paused",
                extra={"user_input": info.new_transcript},
            )
            if self._session._closing:
                self._agent._chat_ctx.items.append(user_message)
                self._session._conversation_item_added(user_message)
            return

        # create a temporary mutable chat context to pass to on_user_turn_completed
        # the user can edit it for the current generation, but changes will not be kept inside the
        # Agent.chat_ctx
        temp_mutable_chat_ctx = self._agent.chat_ctx.copy()
        start_time = time.perf_counter()
        try:
            await self._agent.on_user_turn_completed(
                temp_mutable_chat_ctx, new_message=user_message
            )
        except StopResponse:
            return  # ignore this turn
        except Exception:
            logger.exception("error occured during on_user_turn_completed")
            return

        on_user_turn_completed_delay = time.perf_counter() - start_time

        if isinstance(self.llm, llm.RealtimeModel):
            # ignore stt transcription for realtime model
            user_message = None  # type: ignore
        elif self.llm is None:
            return  # skip response if no llm is set

        if self._scheduling_paused:
            logger.warning(
                "skipping reply to user input, speech scheduling is paused",
                extra={"user_input": info.new_transcript},
            )
            if user_message and self._session._closing:
                self._agent._chat_ctx.items.append(user_message)
                self._session._conversation_item_added(user_message)
            return

        metrics_report: llm.MetricsReport = {}
        if info.started_speaking_at is not None:
            metrics_report["started_speaking_at"] = info.started_speaking_at

        if info.stopped_speaking_at is not None:
            metrics_report["stopped_speaking_at"] = info.stopped_speaking_at

        if info.transcription_delay is not None:
            metrics_report["transcription_delay"] = info.transcription_delay

        if info.end_of_turn_delay is not None:
            metrics_report["end_of_turn_delay"] = info.end_of_turn_delay

        metrics_report["on_user_turn_completed_delay"] = on_user_turn_completed_delay

        if user_message is not None:
            user_message.metrics = metrics_report

        speech_handle: SpeechHandle | None = None
        if preemptive := self._preemptive_generation:
            # make sure the on_user_turn_completed didn't change some request parameters
            # otherwise invalidate the preemptive generation
            if (
                preemptive.info.new_transcript == user_message.text_content
                and preemptive.chat_ctx.is_equivalent(temp_mutable_chat_ctx)
                and preemptive.tools == self.tools
                and preemptive.tool_choice == self._tool_choice
            ):
                speech_handle = preemptive.speech_handle

                # preemptive generation is using another ChatMessage created outside of the on_end_of_turn callback,
                # inject the metrics here.
                preemptive.user_message.metrics = metrics_report
                self._schedule_speech(speech_handle, priority=SpeechHandle.SPEECH_PRIORITY_NORMAL)
                logger.debug(
                    "using preemptive generation",
                    extra={"preemptive_lead_time": time.time() - preemptive.created_at},
                )
            else:
                logger.warning(
                    "preemptive generation enabled but chat context or tools have changed after `on_user_turn_completed`",  # noqa: E501
                )
                preemptive.speech_handle._cancel()

            self._preemptive_generation = None

        if speech_handle is None:
            # Ensure the new message is passed to generate_reply
            # This preserves the original message_id, making it easier for users to track responses
            speech_handle = self._generate_reply(
                user_message=user_message,
                chat_ctx=temp_mutable_chat_ctx,
            )

        if self._user_turn_completed_atask != asyncio.current_task():
            # If a new user turn has already started, interrupt this one since it's now outdated
            # (We still create the SpeechHandle and the generate_reply coroutine, otherwise we may
            # lose data like the beginning of a user speech).
            # await the interrupt to make sure user message is added to the chat context before the new task starts
            await speech_handle.interrupt()

        metadata: Metadata | None = None
        if isinstance(self._turn_detection, str):
            metadata = Metadata(model_name="unknown", model_provider=self._turn_detection)
        elif self._turn_detection is not None:
            metadata = Metadata(
                model_name=self._turn_detection.model, model_provider=self._turn_detection.provider
            )

        eou_metrics = EOUMetrics(
            timestamp=time.time(),
            end_of_utterance_delay=info.end_of_turn_delay or 0.0,
            transcription_delay=info.transcription_delay or 0.0,
            on_user_turn_completed_delay=on_user_turn_completed_delay,
            speech_id=speech_handle.id,
            metadata=metadata,
        )
        self._session.emit("metrics_collected", MetricsCollectedEvent(metrics=eou_metrics))

    # AudioRecognition is calling this method to retrieve the chat context before running the TurnDetector model  # noqa: E501
    def retrieve_chat_ctx(self) -> llm.ChatContext:
        return self._agent.chat_ctx

    # endregion

    def _on_pipeline_reply_done(self, _: asyncio.Task[None]) -> None:
        if not self._speech_q and (not self._current_speech or self._current_speech.done()):
            self._session._update_agent_state("listening")

    @utils.log_exceptions(logger=logger)
    async def _tts_task(
        self,
        speech_handle: SpeechHandle,
        text: str | AsyncIterable[str],
        audio: AsyncIterable[rtc.AudioFrame] | None,
        add_to_chat_ctx: bool,
        model_settings: ModelSettings,
    ) -> None:
        with tracer.start_as_current_span(
            "agent_turn", context=self._session._root_span_context
        ) as current_span:
            current_span.set_attribute(trace_types.ATTR_AGENT_TURN_ID, speech_handle.generation_id)
            if parent_id := speech_handle.parent_generation_id:
                current_span.set_attribute(trace_types.ATTR_AGENT_PARENT_TURN_ID, parent_id)
            speech_handle._agent_turn_context = otel_context.get_current()

            await self._tts_task_impl(
                speech_handle=speech_handle,
                text=text,
                audio=audio,
                add_to_chat_ctx=add_to_chat_ctx,
                model_settings=model_settings,
            )

    async def _tts_task_impl(
        self,
        speech_handle: SpeechHandle,
        text: str | AsyncIterable[str],
        audio: AsyncIterable[rtc.AudioFrame] | None,
        add_to_chat_ctx: bool,
        model_settings: ModelSettings,
    ) -> None:
        current_span = trace.get_current_span()
        current_span.set_attribute(trace_types.ATTR_SPEECH_ID, speech_handle.id)

        tr_output = (
            self._session.output.transcription
            if self._session.output.transcription_enabled
            else None
        )
        audio_output = self._session.output.audio if self._session.output.audio_enabled else None

        wait_for_authorization = asyncio.ensure_future(speech_handle._wait_for_authorization())
        await speech_handle.wait_if_not_interrupted([wait_for_authorization])
        speech_handle._clear_authorization()

        if speech_handle.interrupted:
            current_span.set_attribute(trace_types.ATTR_SPEECH_INTERRUPTED, True)
            await utils.aio.cancel_and_wait(wait_for_authorization)
            return

        text_source: AsyncIterable[str] | None = None
        audio_source: AsyncIterable[str] | None = None

        tee: utils.aio.itertools.Tee[str] | None = None
        if isinstance(text, AsyncIterable):
            tee = utils.aio.itertools.tee(text, 2)
            text_source, audio_source = tee
        elif isinstance(text, str):

            async def _read_text() -> AsyncIterable[str]:
                yield text

            text_source = _read_text()
            audio_source = _read_text()

        tasks: list[asyncio.Task[Any]] = []
        started_speaking_at: float | None = None
        stopped_speaking_at: float | None = None

        def _on_first_frame(_: asyncio.Future[None]) -> None:
            nonlocal started_speaking_at
            started_speaking_at = time.time()
            self._session._update_agent_state("speaking")

        audio_out: _AudioOutput | None = None
        tts_gen_data: _TTSGenerationData | None = None
        if audio_output is not None:
            if audio is None:
                # generate audio using TTS
                tts_task, tts_gen_data = perform_tts_inference(
                    node=self._agent.tts_node,
                    input=audio_source,
                    model_settings=model_settings,
                    text_transforms=self._session.options.tts_text_transforms,
                )
                tasks.append(tts_task)
                if (
                    self.use_tts_aligned_transcript
                    and (tts := self.tts)
                    and (tts.capabilities.aligned_transcript or not tts.capabilities.streaming)
                    and (timed_texts := await tts_gen_data.timed_texts_fut)
                ):
                    text_source = timed_texts

                forward_task, audio_out = perform_audio_forwarding(
                    audio_output=audio_output, tts_output=tts_gen_data.audio_ch
                )
                tasks.append(forward_task)
            else:
                # use the provided audio
                forward_task, audio_out = perform_audio_forwarding(
                    audio_output=audio_output, tts_output=audio
                )
                tasks.append(forward_task)

            audio_out.first_frame_fut.add_done_callback(_on_first_frame)

        # text output
        tr_node = self._agent.transcription_node(text_source, model_settings)
        tr_node_result = await tr_node if asyncio.iscoroutine(tr_node) else tr_node
        text_out: _TextOutput | None = None
        if tr_node_result is not None:
            forward_text, text_out = perform_text_forwarding(
                text_output=tr_output,
                source=tr_node_result,
            )
            tasks.append(forward_text)
            if audio_output is None:
                # update the agent state based on text if no audio output
                text_out.first_text_fut.add_done_callback(_on_first_frame)

        await speech_handle.wait_if_not_interrupted([*tasks])

        if audio_output is not None:
            await speech_handle.wait_if_not_interrupted(
                [asyncio.ensure_future(audio_output.wait_for_playout())]
            )

        stopped_speaking_at = time.time()
        current_span.set_attribute(trace_types.ATTR_SPEECH_INTERRUPTED, speech_handle.interrupted)
        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*tasks)

            if audio_output is not None:
                audio_output.clear_buffer()
                await audio_output.wait_for_playout()

        if tee is not None:
            await tee.aclose()

        # use synchronized transcript when available after interruption
        forwarded_text = text_out.text if text_out else ""
        if speech_handle.interrupted and audio_output is not None:
            playback_ev = await audio_output.wait_for_playout()

            if audio_out is not None and audio_out.first_frame_fut.done():
                if playback_ev.synchronized_transcript is not None:
                    forwarded_text = playback_ev.synchronized_transcript
            else:
                forwarded_text = ""
        current_span.set_attribute(trace_types.ATTR_RESPONSE_TEXT, forwarded_text)

        if forwarded_text and add_to_chat_ctx:
            assistant_metrics: llm.MetricsReport = {}

            if tts_gen_data and tts_gen_data.ttfb is not None:
                assistant_metrics["tts_node_ttfb"] = tts_gen_data.ttfb

            if stopped_speaking_at and started_speaking_at:
                assistant_metrics["started_speaking_at"] = started_speaking_at
                assistant_metrics["stopped_speaking_at"] = stopped_speaking_at

            msg = self._agent._chat_ctx.add_message(
                role="assistant",
                content=forwarded_text,
                interrupted=speech_handle.interrupted,
                metrics=assistant_metrics,
            )
            speech_handle._item_added([msg])
            self._session._conversation_item_added(msg)

        if self._session.agent_state == "speaking":
            self._session._update_agent_state("listening")

    @utils.log_exceptions(logger=logger)
    async def _pipeline_reply_task(
        self,
        *,
        speech_handle: SpeechHandle,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool | llm.RawFunctionTool],
        model_settings: ModelSettings,
        new_message: llm.ChatMessage | None = None,
        instructions: str | None = None,
        _previous_user_metrics: llm.MetricsReport | None = None,
        _previous_tools_messages: Sequence[llm.FunctionCall | llm.FunctionCallOutput] | None = None,
    ) -> None:
        with tracer.start_as_current_span(
            "agent_turn", context=self._session._root_span_context
        ) as current_span:
            current_span.set_attribute(trace_types.ATTR_AGENT_TURN_ID, speech_handle.generation_id)
            if parent_id := speech_handle.parent_generation_id:
                current_span.set_attribute(trace_types.ATTR_AGENT_PARENT_TURN_ID, parent_id)
            speech_handle._agent_turn_context = otel_context.get_current()

            await self._pipeline_reply_task_impl(
                speech_handle=speech_handle,
                chat_ctx=chat_ctx,
                tools=tools,
                model_settings=model_settings,
                new_message=new_message,
                instructions=instructions,
                _previous_user_metrics=_previous_user_metrics,
                _previous_tools_messages=_previous_tools_messages,
            )

    async def _pipeline_reply_task_impl(
        self,
        *,
        speech_handle: SpeechHandle,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool | llm.RawFunctionTool],
        model_settings: ModelSettings,
        new_message: llm.ChatMessage | None = None,
        instructions: str | None = None,
        _previous_user_metrics: llm.MetricsReport | None = None,
        _previous_tools_messages: Sequence[llm.FunctionCall | llm.FunctionCallOutput] | None = None,
    ) -> None:
        from .agent import ModelSettings

        current_span = trace.get_current_span()
        current_span.set_attribute(trace_types.ATTR_SPEECH_ID, speech_handle.id)
        if instructions is not None:
            current_span.set_attribute(trace_types.ATTR_INSTRUCTIONS, instructions)
        if new_message:
            current_span.set_attribute(trace_types.ATTR_USER_INPUT, new_message.text_content or "")

        if (room_io := self._session._room_io) and room_io.room.isconnected():
            _set_participant_attributes(current_span, room_io.room.local_participant)

        audio_output = self._session.output.audio if self._session.output.audio_enabled else None
        text_output = (
            self._session.output.transcription
            if self._session.output.transcription_enabled
            else None
        )
        chat_ctx = chat_ctx.copy()
        tool_ctx = llm.ToolContext(tools)

        if new_message is not None:
            chat_ctx.insert(new_message)

        if instructions is not None:
            try:
                update_instructions(chat_ctx, instructions=instructions, add_if_missing=True)
            except ValueError:
                logger.exception("failed to update the instructions")

        # TODO(theomonnom): since pause is closing STT/LLM/TTS, we have issues for SpeechHandle still in queue  # noqa: E501
        # I should implement a retry mechanism?

        tasks: list[asyncio.Task[Any]] = []
        llm_task, llm_gen_data = perform_llm_inference(
            node=self._agent.llm_node,
            chat_ctx=chat_ctx,
            tool_ctx=tool_ctx,
            model_settings=model_settings,
        )
        tasks.append(llm_task)

        text_tee = utils.aio.itertools.tee(llm_gen_data.text_ch, 2)
        tts_text_input, tr_input = text_tee

        tts_task: asyncio.Task[bool] | None = None
        tts_gen_data: _TTSGenerationData | None = None
        read_transcript_from_tts = False
        if audio_output is not None:
            await llm_gen_data.started_fut  # make sure tts span starts after llm span
            tts_task, tts_gen_data = perform_tts_inference(
                node=self._agent.tts_node,
                input=tts_text_input,
                model_settings=model_settings,
                text_transforms=self._session.options.tts_text_transforms,
            )
            tasks.append(tts_task)
            if (
                self.use_tts_aligned_transcript
                and (tts := self.tts)
                and (tts.capabilities.aligned_transcript or not tts.capabilities.streaming)
                and (timed_texts := await tts_gen_data.timed_texts_fut)
            ):
                tr_input = timed_texts
                read_transcript_from_tts = True

        wait_for_scheduled = asyncio.ensure_future(speech_handle._wait_for_scheduled())
        await speech_handle.wait_if_not_interrupted([wait_for_scheduled])

        # add new message to chat context if the speech is scheduled

        user_metrics: llm.MetricsReport | None = _previous_user_metrics
        if new_message is not None and speech_handle.scheduled:
            self._agent._chat_ctx.insert(new_message)
            self._session._conversation_item_added(new_message)
            user_metrics = new_message.metrics

        if speech_handle.interrupted:
            current_span.set_attribute(trace_types.ATTR_SPEECH_INTERRUPTED, True)
            await utils.aio.cancel_and_wait(*tasks, wait_for_scheduled)
            await text_tee.aclose()
            return

        self._session._update_agent_state("thinking")

        wait_for_authorization = asyncio.ensure_future(speech_handle._wait_for_authorization())
        await speech_handle.wait_if_not_interrupted([wait_for_authorization])
        speech_handle._clear_authorization()

        if speech_handle.interrupted:
            current_span.set_attribute(trace_types.ATTR_SPEECH_INTERRUPTED, True)
            await utils.aio.cancel_and_wait(*tasks, wait_for_authorization)
            await text_tee.aclose()
            return

        reply_started_at = time.time()

        async def _read_text(
            llm_output: AsyncIterable[str | FlushSentinel],
        ) -> AsyncIterable[str]:
            async for chunk in llm_output:
                if isinstance(chunk, FlushSentinel):
                    continue
                yield chunk

        tr_node = self._agent.transcription_node(_read_text(tr_input), model_settings)
        tr_node_result = await tr_node if asyncio.iscoroutine(tr_node) else tr_node
        text_out: _TextOutput | None = None
        text_forward_task: asyncio.Task | None = None
        if tr_node_result is not None:
            text_forward_task, text_out = perform_text_forwarding(
                text_output=text_output, source=tr_node_result
            )
            tasks.append(text_forward_task)

        started_speaking_at: float | None = None
        stopped_speaking_at: float | None = None

        def _on_first_frame(_: asyncio.Future[None]) -> None:
            nonlocal started_speaking_at
            started_speaking_at = time.time()
            self._session._update_agent_state("speaking")

        audio_out: _AudioOutput | None = None
        if audio_output is not None:
            assert tts_gen_data is not None
            # TODO(theomonnom): should the audio be added to the chat_context too?
            forward_task, audio_out = perform_audio_forwarding(
                audio_output=audio_output, tts_output=tts_gen_data.audio_ch
            )
            tasks.append(forward_task)

            audio_out.first_frame_fut.add_done_callback(_on_first_frame)
        elif text_out is not None:
            text_out.first_text_fut.add_done_callback(_on_first_frame)

        # messages in RunResult are ordered by the `created_at` field
        def _tool_execution_started_cb(fnc_call: llm.FunctionCall) -> None:
            speech_handle._item_added([fnc_call])

        def _tool_execution_completed_cb(out: ToolExecutionOutput) -> None:
            if out.fnc_call_out:
                speech_handle._item_added([out.fnc_call_out])

        # start to execute tools (only after play())
        exe_task, tool_output = perform_tool_executions(
            session=self._session,
            speech_handle=speech_handle,
            tool_ctx=tool_ctx,
            tool_choice=model_settings.tool_choice,
            function_stream=llm_gen_data.function_ch,
            tool_execution_started_cb=_tool_execution_started_cb,
            tool_execution_completed_cb=_tool_execution_completed_cb,
        )

        await speech_handle.wait_if_not_interrupted([*tasks])

        # wait for the end of the playout if the audio is enabled
        if audio_output is not None:
            await speech_handle.wait_if_not_interrupted(
                [asyncio.ensure_future(audio_output.wait_for_playout())]
            )

        stopped_speaking_at = time.time()
        assistant_metrics: llm.MetricsReport = {}

        if llm_gen_data.ttft is not None:
            assistant_metrics["llm_node_ttft"] = llm_gen_data.ttft

        if tts_gen_data and tts_gen_data.ttfb is not None:
            assistant_metrics["tts_node_ttfb"] = tts_gen_data.ttfb

        if stopped_speaking_at and started_speaking_at:
            assistant_metrics["started_speaking_at"] = started_speaking_at
            assistant_metrics["stopped_speaking_at"] = stopped_speaking_at

            if user_metrics and "stopped_speaking_at" in user_metrics:
                assistant_metrics["e2e_latency"] = (
                    started_speaking_at - user_metrics["stopped_speaking_at"]
                )

        current_span.set_attribute(trace_types.ATTR_SPEECH_INTERRUPTED, speech_handle.interrupted)
        has_speech_message = False

        # add the tools messages that triggers this reply to the chat context
        if _previous_tools_messages:
            for tool_msg in _previous_tools_messages:
                # reset the created_at to the reply start time
                tool_msg.created_at = reply_started_at
            self._agent._chat_ctx.insert(_previous_tools_messages)
            self._session._tool_items_added(_previous_tools_messages)

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*tasks)
            await text_tee.aclose()

            forwarded_text = text_out.text if text_out else ""
            if forwarded_text:
                has_speech_message = True
            # if the audio playout was enabled, clear the buffer
            if audio_output is not None:
                audio_output.clear_buffer()

                playback_ev = await audio_output.wait_for_playout()
                if audio_out is not None and audio_out.first_frame_fut.done():
                    # playback_ev is valid only if the first frame was already played
                    if playback_ev.synchronized_transcript is not None:
                        forwarded_text = playback_ev.synchronized_transcript
                else:
                    forwarded_text = ""

            if forwarded_text:
                msg = chat_ctx.add_message(
                    role="assistant",
                    content=forwarded_text,
                    id=llm_gen_data.id,
                    interrupted=True,
                    created_at=reply_started_at,
                    metrics=assistant_metrics,
                )
                self._agent._chat_ctx.insert(msg)
                self._session._conversation_item_added(msg)
                speech_handle._item_added([msg])
                current_span.set_attribute(trace_types.ATTR_RESPONSE_TEXT, forwarded_text)

            if self._session.agent_state == "speaking":
                self._session._update_agent_state("listening")

            speech_handle._mark_generation_done()
            await utils.aio.cancel_and_wait(exe_task)
            return

        if read_transcript_from_tts and text_out and not text_out.text:
            logger.warning(
                "`use_tts_aligned_transcript` is enabled but no agent transcript was returned from tts"
            )

        if text_out and text_out.text:
            has_speech_message = True
            msg = chat_ctx.add_message(
                role="assistant",
                content=text_out.text,
                id=llm_gen_data.id,
                interrupted=False,
                created_at=reply_started_at,
                metrics=assistant_metrics,
            )
            self._agent._chat_ctx.insert(msg)
            self._session._conversation_item_added(msg)
            speech_handle._item_added([msg])
            current_span.set_attribute(trace_types.ATTR_RESPONSE_TEXT, text_out.text)

        if len(tool_output.output) > 0:
            self._session._update_agent_state("thinking")
        elif self._session.agent_state == "speaking":
            self._session._update_agent_state("listening")

        await text_tee.aclose()

        speech_handle._mark_generation_done()  # mark the playout done before waiting for the tool execution  # noqa: E501
        self._background_speeches.add(speech_handle)
        try:
            await exe_task
        finally:
            self._background_speeches.discard(speech_handle)

        # important: no agent output should be used after this point

        if len(tool_output.output) > 0:
            if speech_handle.num_steps >= self._session.options.max_tool_steps + 1:
                logger.warning(
                    "maximum number of function calls steps reached",
                    extra={"speech_id": speech_handle.id},
                )
                return

            speech_handle._num_steps += 1

            new_calls: list[llm.FunctionCall] = []
            new_fnc_outputs: list[llm.FunctionCallOutput] = []
            new_agent_task: Agent | None = None
            ignore_task_switch = False
            fnc_executed_ev = FunctionToolsExecutedEvent(
                function_calls=[], function_call_outputs=[]
            )
            for sanitized_out in tool_output.output:
                if sanitized_out.fnc_call_out is not None:
                    new_calls.append(sanitized_out.fnc_call)
                    new_fnc_outputs.append(sanitized_out.fnc_call_out)
                    if sanitized_out.reply_required:
                        fnc_executed_ev._reply_required = True

                # add the function call and output to the event, including the None outputs
                fnc_executed_ev.function_calls.append(sanitized_out.fnc_call)
                fnc_executed_ev.function_call_outputs.append(sanitized_out.fnc_call_out)

                if new_agent_task is not None and sanitized_out.agent_task is not None:
                    logger.error("expected to receive only one AgentTask from the tool executions")
                    ignore_task_switch = True
                    # TODO(long): should we mark the function call as failed to notify the LLM?

                new_agent_task = sanitized_out.agent_task

            if new_agent_task and not ignore_task_switch:
                fnc_executed_ev._handoff_required = True

            self._session.emit("function_tools_executed", fnc_executed_ev)

            draining = self.scheduling_paused
            if fnc_executed_ev._handoff_required and new_agent_task and not ignore_task_switch:
                self._session.update_agent(new_agent_task)
                draining = True

            tool_messages = new_calls + new_fnc_outputs
            if fnc_executed_ev._reply_required:
                chat_ctx.items.extend(tool_messages)

                tool_response_task = self._create_speech_task(
                    self._pipeline_reply_task(
                        speech_handle=speech_handle,
                        chat_ctx=chat_ctx,
                        tools=tools,
                        model_settings=ModelSettings(
                            # Avoid setting tool_choice to "required" or a specific function when
                            # passing tool response back to the LLM
                            tool_choice="none"
                            if draining or model_settings.tool_choice == "none"
                            else "auto",
                        ),
                        # in case the current reply only generated tools (no speech), re-use the current user_metrics for the next
                        # tool response generation
                        _previous_user_metrics=user_metrics if not has_speech_message else None,
                        _previous_tools_messages=tool_messages,
                    ),
                    speech_handle=speech_handle,
                    name="AgentActivity.pipeline_reply",
                )
                tool_response_task.add_done_callback(self._on_pipeline_reply_done)
                self._schedule_speech(
                    speech_handle, SpeechHandle.SPEECH_PRIORITY_NORMAL, force=True
                )
            elif len(new_fnc_outputs) > 0:
                # add the tool calls and outputs to the chat context even no reply is generated
                for tool_msg in tool_messages:
                    tool_msg.created_at = reply_started_at
                self._agent._chat_ctx.insert(tool_messages)
                self._session._tool_items_added(tool_messages)

    @utils.log_exceptions(logger=logger)
    async def _realtime_reply_task(
        self,
        *,
        speech_handle: SpeechHandle,
        model_settings: ModelSettings,
        user_input: str | None = None,
        instructions: str | None = None,
    ) -> None:
        assert self._rt_session is not None, "rt_session is not available"

        # realtime_reply_task is called only when there's text input, native audio input is handled by _realtime_generation_task
        wait_for_authorization = asyncio.ensure_future(speech_handle._wait_for_authorization())
        await speech_handle.wait_if_not_interrupted([wait_for_authorization])
        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(wait_for_authorization)

        if user_input is not None:
            chat_ctx = self._rt_session.chat_ctx.copy()
            msg = chat_ctx.add_message(role="user", content=user_input)
            await self._rt_session.update_chat_ctx(chat_ctx)
            self._agent._chat_ctx.items.append(msg)
            self._session._conversation_item_added(msg)

        ori_tool_choice = self._tool_choice
        if utils.is_given(model_settings.tool_choice):
            self._rt_session.update_options(
                tool_choice=cast(llm.ToolChoice, model_settings.tool_choice)
            )

        try:
            generation_ev = await self._rt_session.generate_reply(
                instructions=instructions or NOT_GIVEN
            )

            # _realtime_generation_task will clear the authorization
            await self._realtime_generation_task(
                speech_handle=speech_handle,
                generation_ev=generation_ev,
                model_settings=model_settings,
                instructions=instructions,
            )
        finally:
            # reset tool_choice value
            if (
                utils.is_given(model_settings.tool_choice)
                and model_settings.tool_choice != ori_tool_choice
            ):
                self._rt_session.update_options(tool_choice=ori_tool_choice)

    @utils.log_exceptions(logger=logger)
    async def _realtime_generation_task(
        self,
        *,
        speech_handle: SpeechHandle,
        generation_ev: llm.GenerationCreatedEvent,
        model_settings: ModelSettings,
        instructions: str | None = None,
    ) -> None:
        with tracer.start_as_current_span(
            "agent_turn", context=self._session._root_span_context
        ) as current_span:
            current_span.set_attribute(trace_types.ATTR_AGENT_TURN_ID, speech_handle.generation_id)
            if parent_id := speech_handle.parent_generation_id:
                current_span.set_attribute(trace_types.ATTR_AGENT_PARENT_TURN_ID, parent_id)
            speech_handle._agent_turn_context = otel_context.get_current()

            await self._realtime_generation_task_impl(
                speech_handle=speech_handle,
                generation_ev=generation_ev,
                model_settings=model_settings,
                instructions=instructions,
            )

    async def _realtime_generation_task_impl(
        self,
        *,
        speech_handle: SpeechHandle,
        generation_ev: llm.GenerationCreatedEvent,
        model_settings: ModelSettings,
        instructions: str | None = None,
    ) -> None:
        current_span = trace.get_current_span()
        current_span.set_attribute(trace_types.ATTR_SPEECH_ID, speech_handle.id)

        room_io = self._session._room_io
        if room_io and room_io.room.isconnected():
            _set_participant_attributes(current_span, room_io.room.local_participant)

        assert self._rt_session is not None, "rt_session is not available"
        assert isinstance(self.llm, llm.RealtimeModel), "llm is not a realtime model"

        current_span.set_attribute(trace_types.ATTR_GEN_AI_REQUEST_MODEL, self.llm.model)
        if self._realtime_spans is not None and generation_ev.response_id:
            self._realtime_spans[generation_ev.response_id] = current_span

        audio_output = self._session.output.audio if self._session.output.audio_enabled else None
        text_output = (
            self._session.output.transcription
            if self._session.output.transcription_enabled
            else None
        )
        tool_ctx = llm.ToolContext(self.tools)

        wait_for_authorization = asyncio.ensure_future(speech_handle._wait_for_authorization())
        await speech_handle.wait_if_not_interrupted([wait_for_authorization])
        speech_handle._clear_authorization()

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(wait_for_authorization)
            current_span.set_attribute(trace_types.ATTR_SPEECH_INTERRUPTED, True)
            return  # TODO(theomonnom): remove the message from the serverside history

        started_speaking_at: float | None = None
        stopped_speaking_at: float | None = None

        def _on_first_frame(_: asyncio.Future[None]) -> None:
            nonlocal started_speaking_at
            started_speaking_at = time.time()
            self._session._update_agent_state("speaking")

        tasks: list[asyncio.Task[Any]] = []
        tees: list[utils.aio.itertools.Tee[Any]] = []

        read_transcript_from_tts = False

        # read text and audio outputs
        @utils.log_exceptions(logger=logger)
        async def _read_messages(
            outputs: list[tuple[MessageGeneration, _TextOutput | None, _AudioOutput | None]],
        ) -> None:
            nonlocal read_transcript_from_tts
            assert isinstance(self.llm, llm.RealtimeModel)

            forward_tasks: list[asyncio.Task[Any]] = []
            try:
                async for msg in generation_ev.message_stream:
                    if len(forward_tasks) > 0:
                        logger.warning(
                            "expected to receive only one message generation from the realtime API"
                        )
                        break

                    msg_modalities = await msg.modalities
                    tts_text_input: AsyncIterable[str] | None = None
                    if "audio" not in msg_modalities and self.tts:
                        if self.llm.capabilities.audio_output:
                            logger.warning(
                                "text response received from realtime API, falling back to use a TTS model."
                            )
                        tee = utils.aio.itertools.tee(msg.text_stream, 2)
                        tts_text_input, tr_text_input = tee
                        tees.append(tee)
                    else:
                        tr_text_input = msg.text_stream.__aiter__()

                    # audio output
                    audio_out = None
                    if audio_output is not None:
                        realtime_audio_result: AsyncIterable[rtc.AudioFrame] | None = None
                        if tts_text_input is not None:
                            tts_task, tts_gen_data = perform_tts_inference(
                                node=self._agent.tts_node,
                                input=tts_text_input,
                                model_settings=model_settings,
                                text_transforms=self._session.options.tts_text_transforms,
                            )

                            if (
                                self.use_tts_aligned_transcript
                                and (tts := self.tts)
                                and (
                                    tts.capabilities.aligned_transcript
                                    or not tts.capabilities.streaming
                                )
                                and (timed_texts := await tts_gen_data.timed_texts_fut)
                            ):
                                tr_text_input = timed_texts
                                read_transcript_from_tts = True

                            tasks.append(tts_task)
                            realtime_audio_result = tts_gen_data.audio_ch
                        elif "audio" in msg_modalities:
                            realtime_audio = self._agent.realtime_audio_output_node(
                                msg.audio_stream, model_settings
                            )
                            realtime_audio_result = (
                                await realtime_audio
                                if asyncio.iscoroutine(realtime_audio)
                                else realtime_audio
                            )
                        elif self.llm.capabilities.audio_output:
                            logger.error(
                                "Text message received from Realtime API with audio modality. "
                                "This usually happens when text chat context is synced to the API. "
                                "Try to add a TTS model as fallback or use text modality with TTS instead."
                            )
                        else:
                            logger.warning(
                                "audio output is enabled but neither tts nor realtime audio is available",  # noqa: E501
                            )

                        if realtime_audio_result is not None:
                            forward_task, audio_out = perform_audio_forwarding(
                                audio_output=audio_output,
                                tts_output=realtime_audio_result,
                            )
                            forward_tasks.append(forward_task)
                            audio_out.first_frame_fut.add_done_callback(_on_first_frame)

                    # text output
                    tr_node = self._agent.transcription_node(tr_text_input, model_settings)
                    tr_node_result = await tr_node if asyncio.iscoroutine(tr_node) else tr_node
                    text_out: _TextOutput | None = None
                    if tr_node_result is not None:
                        forward_task, text_out = perform_text_forwarding(
                            text_output=text_output,
                            source=tr_node_result,
                        )
                        forward_tasks.append(forward_task)

                    if not audio_out and text_out:
                        text_out.first_text_fut.add_done_callback(_on_first_frame)

                    outputs.append((msg, text_out, audio_out))

                await asyncio.gather(*forward_tasks)
            finally:
                await utils.aio.cancel_and_wait(*forward_tasks)

        message_outputs: list[
            tuple[MessageGeneration, _TextOutput | None, _AudioOutput | None]
        ] = []
        tasks.append(
            asyncio.create_task(
                _read_messages(message_outputs),
                name="AgentActivity.realtime_generation.read_messages",
            )
        )

        # read function calls
        fnc_tee = utils.aio.itertools.tee(generation_ev.function_stream, 2)
        fnc_stream, fnc_stream_for_tracing = fnc_tee
        tees.append(fnc_tee)
        function_calls: list[llm.FunctionCall] = []

        async def _read_fnc_stream() -> None:
            async for fnc in fnc_stream_for_tracing:
                function_calls.append(fnc)

        tasks.append(
            asyncio.create_task(
                _read_fnc_stream(),
                name="AgentActivity.realtime_generation.read_fnc_stream",
            )
        )

        def _tool_execution_started_cb(fnc_call: llm.FunctionCall) -> None:
            speech_handle._item_added([fnc_call])
            self._agent._chat_ctx.items.append(fnc_call)
            self._session._tool_items_added([fnc_call])

        def _tool_execution_completed_cb(out: ToolExecutionOutput) -> None:
            if out.fnc_call_out:
                speech_handle._item_added([out.fnc_call_out])

        exe_task, tool_output = perform_tool_executions(
            session=self._session,
            speech_handle=speech_handle,
            tool_ctx=tool_ctx,
            tool_choice=model_settings.tool_choice,
            function_stream=fnc_stream,
            tool_execution_started_cb=_tool_execution_started_cb,
            tool_execution_completed_cb=_tool_execution_completed_cb,
        )

        await speech_handle.wait_if_not_interrupted([*tasks])

        current_span.set_attribute(trace_types.ATTR_SPEECH_INTERRUPTED, speech_handle.interrupted)
        current_span.set_attribute(
            trace_types.ATTR_RESPONSE_FUNCTION_CALLS,
            json.dumps([fnc.model_dump(exclude={"type", "created_at"}) for fnc in function_calls]),
        )

        if audio_output is not None:
            await speech_handle.wait_if_not_interrupted(
                [asyncio.ensure_future(audio_output.wait_for_playout())]
            )
            self._session._update_agent_state("listening")
            current_span.set_attribute(
                trace_types.ATTR_SPEECH_INTERRUPTED, speech_handle.interrupted
            )

        stopped_speaking_at = time.time()

        def _create_assistant_message(
            message_id: str, forwarded_text: str, interrupted: bool
        ) -> llm.ChatMessage:
            assistant_metrics: llm.MetricsReport = {}

            if stopped_speaking_at and started_speaking_at:
                assistant_metrics["started_speaking_at"] = started_speaking_at
                assistant_metrics["stopped_speaking_at"] = stopped_speaking_at

            msg = llm.ChatMessage(
                role="assistant",
                content=[forwarded_text],
                id=message_id,
                interrupted=interrupted,
            )
            if started_speaking_at is not None:
                msg.created_at = started_speaking_at
            msg.metrics = assistant_metrics
            return msg

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*tasks)

            if len(message_outputs) > 0:
                # there should be only one message
                msg_gen, text_out, audio_out = message_outputs[0]
                forwarded_text = text_out.text if text_out else ""
                if audio_output is not None:
                    audio_output.clear_buffer()

                    playback_ev = await audio_output.wait_for_playout()
                    playback_position = playback_ev.playback_position
                    if audio_out is not None and audio_out.first_frame_fut.done():
                        # playback_ev is valid only if the first frame was already played
                        if playback_ev.synchronized_transcript is not None:
                            forwarded_text = playback_ev.synchronized_transcript
                    else:
                        forwarded_text = ""
                        playback_position = 0

                    # truncate server-side message (if supported)
                    if self.llm.capabilities.message_truncation:
                        msg_modalities = await msg_gen.modalities
                        self._rt_session.truncate(
                            message_id=msg_gen.message_id,
                            modalities=msg_modalities,
                            audio_end_ms=int(playback_position * 1000),
                            audio_transcript=forwarded_text,
                        )

                msg: llm.ChatMessage | None = None
                if forwarded_text:
                    msg = _create_assistant_message(
                        message_id=msg_gen.message_id,
                        forwarded_text=forwarded_text,
                        interrupted=True,
                    )
                    self._agent._chat_ctx.items.append(msg)
                    speech_handle._item_added([msg])
                    self._session._conversation_item_added(msg)
                    current_span.set_attribute(trace_types.ATTR_RESPONSE_TEXT, forwarded_text)

            speech_handle._mark_generation_done()
            await utils.aio.cancel_and_wait(exe_task)

            for tee in tees:
                await tee.aclose()
            return

        if len(message_outputs) > 0:
            # there should be only one message
            msg_gen, text_out, _ = message_outputs[0]
            forwarded_text = text_out.text if text_out else ""
            if forwarded_text:
                msg = _create_assistant_message(
                    message_id=msg_gen.message_id,
                    forwarded_text=forwarded_text,
                    interrupted=False,
                )
                self._agent._chat_ctx.items.append(msg)
                speech_handle._item_added([msg])
                self._session._conversation_item_added(msg)
                current_span.set_attribute(trace_types.ATTR_RESPONSE_TEXT, forwarded_text)

            elif read_transcript_from_tts and text_out is not None:
                logger.warning(
                    "`use_tts_aligned_transcript` is enabled but no agent transcript was returned from tts"
                )

        for tee in tees:
            await tee.aclose()

        speech_handle._mark_generation_done()  # mark the playout done before waiting for the tool execution  # noqa: E501
        tool_output.first_tool_started_fut.add_done_callback(
            lambda _: self._session._update_agent_state("thinking")
        )

        self._background_speeches.add(speech_handle)
        try:
            await exe_task
        finally:
            self._background_speeches.discard(speech_handle)

        # important: no agent output should be used after this point

        if len(tool_output.output) > 0:
            speech_handle._num_steps += 1

            new_fnc_outputs: list[llm.FunctionCallOutput] = []
            generate_tool_reply: bool = False
            fnc_executed_ev = FunctionToolsExecutedEvent(
                function_calls=[], function_call_outputs=[]
            )
            new_agent_task: Agent | None = None
            ignore_task_switch = False

            for sanitized_out in tool_output.output:
                # add the function call and output to the event, including the None outputs
                fnc_executed_ev.function_calls.append(sanitized_out.fnc_call)
                fnc_executed_ev.function_call_outputs.append(sanitized_out.fnc_call_out)

                if sanitized_out.fnc_call_out is not None:
                    new_fnc_outputs.append(sanitized_out.fnc_call_out)
                    if sanitized_out.reply_required:
                        generate_tool_reply = True
                        fnc_executed_ev._reply_required = True

                    # add tool output to the chat context
                    self._agent._chat_ctx.items.append(sanitized_out.fnc_call_out)
                    self._session._tool_items_added([sanitized_out.fnc_call_out])

                if new_agent_task is not None and sanitized_out.agent_task is not None:
                    logger.error(
                        "expected to receive only one AgentTask from the tool executions",
                    )
                    ignore_task_switch = True

                new_agent_task = sanitized_out.agent_task

            if new_agent_task and not ignore_task_switch:
                fnc_executed_ev._handoff_required = True

            self._session.emit("function_tools_executed", fnc_executed_ev)

            draining = self.scheduling_paused
            if fnc_executed_ev._handoff_required and new_agent_task and not ignore_task_switch:
                self._session.update_agent(new_agent_task)
                draining = True

            if len(new_fnc_outputs) > 0:
                # wait all speeches played before updating the tool output and generating the response
                # most realtime models dont't support generating multiple responses at the same time
                while self._current_speech or self._speech_q:
                    if (
                        self._current_speech
                        and not self._current_speech.done()
                        and self._current_speech is not speech_handle
                    ):
                        await self._current_speech
                    else:
                        await asyncio.sleep(0)

                chat_ctx = self._rt_session.chat_ctx.copy()
                chat_ctx.items.extend(new_fnc_outputs)
                try:
                    await self._rt_session.update_chat_ctx(chat_ctx)
                except llm.RealtimeError as e:
                    logger.warning(
                        "failed to update chat context before generating the function calls results",  # noqa: E501
                        extra={"error": str(e)},
                    )

            if (
                fnc_executed_ev._reply_required
                and not self.llm.capabilities.auto_tool_reply_generation
            ):
                self._rt_session.interrupt()

                self._create_speech_task(
                    self._realtime_reply_task(
                        speech_handle=speech_handle,
                        model_settings=ModelSettings(
                            # Avoid setting tool_choice to "required" or a specific function when
                            # passing tool response back to the LLM
                            tool_choice="none"
                            if draining or model_settings.tool_choice == "none"
                            else "auto",
                        ),
                    ),
                    speech_handle=speech_handle,
                    name="AgentActivity.realtime_reply",
                )
                self._schedule_speech(
                    speech_handle, SpeechHandle.SPEECH_PRIORITY_NORMAL, force=True
                )
            elif (
                self.llm.capabilities.auto_tool_reply_generation
                and not fnc_executed_ev._reply_required
                and generate_tool_reply
            ):
                logger.warning(
                    f"Tool reply cannot be prevented when using {self.llm._label}, it generates reply automatically."
                )

    def _start_false_interruption_timer(self, timeout: float) -> None:
        if self._false_interruption_timer is not None:
            self._false_interruption_timer.cancel()

        def _on_false_interruption() -> None:
            if self._paused_speech is None or (
                self._current_speech and self._current_speech is not self._paused_speech
            ):
                # already new speech is scheduled, do nothing
                self._paused_speech = None
                return

            resumed = False
            if (
                self._session.options.resume_false_interruption
                and (audio_output := self._session.output.audio)
                and audio_output.can_pause
                and not self._paused_speech.done()
            ):
                self._session._update_agent_state(
                    "speaking", otel_context=self._paused_speech._agent_turn_context
                )
                audio_output.resume()
                resumed = True
                logger.debug("resumed false interrupted speech", extra={"timeout": timeout})

            self._session.emit(
                "agent_false_interruption", AgentFalseInterruptionEvent(resumed=resumed)
            )

            self._paused_speech = None
            self._false_interruption_timer = None

        self._false_interruption_timer = self._session._loop.call_later(
            timeout, _on_false_interruption
        )

    async def _interrupt_paused_speech(self, old_task: asyncio.Task[None] | None = None) -> None:
        if old_task is not None:
            await old_task

        if self._false_interruption_timer is not None:
            self._false_interruption_timer.cancel()
            self._false_interruption_timer = None

        if not self._paused_speech:
            return

        if not self._paused_speech.interrupted and self._paused_speech.allow_interruptions:
            await self._paused_speech.interrupt()  # ensure the speech is done
        self._paused_speech = None

        if self._session.options.resume_false_interruption and self._session.output.audio:
            self._session.output.audio.resume()

    # move them to the end to avoid shadowing the same named modules for mypy
    @property
    def vad(self) -> vad.VAD | None:
        return self._agent.vad if is_given(self._agent.vad) else self._session.vad

    @property
    def stt(self) -> stt.STT | None:
        return self._agent.stt if is_given(self._agent.stt) else self._session.stt

    @property
    def llm(self) -> llm.LLM | llm.RealtimeModel | None:
        return cast(
            Optional[Union[llm.LLM, llm.RealtimeModel]],
            self._agent.llm if is_given(self._agent.llm) else self._session.llm,
        )

    @property
    def tts(self) -> tts.TTS | None:
        return self._agent.tts if is_given(self._agent.tts) else self._session.tts
