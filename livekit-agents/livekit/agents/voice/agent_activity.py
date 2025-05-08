from __future__ import annotations

import asyncio
import contextvars
import heapq
import time
from collections.abc import AsyncIterable, Coroutine
from typing import TYPE_CHECKING, Any

from livekit import rtc

from .. import debug, llm, stt, tts, utils, vad
from ..llm.tool_context import StopResponse
from ..log import logger
from ..metrics import EOUMetrics, LLMMetrics, STTMetrics, TTSMetrics, VADMetrics
from ..tokenize.basic import split_words
from ..types import NOT_GIVEN, NotGivenOr
from ..utils.misc import is_given
from .agent import Agent, ModelSettings
from .audio_recognition import AudioRecognition, RecognitionHooks, _EndOfTurnInfo
from .events import (
    ErrorEvent,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
)
from .generation import (
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

try:
    from livekit.plugins.google.beta.realtime.realtime_api import (
        RealtimeModel as GoogleRealtimeModel,
    )

except ImportError:
    GoogleRealtimeModel = None


def log_event(event: str, **kwargs) -> None:
    debug.Tracing.log_event(event, kwargs)


if TYPE_CHECKING:
    from ..llm import mcp
    from .agent_session import AgentSession, TurnDetectionMode


_AgentActivityContextVar = contextvars.ContextVar["AgentActivity"]("agents_activity")
_SpeechHandleContextVar = contextvars.ContextVar["SpeechHandle"]("agents_speech_handle")


# NOTE: AgentActivity isn't exposed to the public API
class AgentActivity(RecognitionHooks):
    def __init__(self, agent: Agent, sess: AgentSession) -> None:
        self._agent, self._session = agent, sess
        self._rt_session: llm.RealtimeSession | None = None
        self._audio_recognition: AudioRecognition | None = None
        self._lock = asyncio.Lock()
        self._tool_choice: llm.ToolChoice | None = None

        self._started = False
        self._draining = False

        self._current_speech: SpeechHandle | None = None
        self._speech_q: list[tuple[int, float, SpeechHandle]] = []

        # fired when a speech_task finishes or when a new speech_handle is scheduled
        # this is used to wake up the main task when the scheduling state changes
        self._q_updated = asyncio.Event()

        self._main_atask: asyncio.Task | None = None
        self._user_turn_completed_atask: asyncio.Task | None = None
        self._speech_tasks: list[asyncio.Task] = []

        from .. import llm as large_language_model

        self._turn_detection_mode = (
            self.turn_detection if isinstance(self.turn_detection, str) else None
        )

        if self._turn_detection_mode == "vad" and not self.vad:
            logger.warning("turn_detection is set to 'vad', but no VAD model is provided")
            self._turn_detection_mode = None

        if self._turn_detection_mode == "stt" and not self.stt:
            logger.warning(
                "turn_detection is set to 'stt', but no STT model is provided, "
                "ignoring the turn_detection setting"
            )
            self._turn_detection_mode = None

        if isinstance(self.llm, large_language_model.RealtimeModel):
            if self.llm.capabilities.turn_detection and not self.allow_interruptions:
                raise ValueError(
                    "the RealtimeModel uses a server-side turn detection, "
                    "allow_interruptions cannot be False, disable turn_detection in "
                    "the RealtimeModel and use VAD on the AgentSession instead"
                )

            if (
                self._turn_detection_mode == "realtime_llm"
                and not self.llm.capabilities.turn_detection
            ):
                logger.warning(
                    "turn_detection is set to 'realtime_llm', but the LLM is not a RealtimeModel "
                    "or the server-side turn detection is not supported/enabled, "
                    "ignoring the turn_detection setting"
                )
                self._turn_detection_mode = None

            if self._turn_detection_mode == "stt":
                logger.warning(
                    "turn_detection is set to 'stt', but the LLM is a RealtimeModel, "
                    "ignoring the turn_detection setting"
                )
                self._turn_detection_mode = None

            elif (
                self._turn_detection_mode
                and self._turn_detection_mode != "realtime_llm"
                and self.llm.capabilities.turn_detection
            ):
                logger.warning(
                    f"turn_detection is set to '{self._turn_detection_mode}', but the LLM "
                    "is a RealtimeModel and server-side turn detection enabled, "
                    "ignoring the turn_detection setting"
                )
                self._turn_detection_mode = None

            # fallback to VAD if server side turn detection is disabled and VAD is available
            if (
                not self.llm.capabilities.turn_detection
                and self.vad
                and self._turn_detection_mode is None
            ):
                self._turn_detection_mode = "vad"
        elif self._turn_detection_mode == "realtime_llm":
            logger.warning(
                "turn_detection is set to 'realtime_llm', but the LLM is not a RealtimeModel"
            )
            self._turn_detection_mode = None

        if (
            not self.vad
            and self.stt
            and isinstance(self.llm, llm.LLM)
            and self.allow_interruptions
            and self._turn_detection_mode is None
        ):
            logger.warning(
                "VAD is not set. Enabling VAD is recommended when using LLM and STT "
                "for more responsive interruption handling."
            )

        self._mcp_tools: list[mcp.MCPTool] = []

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def session(self) -> AgentSession:
        return self._session

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def turn_detection(self) -> TurnDetectionMode | None:
        return (
            self._agent.turn_detection
            if is_given(self._agent.turn_detection)
            else self._session.turn_detection
        )

    @property
    def stt(self) -> stt.STT | None:
        return self._agent.stt if is_given(self._agent.stt) else self._session.stt

    @property
    def llm(self) -> llm.LLM | llm.RealtimeModel | None:
        return self._agent.llm if is_given(self._agent.llm) else self._session.llm

    @property
    def tts(self) -> tts.TTS | None:
        return self._agent.tts if is_given(self._agent.tts) else self._session.tts

    @property
    def vad(self) -> vad.VAD | None:
        return self._agent.vad if is_given(self._agent.vad) else self._session.vad

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
    def realtime_llm_session(self) -> llm.RealtimeSession | None:
        return self._rt_session

    @property
    def current_speech(self) -> SpeechHandle | None:
        return self._current_speech

    @property
    def tools(self) -> list[llm.FunctionTool | llm.RawFunctionTool | mcp.MCPTool]:
        return self._agent.tools + self._mcp_tools

    async def update_instructions(self, instructions: str) -> None:
        self._agent._instructions = instructions

        if self._rt_session is not None:
            await self._rt_session.update_instructions(instructions)
        else:
            update_instructions(
                self._agent._chat_ctx, instructions=instructions, add_if_missing=True
            )

    async def update_tools(self, tools: list[llm.FunctionTool]) -> None:
        tools = list(set(tools))
        self._agent._tools = tools

        if self._rt_session is not None:
            await self._rt_session.update_tools(tools)

        if isinstance(self.llm, llm.LLM):
            # for realtime LLM, we assume the server will remove unvalid tool messages
            await self.update_chat_ctx(self._agent._chat_ctx.copy(tools=tools))

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        chat_ctx = chat_ctx.copy(tools=self.tools)

        self._agent._chat_ctx = chat_ctx

        if self._rt_session is not None:
            remove_instructions(chat_ctx)
            await self._rt_session.update_chat_ctx(chat_ctx)
        else:
            update_instructions(
                chat_ctx, instructions=self._agent.instructions, add_if_missing=True
            )

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        if utils.is_given(tool_choice):
            self._tool_choice = tool_choice

        if self._rt_session is not None:
            self._rt_session.update_options(tool_choice=self._tool_choice)

    def _create_speech_task(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        owned_speech_handle: SpeechHandle | None = None,
        name: str | None = None,
    ) -> asyncio.Task:
        """
        This method must only be used for tasks that "could" create a new SpeechHandle.
        When draining, every task created with this method will be awaited.
        """
        # https://github.com/python/cpython/pull/31837 alternative impl
        tk = _AgentActivityContextVar.set(self)

        task = asyncio.create_task(coro, name=name)
        self._speech_tasks.append(task)
        task.add_done_callback(lambda _: self._speech_tasks.remove(task))

        if owned_speech_handle is not None:
            # make sure to finish playout in case something goes wrong
            # the tasks should normally do this before their function calls
            task.add_done_callback(lambda _: owned_speech_handle._mark_playout_done())

        task.add_done_callback(lambda _: self._wake_up_main_task())
        _AgentActivityContextVar.reset(tk)
        return task

    def _wake_up_main_task(self) -> None:
        self._q_updated.set()

    # TODO(theomonnom): Shoukd pause and resume call on_enter and on_exit? probably not
    async def pause(self) -> None:
        pass

    async def resume(self) -> None:
        pass

    async def start(self) -> None:
        from .agent import _authorize_inline_task

        async with self._lock:
            self._agent._activity = self

            if self.mcp_servers:

                @utils.log_exceptions(logger=logger)
                async def _list_mcp_tools_task(mcp_server: mcp.MCPServer) -> list[mcp.MCPTool]:
                    if not mcp_server.initialized:
                        await mcp_server.initialize()

                    return await mcp_server.list_tools()

                gathered = await asyncio.gather(
                    *(_list_mcp_tools_task(s) for s in self.mcp_servers), return_exceptions=True
                )
                tools: list[mcp.MCPTool] = []
                for mcp_server, res in zip(self.mcp_servers, gathered):
                    if isinstance(res, BaseException):
                        logger.error(
                            f"Failed to list tools from MCP server {mcp_server}", exc_info=res
                        )
                    else:
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

            elif isinstance(self.llm, llm.LLM):
                try:
                    update_instructions(
                        self._agent._chat_ctx,
                        instructions=self._agent.instructions,
                        add_if_missing=True,
                    )
                except ValueError:
                    logger.exception("failed to update the instructions")

            # metrics and error handling
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

            self._main_atask = asyncio.create_task(self._main_task(), name="_main_task")
            self._audio_recognition = AudioRecognition(
                hooks=self,
                stt=self._agent.stt_node if self.stt else None,
                vad=self.vad,
                turn_detector=(
                    self.turn_detection if not isinstance(self.turn_detection, str) else None
                ),
                min_endpointing_delay=self._session.options.min_endpointing_delay,
                max_endpointing_delay=self._session.options.max_endpointing_delay,
                manual_turn_detection=self._turn_detection_mode == "manual",
            )
            self._audio_recognition.start()
            self._started = True

            task = self._create_speech_task(self._agent.on_enter(), name="AgentTask_on_enter")
            _authorize_inline_task(task)

    async def drain(self) -> None:
        from .agent import _authorize_inline_task

        async with self._lock:
            if self._draining:
                return

            task = self._create_speech_task(self._agent.on_exit(), name="AgentTask_on_exit")
            _authorize_inline_task(task)

            self._wake_up_main_task()
            self._draining = True
            if self._main_atask is not None:
                await asyncio.shield(self._main_atask)

    async def aclose(self) -> None:
        async with self._lock:
            if not self._draining:
                logger.warning("task closing without draining")

            # Unregister event handlers to prevent duplicate metrics
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

            if self._audio_recognition is not None:
                await self._audio_recognition.aclose()

            if self._main_atask is not None:
                await utils.aio.cancel_and_wait(self._main_atask)

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

        self._create_speech_task(
            self._tts_task(
                speech_handle=handle,
                text=text,
                audio=audio or None,
                add_to_chat_ctx=add_to_chat_ctx,
                model_settings=ModelSettings(),
            ),
            owned_speech_handle=handle,
            name="AgentActivity.tts_say",
        )
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

        log_event(
            "generate_reply",
            new_message=user_message.text_content if user_message else None,
            instructions=instructions or None,
        )

        from .agent import _get_inline_task_info

        task = asyncio.current_task()
        if not is_given(tool_choice) and task is not None:
            if task_info := _get_inline_task_info(task):
                if task_info.function_call is not None:
                    # when generate_reply is called inside a function_tool, set tool_choice to None by default  # noqa: E501
                    tool_choice = "none"

        handle = SpeechHandle.create(
            allow_interruptions=allow_interruptions
            if is_given(allow_interruptions)
            else self.allow_interruptions
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
                    model_settings=ModelSettings(tool_choice=tool_choice),
                ),
                owned_speech_handle=handle,
                name="AgentActivity.realtime_reply",
            )

        elif isinstance(self.llm, llm.LLM):
            # instructions used inside generate_reply are "extra" instructions.
            # this matches the behavior of the Realtime API:
            # https://platform.openai.com/docs/api-reference/realtime-client-events/response/create
            if instructions:
                instructions = "\n".join([self._agent.instructions, instructions])

            self._create_speech_task(
                self._pipeline_reply_task(
                    speech_handle=handle,
                    chat_ctx=chat_ctx or self._agent._chat_ctx,
                    tools=self.tools,
                    new_message=user_message.model_copy() if user_message else None,
                    instructions=instructions or None,
                    model_settings=ModelSettings(
                        tool_choice=tool_choice
                        if utils.is_given(tool_choice) or self._tool_choice is None
                        else self._tool_choice
                    ),
                ),
                owned_speech_handle=handle,
                name="AgentActivity.pipeline_reply",
            )

        self._schedule_speech(handle, SpeechHandle.SPEECH_PRIORITY_NORMAL)
        return handle

    def interrupt(self) -> asyncio.Future:
        """Interrupt the current speech generation and any queued speeches.

        Returns:
            An asyncio.Future that completes when the interruption is fully processed
            and chat context has been updated
        """
        future = asyncio.Future()
        current_speech = self._current_speech

        if current_speech is not None:
            current_speech = current_speech.interrupt()

        for speech in self._speech_q:
            _, _, speech = speech
            speech.interrupt()

        if self._rt_session is not None:
            self._rt_session.interrupt()

        if current_speech is None:
            future.set_result(None)
        else:

            def on_playout_done(sh: SpeechHandle) -> None:
                if future.done():
                    return

                future.set_result(None)

            current_speech.add_done_callback(on_playout_done)
            if current_speech.done():
                future.set_result(None)

        return future

    def clear_user_turn(self) -> None:
        if self._audio_recognition:
            self._audio_recognition.clear_user_turn()

        if self._rt_session is not None:
            self._rt_session.clear_audio()

    def commit_user_turn(self) -> None:
        assert self._audio_recognition is not None
        self._audio_recognition.commit_user_turn(
            audio_detached=not self._session.input.audio_enabled
        )

    def _schedule_speech(
        self, speech: SpeechHandle, priority: int, bypass_draining: bool = False
    ) -> None:
        """
        This method is used to schedule a new speech.

        Args:
            bypass_draining: bypass_draining should only be used to allow the last tool response to be scheduled

        Raises RuntimeError if the agent is draining
        """  # noqa: E501
        if self.draining and not bypass_draining:
            raise RuntimeError("cannot schedule new speech, the agent is draining")

        heapq.heappush(self._speech_q, (priority, time.time(), speech))
        self._wake_up_main_task()

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        while True:
            await self._q_updated.wait()
            while self._speech_q:
                _, _, speech = heapq.heappop(self._speech_q)
                self._current_speech = speech
                speech._authorize_playout()
                await speech.wait_for_playout()
                self._current_speech = None

            # If we're draining and there are no more speech tasks, we can exit.
            # Only speech tasks can bypass draining to create a tool response
            if self._draining and len(self._speech_tasks) == 0:
                break

            self._q_updated.clear()

    # -- Realtime Session events --

    def _on_metrics_collected(self, ev: STTMetrics | TTSMetrics | VADMetrics | LLMMetrics) -> None:
        if (speech_handle := _SpeechHandleContextVar.get(None)) and (
            isinstance(ev, LLMMetrics) or isinstance(ev, TTSMetrics)
        ):
            ev.speech_id = speech_handle.id
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
        log_event("input_speech_started")

        if self.vad is None:
            self._session._update_user_state("speaking")

        # self.interrupt() isn't going to raise when allow_interruptions is False, llm.InputSpeechStartedEvent is only fired by the server when the turn_detection is enabled.  # noqa: E501
        # When using the server-side turn_detection, we don't allow allow_interruptions to be False.
        try:
            self.interrupt()  # input_speech_started is also interrupting on the serverside realtime session  # noqa: E501
        except RuntimeError:
            logger.exception(
                "RealtimeAPI input_speech_started, but current speech is not interruptable, this should never happen!"  # noqa: E501
            )

    def _on_input_speech_stopped(self, ev: llm.InputSpeechStoppedEvent) -> None:
        log_event("input_speech_stopped")

        if self.vad is None:
            self._session._update_user_state("listening")

        if ev.user_transcription_enabled:
            self._session.emit(
                "user_input_transcribed",
                UserInputTranscribedEvent(transcript="", is_final=False),
            )

    def _on_input_audio_transcription_completed(self, ev: llm.InputTranscriptionCompleted) -> None:
        log_event("input_audio_transcription_completed")
        self._session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript=ev.transcript, is_final=True),
        )
        msg = llm.ChatMessage(role="user", content=[ev.transcript], id=ev.item_id)
        self._agent._chat_ctx.items.append(msg)
        self._session._conversation_item_added(msg)

    def _on_generation_created(self, ev: llm.GenerationCreatedEvent) -> None:
        if ev.user_initiated:
            # user_initiated generations are directly handled inside _realtime_reply_task
            return

        if self.draining:
            # TODO(theomonnom): should we "forward" this new turn to the next agent?
            logger.warning("skipping new realtime generation, the agent is draining")
            return

        handle = SpeechHandle.create(allow_interruptions=self.allow_interruptions)
        self._session.emit(
            "speech_created",
            SpeechCreatedEvent(speech_handle=handle, user_initiated=False, source="generate_reply"),
        )

        self._create_speech_task(
            self._realtime_generation_task(
                speech_handle=handle, generation_ev=ev, model_settings=ModelSettings()
            ),
            owned_speech_handle=handle,
            name="AgentActivity.realtime_generation",
        )
        self._schedule_speech(handle, SpeechHandle.SPEECH_PRIORITY_NORMAL)

    # region recognition hooks

    def on_start_of_speech(self, ev: vad.VADEvent) -> None:
        self._session._update_user_state("speaking")

    def on_end_of_speech(self, ev: vad.VADEvent) -> None:
        self._session._update_user_state("listening")

    def on_vad_inference_done(self, ev: vad.VADEvent) -> None:
        if self._turn_detection_mode not in ("vad", None):
            # ignore vad inference done event if turn_detection is not set to vad or default
            return

        if isinstance(self.llm, llm.RealtimeModel) and self.llm.capabilities.turn_detection:
            # ignore if turn_detection is enabled on the realtime model
            return

        if ev.speech_duration < self._session.options.min_interruption_duration:
            return

        if (
            self.stt is not None
            and self._session.options.min_interruption_words > 0
            and self._audio_recognition is not None
        ):
            text = self._audio_recognition.current_transcript

            # TODO(long): better word splitting for multi-language
            if len(split_words(text)) < self._session.options.min_interruption_words:
                return

        if (
            self._current_speech is not None
            and not self._current_speech.interrupted
            and self._current_speech.allow_interruptions
        ):
            log_event(
                "speech interrupted by vad",
                speech_id=self._current_speech.id,
            )
            if self._rt_session is not None:
                self._rt_session.interrupt()

            self._current_speech.interrupt()

    def on_interim_transcript(self, ev: stt.SpeechEvent) -> None:
        if isinstance(self.llm, llm.RealtimeModel) and self.llm.capabilities.user_transcription:
            # skip stt transcription if user_transcription is enabled on the realtime model
            return

        self._session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript=ev.alternatives[0].text, is_final=False),
        )

    def on_final_transcript(self, ev: stt.SpeechEvent) -> None:
        if isinstance(self.llm, llm.RealtimeModel) and self.llm.capabilities.user_transcription:
            # skip stt transcription if user_transcription is enabled on the realtime model
            return

        self._session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript=ev.alternatives[0].text, is_final=True),
        )

    async def on_end_of_turn(self, info: _EndOfTurnInfo) -> bool:
        # IMPORTANT: This method can be cancelled by the AudioRecognition
        # We explicitly create a new task to avoid cancelling user code.

        if self.draining:
            logger.warning(
                "skipping user input, task is draining",
                extra={"user_input": info.new_transcript},
            )
            log_event(
                "skipping user input, task is draining",
                user_input=info.new_transcript,
            )
            # TODO(theomonnom): should we "forward" this new turn to the next agent/activity?
            return True

        if (
            self.stt is not None
            and self._turn_detection_mode != "manual"
            and self._current_speech is not None
            and self._current_speech.allow_interruptions
            and not self._current_speech.interrupted
            and self._session.options.min_interruption_words > 0
            and len(split_words(info.new_transcript)) < self._session.options.min_interruption_words
        ):
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
        self, old_task: asyncio.Task | None, info: _EndOfTurnInfo
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

            log_event(
                "speech interrupted, new user turn detected",
                speech_id=self._current_speech.id,
            )
            self._current_speech.interrupt()
            if self._rt_session is not None:
                self._rt_session.interrupt()

        # id is generated
        user_message = llm.ChatMessage(role="user", content=[info.new_transcript])

        # create a temporary mutable chat context to pass to on_user_turn_completed
        # the user can edit it for the current generation, but changes will not be kept inside the
        # Agent.chat_ctx
        temp_mutable_chat_ctx = self._agent.chat_ctx.copy()
        start_time = time.time()
        try:
            await self._agent.on_user_turn_completed(
                temp_mutable_chat_ctx, new_message=user_message
            )
        except StopResponse:
            return  # ignore this turn
        except Exception:
            logger.exception("error occured during on_user_turn_completed")
            return

        callback_duration = time.time() - start_time

        if isinstance(self.llm, llm.RealtimeModel):
            # ignore stt transcription for realtime model
            user_message = None

        # Ensure the new message is passed to generate_reply
        # This preserves the original message_id, making it easier for users to track responses
        speech_handle = self._generate_reply(
            user_message=user_message, chat_ctx=temp_mutable_chat_ctx
        )

        if self._user_turn_completed_atask != asyncio.current_task():
            # If a new user turn has already started, interrupt this one since it's now outdated
            # (We still create the SpeechHandle and the generate_reply coroutine, otherwise we may
            # lose data like the beginning of a user speech).
            speech_handle.interrupt()

        eou_metrics = EOUMetrics(
            timestamp=time.time(),
            end_of_utterance_delay=info.end_of_utterance_delay,
            transcription_delay=info.transcription_delay,
            on_user_turn_completed_delay=callback_duration,
            speech_id=speech_handle.id,
        )
        self._session.emit("metrics_collected", MetricsCollectedEvent(metrics=eou_metrics))

    # AudioRecognition is calling this method to retrieve the chat context before running the TurnDetector model  # noqa: E501
    def retrieve_chat_ctx(self) -> llm.ChatContext:
        return self._agent.chat_ctx

    # endregion

    @utils.log_exceptions(logger=logger)
    async def _tts_task(
        self,
        speech_handle: SpeechHandle,
        text: str | AsyncIterable[str],
        audio: AsyncIterable[rtc.AudioFrame] | None,
        add_to_chat_ctx: bool,
        model_settings: ModelSettings,
    ) -> None:
        _SpeechHandleContextVar.set(speech_handle)

        tr_output = (
            self._session.output.transcription
            if self._session.output.transcription_enabled
            else None
        )
        audio_output = self._session.output.audio if self._session.output.audio_enabled else None

        await speech_handle.wait_if_not_interrupted(
            [asyncio.ensure_future(speech_handle._wait_for_authorization())]
        )

        if speech_handle.interrupted:
            return

        text_source: AsyncIterable[str] | None = None
        audio_source: AsyncIterable[str] | None = None

        if isinstance(text, AsyncIterable):
            text_source, audio_source = utils.aio.itertools.tee(text, 2)
        elif isinstance(text, str):

            async def _read_text() -> AsyncIterable[str]:
                yield text

            text_source = _read_text()
            audio_source = _read_text()

        tasks = []

        tr_node = self._agent.transcription_node(text_source, model_settings)
        if asyncio.iscoroutine(tr_node):
            tr_node = await tr_node

        forward_text, text_out = perform_text_forwarding(
            text_output=tr_output,
            source=tr_node,
        )
        tasks.append(forward_text)

        def _on_first_frame(_: asyncio.Future) -> None:
            self._session._update_agent_state("speaking")

        if audio_output is None:
            # update the agent state based on text if no audio output
            text_out.first_text_fut.add_done_callback(_on_first_frame)
        else:
            if audio is None:
                # generate audio using TTS
                tts_task, tts_gen_data = perform_tts_inference(
                    node=self._agent.tts_node,
                    input=audio_source,
                    model_settings=model_settings,
                )
                tasks.append(tts_task)

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

        await speech_handle.wait_if_not_interrupted([*tasks])

        if audio_output is not None:
            await speech_handle.wait_if_not_interrupted(
                [asyncio.ensure_future(audio_output.wait_for_playout())]
            )

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*tasks)

            if audio_output is not None:
                audio_output.clear_buffer()
                await audio_output.wait_for_playout()

        if add_to_chat_ctx:
            msg = self._agent._chat_ctx.add_message(
                role="assistant", content=text_out.text, interrupted=speech_handle.interrupted
            )
            speech_handle._set_chat_message(msg)
            self._session._conversation_item_added(msg)

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
        _tools_messages: list[llm.ChatItem] | None = None,
    ) -> None:
        from .agent import ModelSettings

        _SpeechHandleContextVar.set(speech_handle)

        log_event(
            "generation started",
            speech_id=speech_handle.id,
            step_index=speech_handle.step_index,
        )

        audio_output = self._session.output.audio if self._session.output.audio_enabled else None
        text_output = (
            self._session.output.transcription
            if self._session.output.transcription_enabled
            else None
        )
        chat_ctx = chat_ctx.copy()
        tool_ctx = llm.ToolContext(tools)

        if new_message is not None:
            idx = chat_ctx.find_insertion_index(created_at=new_message.created_at)
            chat_ctx.items.insert(idx, new_message)
            idx = self._agent._chat_ctx.find_insertion_index(created_at=new_message.created_at)
            self._agent._chat_ctx.items.insert(idx, new_message)
            self._session._conversation_item_added(new_message)

        if instructions is not None:
            try:
                update_instructions(chat_ctx, instructions=instructions, add_if_missing=True)
            except ValueError:
                logger.exception("failed to update the instructions")

        self._session._update_agent_state("thinking")
        tasks = []
        llm_task, llm_gen_data = perform_llm_inference(
            node=self._agent.llm_node,
            chat_ctx=chat_ctx,
            tool_ctx=tool_ctx,
            model_settings=model_settings,
        )
        tasks.append(llm_task)
        tts_text_input, llm_output = utils.aio.itertools.tee(llm_gen_data.text_ch)

        tts_task: asyncio.Task | None = None
        tts_gen_data: _TTSGenerationData | None = None
        if audio_output is not None:
            tts_task, tts_gen_data = perform_tts_inference(
                node=self._agent.tts_node,
                input=tts_text_input,
                model_settings=model_settings,
            )
            tasks.append(tts_task)

        await speech_handle.wait_if_not_interrupted(
            [asyncio.ensure_future(speech_handle._wait_for_authorization())]
        )

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*tasks)
            return

        tr_node = self._agent.transcription_node(llm_output, model_settings)
        if asyncio.iscoroutine(tr_node):
            tr_node = await tr_node

        forward_task, text_out = perform_text_forwarding(text_output=text_output, source=tr_node)
        tasks.append(forward_task)

        def _on_first_frame(_: asyncio.Future) -> None:
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
        else:
            text_out.first_text_fut.add_done_callback(_on_first_frame)

        # start to execute tools (only after play())
        exe_task, tool_output = perform_tool_executions(
            session=self._session,
            speech_handle=speech_handle,
            tool_ctx=tool_ctx,
            tool_choice=model_settings.tool_choice,
            function_stream=llm_gen_data.function_ch,
        )

        await speech_handle.wait_if_not_interrupted([*tasks])

        # wait for the end of the playout if the audio is enabled
        if audio_output is not None:
            await speech_handle.wait_if_not_interrupted(
                [asyncio.ensure_future(audio_output.wait_for_playout())]
            )
            if not speech_handle.interrupted:
                self._session._update_agent_state("listening")

        # add the tools messages that triggers this reply to the chat context
        if _tools_messages:
            self._agent._chat_ctx.items.extend(_tools_messages)

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*tasks)

            forwarded_text = text_out.text
            # if the audio playout was enabled, clear the buffer
            if audio_output is not None:
                audio_output.clear_buffer()

                playback_ev = await audio_output.wait_for_playout()
                if audio_out is not None and audio_out.first_frame_fut.done():
                    # playback_ev is valid only if the first frame was already played
                    log_event(
                        "playout interrupted",
                        playback_position=playback_ev.playback_position,
                        speech_id=speech_handle.id,
                    )
                    if playback_ev.synchronized_transcript is not None:
                        forwarded_text = playback_ev.synchronized_transcript
                else:
                    forwarded_text = ""

            msg = chat_ctx.add_message(
                role="assistant",
                content=forwarded_text,
                id=llm_gen_data.id,
                interrupted=True,
            )
            self._agent._chat_ctx.items.append(msg)
            self._session._update_agent_state("listening")
            self._session._conversation_item_added(msg)
            speech_handle._set_chat_message(msg)
            speech_handle._mark_playout_done()
            await utils.aio.cancel_and_wait(exe_task)
            return

        if text_out.text:
            msg = chat_ctx.add_message(
                role="assistant", content=text_out.text, id=llm_gen_data.id, interrupted=False
            )
            self._agent._chat_ctx.items.append(msg)
            self._session._conversation_item_added(msg)
            speech_handle._set_chat_message(msg)

        self._session._update_agent_state("listening")
        log_event("playout completed", speech_id=speech_handle.id)

        speech_handle._mark_playout_done()  # mark the playout done before waiting for the tool execution  # noqa: E501

        tool_output.first_tool_fut.add_done_callback(
            lambda _: self._session._update_agent_state("thinking")
        )
        await exe_task

        # important: no agent output should be used after this point

        if len(tool_output.output) > 0:
            if speech_handle.step_index >= self._session.options.max_tool_steps:
                logger.warning(
                    "maximum number of function calls steps reached",
                    extra={"speech_id": speech_handle.id},
                )
                log_event(
                    "maximum number of function calls steps reached",
                    speech_id=speech_handle.id,
                )
                return

            new_calls: list[llm.FunctionCall] = []
            new_fnc_outputs: list[llm.FunctionCallOutput] = []
            generate_tool_reply: bool = False
            new_agent_task: Agent | None = None
            ignore_task_switch = False
            fnc_executed_ev = FunctionToolsExecutedEvent(
                function_calls=[],
                function_call_outputs=[],
            )
            for py_out in tool_output.output:
                sanitized_out = py_out.sanitize()

                if sanitized_out.fnc_call_out is not None:
                    new_calls.append(sanitized_out.fnc_call)
                    new_fnc_outputs.append(sanitized_out.fnc_call_out)
                    if sanitized_out.reply_required:
                        generate_tool_reply = True

                # add the function call and output to the event, including the None outputs
                fnc_executed_ev.function_calls.append(sanitized_out.fnc_call)
                fnc_executed_ev.function_call_outputs.append(sanitized_out.fnc_call_out)

                if new_agent_task is not None and sanitized_out.agent_task is not None:
                    logger.error(
                        "expected to receive only one AgentTask from the tool executions",
                    )
                    ignore_task_switch = True
                    # TODO(long): should we mark the function call as failed to notify the LLM?

                new_agent_task = sanitized_out.agent_task
            self._session.emit("function_tools_executed", fnc_executed_ev)

            draining = self.draining
            if not ignore_task_switch and new_agent_task is not None:
                self._session.update_agent(new_agent_task)
                draining = True

            if generate_tool_reply:
                chat_ctx.items.extend(new_calls)
                chat_ctx.items.extend(new_fnc_outputs)

                handle = SpeechHandle.create(
                    allow_interruptions=speech_handle.allow_interruptions,
                    step_index=speech_handle.step_index + 1,
                    parent=speech_handle,
                )
                self._session.emit(
                    "speech_created",
                    SpeechCreatedEvent(
                        speech_handle=handle, user_initiated=False, source="tool_response"
                    ),
                )
                self._create_speech_task(
                    self._pipeline_reply_task(
                        speech_handle=handle,
                        chat_ctx=chat_ctx,
                        tools=tools,
                        model_settings=ModelSettings(
                            tool_choice=model_settings.tool_choice if not draining else "none",
                        ),
                        _tools_messages=[*new_calls, *new_fnc_outputs],
                    ),
                    owned_speech_handle=handle,
                    name="AgentActivity.pipeline_reply",
                )
                self._schedule_speech(
                    handle, SpeechHandle.SPEECH_PRIORITY_NORMAL, bypass_draining=True
                )
            elif len(new_fnc_outputs) > 0:
                # add the tool calls and outputs to the chat context even no reply is generated
                self._agent._chat_ctx.items.extend(new_calls)
                self._agent._chat_ctx.items.extend(new_fnc_outputs)

    @utils.log_exceptions(logger=logger)
    async def _realtime_reply_task(
        self,
        *,
        speech_handle: SpeechHandle,
        model_settings: ModelSettings,
        user_input: str | None = None,
        instructions: str | None = None,
    ) -> None:
        _SpeechHandleContextVar.set(speech_handle)  # not needed, but here for completeness

        assert self._rt_session is not None, "rt_session is not available"

        await speech_handle.wait_if_not_interrupted(
            [asyncio.ensure_future(speech_handle._wait_for_authorization())]
        )

        if user_input is not None:
            chat_ctx = self._rt_session.chat_ctx.copy()
            msg = chat_ctx.add_message(role="user", content=user_input)
            await self._rt_session.update_chat_ctx(chat_ctx)
            self._agent._chat_ctx.items.append(msg)
            self._session._conversation_item_added(msg)

        ori_tool_choice = self._tool_choice
        if utils.is_given(model_settings.tool_choice):
            self._rt_session.update_options(tool_choice=model_settings.tool_choice)

        try:
            generation_ev = await self._rt_session.generate_reply(
                instructions=instructions or NOT_GIVEN
            )

            await self._realtime_generation_task(
                speech_handle=speech_handle,
                generation_ev=generation_ev,
                model_settings=model_settings,
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
    ) -> None:
        _SpeechHandleContextVar.set(speech_handle)

        assert self._rt_session is not None, "rt_session is not available"

        log_event(
            "generation started",
            speech_id=speech_handle.id,
            step_index=speech_handle.step_index,
            realtime=True,
        )

        audio_output = self._session.output.audio if self._session.output.audio_enabled else None
        text_output = (
            self._session.output.transcription
            if self._session.output.transcription_enabled
            else None
        )
        tool_ctx = llm.ToolContext(self.tools)

        await speech_handle.wait_if_not_interrupted(
            [asyncio.ensure_future(speech_handle._wait_for_authorization())]
        )

        if speech_handle.interrupted:
            return  # TODO(theomonnom): remove the message from the serverside history

        def _on_first_frame(_: asyncio.Future) -> None:
            self._session._update_agent_state("speaking")

        @utils.log_exceptions(logger=logger)
        async def _read_messages(
            outputs: list[tuple[str, _TextOutput, _AudioOutput | None]],
        ) -> None:
            forward_tasks: list[asyncio.Task] = []
            try:
                async for msg in generation_ev.message_stream:
                    if len(forward_tasks) > 0:
                        logger.warning(
                            "expected to receive only one message generation from the realtime API"
                        )
                        break

                    tr_node = self._agent.transcription_node(msg.text_stream, model_settings)
                    if asyncio.iscoroutine(tr_node):
                        tr_node = await tr_node

                    forward_task, text_out = perform_text_forwarding(
                        text_output=text_output,
                        source=tr_node,
                    )
                    forward_tasks.append(forward_task)

                    audio_out = None
                    if audio_output is not None:
                        realtime_audio = self._agent.realtime_audio_output_node(
                            msg.audio_stream, model_settings
                        )
                        if asyncio.iscoroutine(realtime_audio):
                            realtime_audio = await realtime_audio
                        if realtime_audio is not None:
                            forward_task, audio_out = perform_audio_forwarding(
                                audio_output=audio_output, tts_output=realtime_audio
                            )
                            forward_tasks.append(forward_task)
                            audio_out.first_frame_fut.add_done_callback(_on_first_frame)
                    else:
                        text_out.first_text_fut.add_done_callback(_on_first_frame)

                    outputs.append((msg.message_id, text_out, audio_out))

                await asyncio.gather(*forward_tasks)
            finally:
                await utils.aio.cancel_and_wait(*forward_tasks)

        message_outputs: list[tuple[str, _TextOutput, _AudioOutput | None]] = []
        tasks = [
            asyncio.create_task(
                _read_messages(message_outputs),
                name="AgentActivity.realtime_generation.read_messages",
            )
        ]

        exe_task, tool_output = perform_tool_executions(
            session=self._session,
            speech_handle=speech_handle,
            tool_ctx=tool_ctx,
            tool_choice=model_settings.tool_choice,
            function_stream=generation_ev.function_stream,
        )

        await speech_handle.wait_if_not_interrupted([*tasks])

        if audio_output is not None:
            await speech_handle.wait_if_not_interrupted(
                [asyncio.ensure_future(audio_output.wait_for_playout())]
            )
            self._session._update_agent_state("listening")

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*tasks)

            if len(message_outputs) > 0:
                # there should be only one message
                msg_id, text_out, audio_out = message_outputs[0]
                forwarded_text = text_out.text
                if audio_output is not None:
                    audio_output.clear_buffer()

                    playback_ev = await audio_output.wait_for_playout()
                    playback_position = playback_ev.playback_position
                    if audio_out is not None and audio_out.first_frame_fut.done():
                        # playback_ev is valid only if the first frame was already played
                        log_event(
                            "playout interrupted",
                            playback_position=playback_ev.playback_position,
                            speech_id=speech_handle.id,
                        )
                        if playback_ev.synchronized_transcript is not None:
                            forwarded_text = playback_ev.synchronized_transcript
                    else:
                        forwarded_text = ""
                        playback_position = 0

                    # truncate server-side message
                    self._rt_session.truncate(
                        message_id=msg_id, audio_end_ms=int(playback_position * 1000)
                    )

                msg = llm.ChatMessage(
                    role="assistant", content=[forwarded_text], id=msg_id, interrupted=True
                )
                self._agent._chat_ctx.items.append(msg)
                speech_handle._set_chat_message(msg)
                self._session._conversation_item_added(msg)

            speech_handle._mark_playout_done()
            await utils.aio.cancel_and_wait(exe_task)
            return

        if len(message_outputs) > 0:
            # there should be only one message
            msg_id, text_out, _ = message_outputs[0]
            msg = llm.ChatMessage(
                role="assistant", content=[text_out.text], id=msg_id, interrupted=False
            )
            self._agent._chat_ctx.items.append(msg)
            speech_handle._set_chat_message(msg)
            self._session._conversation_item_added(msg)

        # mark playout must be done before _set_chat_message
        speech_handle._mark_playout_done()  # mark the playout done before waiting for the tool execution  # noqa: E501

        tool_output.first_tool_fut.add_done_callback(
            lambda _: self._session._update_agent_state("thinking")
        )
        await exe_task

        # important: no agent ouput should be used after this point

        if len(tool_output.output) > 0:
            new_fnc_outputs: list[llm.FunctionCallOutput] = []
            generate_tool_reply: bool = False
            fnc_executed_ev = FunctionToolsExecutedEvent(
                function_calls=[],
                function_call_outputs=[],
            )
            new_agent_task: Agent | None = None
            ignore_task_switch = False

            for py_out in tool_output.output:
                sanitized_out = py_out.sanitize()

                # add the function call and output to the event, including the None outputs
                fnc_executed_ev.function_calls.append(sanitized_out.fnc_call)
                fnc_executed_ev.function_call_outputs.append(sanitized_out.fnc_call_out)

                if sanitized_out.fnc_call_out is not None:
                    new_fnc_outputs.append(sanitized_out.fnc_call_out)
                    if sanitized_out.reply_required:
                        generate_tool_reply = True

                if new_agent_task is not None and sanitized_out.agent_task is not None:
                    logger.error(
                        "expected to receive only one AgentTask from the tool executions",
                    )
                    ignore_task_switch = True

                new_agent_task = sanitized_out.agent_task
            self._session.emit("function_tools_executed", fnc_executed_ev)

            draining = self.draining
            if not ignore_task_switch and new_agent_task is not None:
                self._session.update_agent(new_agent_task)
                draining = True

            if len(new_fnc_outputs) > 0:
                chat_ctx = self._rt_session.chat_ctx.copy()
                chat_ctx.items.extend(new_fnc_outputs)
                try:
                    await self._rt_session.update_chat_ctx(chat_ctx)
                except llm.RealtimeError as e:
                    logger.warning(
                        "failed to update chat context before generating the function calls results",  # noqa: E501
                        extra={"error": str(e)},
                    )

            if generate_tool_reply and (
                # no direct cancellation in Gemini
                GoogleRealtimeModel is None or not isinstance(self.llm, GoogleRealtimeModel)
            ):
                self._rt_session.interrupt()

                handle = SpeechHandle.create(
                    allow_interruptions=speech_handle.allow_interruptions,
                    step_index=speech_handle.step_index + 1,
                    parent=speech_handle,
                )
                self._session.emit(
                    "speech_created",
                    SpeechCreatedEvent(
                        speech_handle=handle,
                        user_initiated=False,
                        source="tool_response",
                    ),
                )
                self._create_speech_task(
                    self._realtime_reply_task(
                        speech_handle=handle,
                        model_settings=ModelSettings(
                            tool_choice=model_settings.tool_choice if not draining else "none",
                        ),
                    ),
                    owned_speech_handle=handle,
                    name="AgentActivity.realtime_reply",
                )
                self._schedule_speech(
                    handle,
                    SpeechHandle.SPEECH_PRIORITY_NORMAL,
                    bypass_draining=True,
                )
