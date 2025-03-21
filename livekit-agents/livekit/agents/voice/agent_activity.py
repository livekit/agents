from __future__ import annotations

import asyncio
import contextvars
import heapq
import time
from collections.abc import AsyncIterable, Coroutine
from typing import TYPE_CHECKING, Any

from livekit import rtc

from .. import debug, llm, stt, tts, utils, vad
from ..log import logger
from ..metrics import AgentMetrics
from ..types import NOT_GIVEN, AgentState, NotGivenOr
from ..utils.misc import is_given
from .audio_recognition import AudioRecognition, RecognitionHooks, _TurnDetector
from .events import (
    MetricsCollectedEvent,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
    UserStartedSpeakingEvent,
    UserStoppedSpeakingEvent,
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
    truncate_message,
    update_instructions,
)
from .speech_handle import SpeechHandle


def log_event(event: str, **kwargs) -> None:
    debug.Tracing.log_event(event, kwargs)


from .agent import Agent, ModelSettings

if TYPE_CHECKING:
    from .agent_session import AgentSession


_AgentActivityContextVar = contextvars.ContextVar["AgentActivity"]("agents_activity")
_SpeechHandleContextVar = contextvars.ContextVar["SpeechHandle"]("agents_speech_handle")


# NOTE: AgentActivity isn't exposed to the public API
class AgentActivity(RecognitionHooks):
    def __init__(self, agent: Agent, sess: AgentSession) -> None:
        self._agent, self._session = agent, sess
        self._rt_session: llm.RealtimeSession | None = None
        self._audio_recognition: AudioRecognition | None = None
        self._lock = asyncio.Lock()

        self._started = False
        self._draining = False

        self._current_speech: SpeechHandle | None = None
        self._speech_q: list[tuple[int, float, SpeechHandle]] = []

        # fired when a speech_task finishes or when a new speech_handle is scheduled
        # this is used to wake up the main task when the scheduling state changes
        self._q_updated = asyncio.Event()

        self._main_atask: asyncio.Task | None = None
        self._speech_tasks: list[asyncio.Task] = []

        from .. import llm as large_language_model

        if (
            isinstance(self.llm, large_language_model.RealtimeModel)
            and self.llm.capabilities.turn_detection
            and not self.allow_interruptions
        ):
            raise ValueError(
                "the RealtimeModel uses a server-side turn detection, allow_interruptions cannot be False, "
                "disable turn_detection in the RealtimeModel and use VAD on the AgentTask/VoiceAgent instead"
            )

        if not self.vad and self.stt and isinstance(self.llm, llm.LLM) and self.allow_interruptions:
            logger.warning(
                "VAD is not set. Enabling VAD is recommended when using LLM and STT "
                "for more responsive interruption handling."
            )

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def agent(self) -> AgentSession:
        return self._session

    @property
    def turn_detector(self) -> _TurnDetector | None:
        return self._agent._eou or self._session._turn_detector

    @property
    def stt(self) -> stt.STT | None:
        return self._agent.stt or self._session.stt

    @property
    def llm(self) -> llm.LLM | llm.RealtimeModel | None:
        return self._agent.llm or self._session.llm

    @property
    def tts(self) -> tts.TTS | None:
        return self._agent.tts or self._session.tts

    @property
    def vad(self) -> vad.VAD | None:
        return self._agent.vad or self._session.vad

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

    async def update_instructions(self, instructions: str) -> None:
        self._agent._instructions = instructions

        if self._rt_session is not None:
            await self._rt_session.update_instructions(instructions)

    async def update_tools(self, tools: list[llm.FunctionTool]) -> None:
        tools = list(set(tools))
        self._agent._tools = tools

        if self._rt_session is not None:
            await self._rt_session.update_tools(tools)

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        chat_ctx = chat_ctx.copy()
        self._agent._chat_ctx = chat_ctx
        update_instructions(chat_ctx, instructions=self._agent.instructions, add_if_missing=True)

        if self._rt_session is not None:
            await self._rt_session.update_chat_ctx(chat_ctx)

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

            if isinstance(self.llm, llm.RealtimeModel):
                self._rt_session = self.llm.session()
                self._rt_session.on("generation_created", self._on_generation_created)
                self._rt_session.on("input_speech_started", self._on_input_speech_started)
                self._rt_session.on("input_speech_stopped", self._on_input_speech_stopped)
                self._rt_session.on(
                    "input_audio_transcription_completed",
                    self._on_input_audio_transcription_completed,
                )
                try:
                    await self._rt_session.update_instructions(self._agent.instructions)
                except llm.RealtimeError:
                    logger.exception("failed to update the instructions")

                try:
                    await self._rt_session.update_chat_ctx(self._agent.chat_ctx)
                except llm.RealtimeError:
                    logger.exception("failed to update the chat_ctx")

                try:
                    await self._rt_session.update_tools(self._agent.tools)
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

            # metrics
            if isinstance(self.llm, llm.LLM):
                self.llm.on("metrics_collected", self._on_metrics_collected)

            if isinstance(self.stt, stt.STT):
                self.stt.on("metrics_collected", self._on_metrics_collected)

            if isinstance(self.tts, tts.TTS):
                self.tts.on("metrics_collected", self._on_metrics_collected)

            if isinstance(self.vad, vad.VAD):
                self.vad.on("metrics_collected", self._on_metrics_collected)

            self._main_atask = asyncio.create_task(self._main_task(), name="_main_task")
            self._audio_recognition = AudioRecognition(
                hooks=self,
                stt=self._agent.stt_node if self.stt else None,
                vad=self.vad,
                turn_detector=self.turn_detector,
                min_endpointing_delay=self._session.options.min_endpointing_delay,
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

        if self._current_speech and not self._current_speech.allow_interruptions:
            # drop the audio if the current speech is not interruptable
            return

        if self._rt_session is not None:
            self._rt_session.push_audio(frame)

        if self._audio_recognition is not None:
            self._audio_recognition.push_audio(frame)

    def say(
        self,
        text: str | AsyncIterable[str],
        *,
        audio: NotGivenOr[AsyncIterable[rtc.AudioFrame]] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        add_to_chat_ctx: bool = True,
    ) -> SpeechHandle:
        if not is_given(audio) and not self.tts:
            raise RuntimeError("trying to generate speech from text without a TTS model")

        if (
            isinstance(self.llm, llm.RealtimeModel)
            and self.llm.capabilities.turn_detection
            and allow_interruptions is False
        ):
            logger.warning(
                "the RealtimeModel uses a server-side turn detection, allow_interruptions cannot be False when using VoiceAgent.say(), "
                "disable turn_detection in the RealtimeModel and use VAD on the AgentTask/VoiceAgent instead"
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

    def generate_reply(
        self,
        *,
        user_input: NotGivenOr[str] = NOT_GIVEN,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> SpeechHandle:
        if self._current_speech is not None and not self._current_speech.interrupted:
            raise RuntimeError("another reply is already in progress")

        if (
            isinstance(self.llm, llm.RealtimeModel)
            and self.llm.capabilities.turn_detection
            and allow_interruptions is False
        ):
            logger.warning(
                "the RealtimeModel uses a server-side turn detection, allow_interruptions cannot be False when using VoiceAgent.generate_reply(), "
                "disable turn_detection in the RealtimeModel and use VAD on the AgentTask/VoiceAgent instead"
            )
            allow_interruptions = NOT_GIVEN

        log_event(
            "generate_reply", user_input=user_input or None, instructions=instructions or None
        )

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
                    user_input=user_input or None,
                    instructions=instructions or None,
                    model_settings=ModelSettings(),
                ),
                owned_speech_handle=handle,
                name="AgentActivity.realtime_reply",
            )

        elif isinstance(self.llm, llm.LLM):
            self._create_speech_task(
                self._pipeline_reply_task(
                    speech_handle=handle,
                    chat_ctx=self._agent._chat_ctx,
                    tools=self._agent.tools,
                    user_input=user_input or None,
                    instructions=instructions or None,
                    model_settings=ModelSettings(),
                ),
                owned_speech_handle=handle,
                name="AgentActivity.pipeline_reply",
            )

        self._schedule_speech(handle, SpeechHandle.SPEECH_PRIORITY_NORMAL)
        return handle

    def interrupt(self) -> None:
        if self._current_speech is not None:
            self._current_speech.interrupt()

        for speech in self._speech_q:
            _, _, speech = speech
            speech.interrupt()

        if self._rt_session is not None:
            self._rt_session.interrupt()

    def _schedule_speech(
        self, speech: SpeechHandle, priority: int, bypass_draining: bool = False
    ) -> None:
        """
        This method is used to schedule a new speech.

        Args:
            bypass_draining: bypass_draining should only be used to allow the last tool response to be scheduled

        Raises RuntimeError if the agent is draining
        """
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

    def _on_metrics_collected(self, ev: AgentMetrics) -> None:
        if speech_handle := _SpeechHandleContextVar.get(None):
            ev.speech_id = speech_handle.id
            self._session.emit("metrics_collected", MetricsCollectedEvent(metrics=ev))

    def _on_input_speech_started(self, _: llm.InputSpeechStartedEvent) -> None:
        log_event("input_speech_started")
        # self.interrupt() isn't going to raise when allow_interruptions is False, llm.InputSpeechStartedEvent is only fired by the server when the turn_detection is enabled.
        # When using the server-side turn_detection, we don't allow allow_interruptions to be False.
        try:
            self.interrupt()  # input_speech_started is also interrupting on the serverside realtime session
        except RuntimeError:
            logger.exception(
                "RealtimeAPI input_speech_started, but current speech is not interruptable, this should never happen!"
            )

    def _on_input_speech_stopped(self, ev: llm.InputSpeechStoppedEvent) -> None:
        log_event("input_speech_stopped")
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
        self._session.emit("user_started_speaking", UserStartedSpeakingEvent())

    def on_end_of_speech(self, ev: vad.VADEvent) -> None:
        self._session.emit("user_stopped_speaking", UserStoppedSpeakingEvent())

    def on_vad_inference_done(self, ev: vad.VADEvent) -> None:
        if ev.speech_duration > self._session.options.min_interruption_duration:
            if (
                self._current_speech is not None
                and not self._current_speech.interrupted
                and self._current_speech.allow_interruptions
            ):
                log_event(
                    "speech interrupted by vad",
                    speech_id=self._current_speech.id,
                )
                self._current_speech.interrupt()

    def on_interim_transcript(self, ev: stt.SpeechEvent) -> None:
        self._session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript=ev.alternatives[0].text, is_final=False),
        )

    def on_final_transcript(self, ev: stt.SpeechEvent) -> None:
        self._session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript=ev.alternatives[0].text, is_final=True),
        )

    async def on_end_of_turn(self, new_transcript: str) -> None:
        # When the audio recognition detects the end of a user turn:
        #  - check if there is no current generation happening
        #  - cancel the current generation if it allows interruptions (otherwise skip this current
        #  turn)
        #  - generate a reply to the user input

        if self._current_speech is not None:
            if not self._current_speech.allow_interruptions:
                logger.warning(
                    "skipping user input, current speech generation cannot be interrupted",
                    extra={"user_input": new_transcript},
                )
                return

            log_event(
                "speech interrupted, new user turn detected",
                speech_id=self._current_speech.id,
            )
            self._current_speech.interrupt()

        if self.draining:
            logger.warning(
                "skipping user input, task is draining",
                extra={"user_input": new_transcript},
            )
            log_event(
                "skipping user input, task is draining",
                user_input=new_transcript,
            )
            return

        await self._agent.on_end_of_turn(
            self._agent.chat_ctx,
            llm.ChatMessage(
                role="user", content=[new_transcript]
            ),  # TODO(theomonnom): This doesn't allow edits yet
        )

        if self.draining:
            # TODO(theomonnom): should we "forward" this new turn to the next agent?
            logger.warning("ignoring new user turn, the agent is draining")
            return

        self.generate_reply(user_input=new_transcript)

    # AudioRecognition is calling this method to retrieve the chat context before running the TurnDetector model
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
        forward_text, text_out = perform_text_forwarding(
            text_output=tr_output,
            source=self._agent.transcription_node(text_source, model_settings),
        )
        tasks.append(forward_text)
        if audio_output is None:
            # update the agent state based on text if no audio output
            text_out.first_text_fut.add_done_callback(
                lambda _: self._session._update_agent_state(AgentState.SPEAKING)
            )

        if audio_output is not None:
            if audio is None:
                # generate audio using TTS
                tts_task, tts_gen_data = perform_tts_inference(
                    node=self._agent.tts_node, input=audio_source, model_settings=model_settings
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

            audio_out.first_frame_fut.add_done_callback(
                lambda _: self._session._update_agent_state(AgentState.SPEAKING)
            )

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
            self._agent._chat_ctx.add_message(role="assistant", content=text_out.text)

    @utils.log_exceptions(logger=logger)
    async def _pipeline_reply_task(
        self,
        *,
        speech_handle: SpeechHandle,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings,
        user_input: str | None = None,
        instructions: str | None = None,
    ) -> None:
        from .agent import ModelSettings

        _SpeechHandleContextVar.set(speech_handle)

        log_event(
            "generation started", speech_id=speech_handle.id, step_index=speech_handle.step_index
        )

        audio_output = self._session.output.audio if self._session.output.audio_enabled else None
        text_output = (
            self._session.output.transcription
            if self._session.output.transcription_enabled
            else None
        )
        chat_ctx = chat_ctx.copy()
        tool_ctx = llm.ToolContext(tools)

        if user_input is not None:
            user_msg = chat_ctx.add_message(role="user", content=user_input)
            self._agent._chat_ctx.items.append(user_msg)

        if instructions is not None:
            try:
                update_instructions(chat_ctx, instructions=instructions, add_if_missing=True)
            except ValueError:
                logger.exception("failed to update the instructions")

        self._session._update_agent_state(AgentState.THINKING)
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
                node=self._agent.tts_node, input=tts_text_input, model_settings=model_settings
            )
            tasks.append(tts_task)

        await speech_handle.wait_if_not_interrupted(
            [asyncio.ensure_future(speech_handle._wait_for_authorization())]
        )

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*tasks)
            return

        forward_task, text_out = perform_text_forwarding(
            text_output=text_output,
            source=self._agent.transcription_node(llm_output, model_settings),
        )
        tasks.append(forward_task)

        if audio_output is not None:
            assert tts_gen_data is not None
            # TODO(theomonnom): should the audio be added to the chat_context too?
            forward_task, audio_out = perform_audio_forwarding(
                audio_output=audio_output, tts_output=tts_gen_data.audio_ch
            )
            tasks.append(forward_task)
            audio_out.first_frame_fut.add_done_callback(
                lambda _: self._session._update_agent_state(AgentState.SPEAKING)
            )
        else:
            text_out.first_text_fut.add_done_callback(
                lambda _: self._session._update_agent_state(AgentState.SPEAKING)
            )

        # start to execute tools (only after play())
        exe_task, fnc_outputs = perform_tool_executions(
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
                self._session._update_agent_state(AgentState.LISTENING)

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*tasks, exe_task)

            # if the audio playout was enabled, clear the buffer
            if audio_output is not None:
                audio_output.clear_buffer()
                playback_ev = await audio_output.wait_for_playout()

                log_event(
                    "playout interrupted",
                    playback_position=playback_ev.playback_position,
                    speech_id=speech_handle.id,
                )

                truncated_text = truncate_message(
                    message=text_out.text, played_duration=playback_ev.playback_position
                )
                msg = chat_ctx.add_message(
                    role="assistant", content=truncated_text, id=llm_gen_data.id
                )
                self._agent._chat_ctx.items.append(msg)
                self._session._update_agent_state(AgentState.LISTENING)

            return

        if text_out.text:
            msg = chat_ctx.add_message(role="assistant", content=text_out.text, id=llm_gen_data.id)
            self._agent._chat_ctx.items.append(msg)

        log_event("playout completed", speech_id=speech_handle.id)

        speech_handle._mark_playout_done()  # mark the playout done before waiting for the tool execution
        await exe_task

        # important: no agent ouput should be used after this point

        if len(fnc_outputs) > 0:
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
            new_agent_task: Agent | None = None
            ignore_task_switch = False
            for py_out in fnc_outputs:
                sanitized_out = py_out.sanitize()

                if sanitized_out.fnc_call_out is not None:
                    new_calls.append(sanitized_out.fnc_call)
                    new_fnc_outputs.append(sanitized_out.fnc_call_out)

                if new_agent_task is not None and sanitized_out.agent_task is not None:
                    logger.error(
                        "expected to receive only one AgentTask from the tool executions",
                    )
                    ignore_task_switch = True

                new_agent_task = sanitized_out.agent_task

            draining = self.draining
            if not ignore_task_switch and new_agent_task is not None:
                self._session.update_agent(new_agent_task)
                draining = True

            if len(new_fnc_outputs) > 0:
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
                    ),
                    owned_speech_handle=handle,
                    name="AgentActivity.pipeline_reply",
                )
                self._schedule_speech(
                    handle, SpeechHandle.SPEECH_PRIORITY_NORMAL, bypass_draining=True
                )

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

        if user_input is not None:
            chat_ctx = self._rt_session.chat_ctx.copy()
            chat_ctx.add_message(role="user", content=user_input)
            await self._rt_session.update_chat_ctx(chat_ctx)

        self._rt_session.update_options(tool_choice=model_settings.tool_choice)

        generation_ev = await self._rt_session.generate_reply(
            instructions=instructions or NOT_GIVEN
        )

        await self._realtime_generation_task(
            speech_handle=speech_handle,
            generation_ev=generation_ev,
            model_settings=model_settings,
        )

        # TODO(theomonnom): reset tool_choice value (not needed for now, because it isn't exposed to the user)
        # tool_choice is currently only set to None when draining

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
        tool_ctx = llm.ToolContext(self._agent.tools)

        await speech_handle.wait_if_not_interrupted(
            [asyncio.ensure_future(speech_handle._wait_for_authorization())]
        )

        if speech_handle.interrupted:
            return  # TODO(theomonnom): remove the message from the serverside history

        @utils.log_exceptions(logger=logger)
        async def _read_messages(
            outputs: list[tuple[_TextOutput | None, _AudioOutput | None]],
        ) -> None:
            forward_tasks: list[asyncio.Task] = []
            async for msg in generation_ev.message_stream:
                if len(forward_tasks) > 0:
                    logger.warning(
                        "expected to receive only one message generation from the realtime API"
                    )
                    break

                text_out = None
                audio_out = None

                if text_output is not None:
                    forward_task, text_out = perform_text_forwarding(
                        text_output=text_output,
                        source=self._agent.transcription_node(msg.text_stream, model_settings),
                    )

                    forward_tasks.append(forward_task)

                    if text_out is not None:
                        text_out.first_text_fut.add_done_callback(
                            lambda _: self._session._update_agent_state(AgentState.SPEAKING)
                        )

                if audio_output is not None:
                    forward_task, audio_out = perform_audio_forwarding(
                        audio_output=audio_output, tts_output=msg.audio_stream
                    )
                    forward_tasks.append(forward_task)
                    audio_out.first_frame_fut.add_done_callback(
                        lambda _: self._session._update_agent_state(AgentState.SPEAKING)
                    )

                outputs.append((text_out, audio_out))

            try:
                await asyncio.gather(*forward_tasks)
            finally:
                await utils.aio.cancel_and_wait(*forward_tasks)

        message_outputs: list[tuple[_TextOutput | None, _AudioOutput | None]] = []
        tasks = [
            self._create_speech_task(
                _read_messages(message_outputs),
                name="AgentActivity.realtime_generation.read_messages",
            )
        ]

        exe_task, fnc_outputs = perform_tool_executions(
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
            if not speech_handle.interrupted:
                self._session._update_agent_state(AgentState.LISTENING)

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*tasks, exe_task)

            if audio_output is not None:
                audio_output.clear_buffer()
                playback_ev = await audio_output.wait_for_playout()

                log_event(
                    "playout interrupted",
                    playback_position=playback_ev.playback_position,
                    speech_id=speech_handle.id,
                )
                self._session._update_agent_state(AgentState.LISTENING)

            # TODO(theomonnom): truncate message (+ OAI serverside mesage)
            return

        speech_handle._mark_playout_done()  # mark the playout done before waiting for the tool execution
        await exe_task

        # important: no agent ouput should be used after this point

        if len(fnc_outputs) > 0:
            new_fnc_outputs: list[llm.FunctionCallOutput] = []
            new_agent_task: Agent | None = None
            ignore_task_switch = False

            for py_out in fnc_outputs:
                sanitized_out = py_out.sanitize()

                if sanitized_out.fnc_call_out is not None:
                    new_fnc_outputs.append(sanitized_out.fnc_call_out)

                if new_agent_task is not None and sanitized_out.agent_task is not None:
                    logger.error(
                        "expected to receive only one AgentTask from the tool executions",
                    )
                    ignore_task_switch = True

                new_agent_task = sanitized_out.agent_task

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
                        "failed to update chat context before generating the function calls results",
                        extra={"error": str(e)},
                    )

                # should we have separate flag for this?
                if self.llm.capabilities.message_truncation:
                    self._rt_session.interrupt()

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
