from __future__ import annotations

import asyncio
import heapq
import time
from typing import (
    TYPE_CHECKING,
    AsyncIterable,
)

from livekit import rtc

from .. import debug, llm, stt, tts, utils, vad
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils.misc import is_given
from .audio_recognition import AudioRecognition, RecognitionHooks, _TurnDetector
from .events import AgentContext
from .generation import (
    _AudioOutput,
    _TextOutput,
    _TTSGenerationData,
    perform_audio_forwarding,
    perform_llm_inference,
    perform_text_forwarding,
    perform_tool_executions,
    perform_tts_inference,
    update_instructions,
)
from .speech_handle import SpeechHandle


def log_event(event: str, **kwargs) -> None:
    debug.Tracing.log_event(event, kwargs)


if TYPE_CHECKING:
    from .pipeline_agent import PipelineAgent
    from .task import AgentTask


class TaskActivity(RecognitionHooks):
    def __init__(self, task: AgentTask, agent: PipelineAgent) -> None:
        self._agent_task, self._agent = task, agent
        self._rt_session: llm.RealtimeSession | None = None
        self._audio_recognition: AudioRecognition | None = None
        self._lock = asyncio.Lock()

        self._draining = False

        self._current_speech: SpeechHandle | None = None
        self._speech_q: list[tuple[int, float, SpeechHandle]] = []
        self._speech_q_changed = asyncio.Event()

        self._main_atask: asyncio.Task | None = None
        self._tasks: list[asyncio.Task] = []
        self._started = False

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def agent(self) -> PipelineAgent:
        return self._agent

    @property
    def turn_detector(self) -> _TurnDetector | None:
        return self._agent_task._eou or self._agent._turn_detector

    @property
    def stt(self) -> stt.STT | None:
        return self._agent_task.stt or self._agent.stt

    @property
    def llm(self) -> llm.LLM | llm.RealtimeModel | None:
        return self._agent_task.llm or self._agent.llm

    @property
    def tts(self) -> tts.TTS | None:
        return self._agent_task.tts or self._agent.tts

    @property
    def vad(self) -> vad.VAD | None:
        return self._agent_task.vad or self._agent.vad

    @property
    def current_speech(self) -> SpeechHandle | None:
        return self._current_speech

    async def drain(self) -> None:
        async with self._lock:
            if self._draining:
                return

            await self._agent_task.on_exit()

            self._speech_q_changed.set()  # TODO(theomonnom): we shouldn't need this here
            self._draining = True
            if self._main_atask is not None:
                await asyncio.shield(self._main_atask)

    async def start(self) -> None:
        async with self._lock:
            self._agent_task._activity = self
            self._main_atask = asyncio.create_task(self._main_task(), name="_main_task")
            self._audio_recognition = AudioRecognition(
                hooks=self,
                stt=self._agent_task.stt_node,
                vad=self.vad,
                turn_detector=self.turn_detector,
                min_endpointing_delay=self._agent.options.min_endpointing_delay,
            )
            self._audio_recognition.start()

            if isinstance(self.llm, llm.RealtimeModel):
                self._rt_session = self.llm.session()
                self._rt_session.on("generation_created", self._on_generation_created)
                self._rt_session.on(
                    "input_speech_started", self._on_input_speech_started
                )
                self._rt_session.on(
                    "input_speech_stopped", self._on_input_speech_stopped
                )
                self._rt_session.on(
                    "input_audio_transcription_completed",
                    self._on_input_audio_transcription_completed,
                )
                self._rt_session.on(
                    "input_audio_transcription_failed",
                    self._on_input_audio_transcription_failed,
                )
                try:
                    await self._rt_session.update_instructions(
                        self._agent_task.instructions
                    )
                except llm.RealtimeError:
                    logger.exception("failed to update the instructions")

                try:
                    await self._rt_session.update_chat_ctx(self._agent_task.chat_ctx)
                except llm.RealtimeError:
                    logger.exception("failed to update the chat_ctx")

                try:
                    await self._rt_session.update_fnc_ctx(self._agent_task.ai_functions)
                except llm.RealtimeError:
                    logger.exception("failed to update the fnc_ctx")

            elif isinstance(self.llm, llm.LLM):
                try:
                    update_instructions(
                        self._agent_task._chat_ctx,
                        instructions=self._agent_task.instructions,
                        add_if_missing=True,
                    )
                except ValueError:
                    logger.exception("failed to update the instructions")

            self._started = True
            await self._agent_task.on_enter()

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

            self._agent_task._activity = None

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if not self._started:
            return

        if self._rt_session is not None:
            self._rt_session.push_audio(frame)

        if self._audio_recognition is not None:
            self._audio_recognition.push_audio(frame)

    def say(
        self,
        source: str | AsyncIterable[rtc.AudioFrame],
        *,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> SpeechHandle:
        if isinstance(source, str) and self.tts is None:
            raise ValueError("trying to generate speech from text without a TTS model")

        handle = SpeechHandle.create(
            allow_interruptions=allow_interruptions
            if is_given(allow_interruptions)
            else self._agent.options.allow_interruptions
        )

        task = asyncio.create_task(
            self._tts_task(
                speech_handle=handle,
                source=source,
            ),
            name="_tts_task",
        )
        self._tasks.append(task)
        task.add_done_callback(lambda _: handle._mark_playout_done())
        task.add_done_callback(lambda _: self._tasks.remove(task))
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
            raise ValueError("another reply is already in progress")

        log_event(
            "generate_reply",
            user_input=user_input or None,
            instructions=instructions or None,
        )

        handle = SpeechHandle.create(
            allow_interruptions=allow_interruptions
            if is_given(allow_interruptions)
            else self._agent.options.allow_interruptions
        )

        if isinstance(self.llm, llm.RealtimeModel):
            task = asyncio.create_task(
                self._realtime_reply_task(
                    speech_handle=handle,
                    user_input=user_input or None,
                    instructions=instructions or None,
                ),
                name="_realtime_reply_task",
            )
            self._tasks.append(task)
            task.add_done_callback(lambda _: handle._mark_playout_done())
            task.add_done_callback(lambda _: self._tasks.remove(task))

        elif isinstance(self.llm, llm.LLM):
            task = asyncio.create_task(
                self._pipeline_reply_task(
                    speech_handle=handle,
                    chat_ctx=self._agent_task._chat_ctx,
                    fnc_ctx=self._agent_task._fnc_ctx,
                    user_input=user_input or None,
                    instructions=instructions or None,
                ),
                name="_pipeline_reply_task",
            )
            self._tasks.append(task)
            task.add_done_callback(lambda _: handle._mark_playout_done())
            task.add_done_callback(lambda _: self._tasks.remove(task))

        self._schedule_speech(handle, SpeechHandle.SPEECH_PRIORITY_NORMAL)
        return handle

    def interrupt(self) -> None:
        if self._current_speech is not None:
            self._current_speech.interrupt()

        for speech in self._speech_q:
            _, _, speech = speech
            speech.interrupt()

    def _schedule_speech(self, speech: SpeechHandle, priority: int) -> None:
        if self._draining:
            raise RuntimeError("cannot schedule new speech, task is draining")

        heapq.heappush(self._speech_q, (priority, time.time(), speech))
        self._speech_q_changed.set()

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        try:
            while True:
                await self._speech_q_changed.wait()
                while self._speech_q:
                    _, _, speech = heapq.heappop(self._speech_q)
                    self._current_speech = speech
                    speech._authorize_playout()
                    await speech.wait_for_playout()
                    self._current_speech = None

                if self._draining:  # no more speech can be scheduled
                    break

                self._speech_q_changed.clear()
        finally:
            log_event(f"task done, waiting for {len(self._tasks)} tasks")
            await asyncio.gather(*self._tasks)
            log_event("marking agent task as done")

    # -- Realtime Session events --

    def _on_input_speech_started(self, _: llm.InputSpeechStartedEvent) -> None:
        log_event("input_speech_started")
        self.interrupt()  # input_speech_started is also interrupting on the serverside realtime session

    def _on_input_speech_stopped(self, _: llm.InputSpeechStoppedEvent) -> None:
        log_event("input_speech_stopped")
        self.on_interim_transcript(
            stt.SpeechEvent(
                stt.SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[stt.SpeechData(text="", language="")],
            )
        )

    def _on_input_audio_transcription_completed(
        self, ev: llm.InputTranscriptionCompleted
    ) -> None:
        log_event("input_audio_transcription_completed")
        self.on_final_transcript(
            stt.SpeechEvent(
                stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(text=ev.transcript, language="")],
            )
        )

    def _on_input_audio_transcription_failed(
        self, ev: llm.InputTranscriptionFailed
    ) -> None:
        log_event("input_audio_transcription_failed")
        self.on_final_transcript(
            stt.SpeechEvent(
                stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(text="", language="")],
            )
        )

    def _on_generation_created(self, ev: llm.GenerationCreatedEvent) -> None:
        if self.draining:
            logger.warning("skipping new generation, task is draining")
            log_event("skipping new generation, task is draining")
            return

        if ev.user_initiated:
            # user_initiated generations are directly handled inside _realtime_reply_task
            return

        handle = SpeechHandle.create(
            allow_interruptions=self._agent.options.allow_interruptions
        )
        task = asyncio.create_task(
            self._realtime_generation_task(
                speech_handle=handle,
                generation_ev=ev,
            ),
        )
        self._tasks.append(task)
        task.add_done_callback(lambda _: handle._mark_playout_done())
        task.add_done_callback(lambda _: self._tasks.remove(task))
        self._schedule_speech(handle, SpeechHandle.SPEECH_PRIORITY_NORMAL)

    # region recognition hooks

    def on_start_of_speech(self, ev: vad.VADEvent) -> None:
        pass
        # self.emit("user_started_speaking", events.UserStartedSpeakingEvent())

    def on_end_of_speech(self, ev: vad.VADEvent) -> None:
        pass
        # self.emit("user_stopped_speaking", events.UserStoppedSpeakingEvent())

    def on_vad_inference_done(self, ev: vad.VADEvent) -> None:
        if ev.speech_duration > self._agent.options.min_interruption_duration:
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
        self._agent._on_user_transcript(ev)

    def on_final_transcript(self, ev: stt.SpeechEvent) -> None:
        self._agent._on_user_transcript(ev)

    def on_end_of_turn(self, new_transcript: str) -> None:
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

        self.generate_reply(user_input=new_transcript)

    def retrieve_chat_ctx(self) -> llm.ChatContext:
        return self._agent_task.chat_ctx

    # endregion

    @utils.log_exceptions(logger=logger)
    async def _tts_task(
        self, speech_handle: SpeechHandle, source: str | AsyncIterable[rtc.AudioFrame]
    ) -> None:
        text_output = self._agent.output.text
        audio_output = self._agent.output.audio

        await speech_handle.wait_if_not_interrupted(
            [asyncio.ensure_future(speech_handle._wait_for_authorization())]
        )

        if speech_handle.interrupted:
            return

        tasks = []
        if isinstance(source, str):
            if text_output is not None:
                await text_output.capture_text(source)
                text_output.flush()

            async def _read_text() -> AsyncIterable[str]:
                yield source

            tts_task: asyncio.Task | None = None
            tts_gen_data: _TTSGenerationData | None = None
            if audio_output is not None:
                tts_task, tts_gen_data = perform_tts_inference(
                    node=self._agent_task.tts_node, input=_read_text()
                )
                tasks.append(tts_task)

                forward_task, _ = perform_audio_forwarding(
                    audio_output=audio_output, tts_output=tts_gen_data.audio_ch
                )
                tasks.append(forward_task)
        elif isinstance(source, AsyncIterable):
            if audio_output is not None:
                forward_task, _ = perform_audio_forwarding(
                    audio_output=audio_output, tts_output=source
                )
                tasks.append(forward_task)

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

    @utils.log_exceptions(logger=logger)
    async def _pipeline_reply_task(
        self,
        *,
        speech_handle: SpeechHandle,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext,
        user_input: str | None = None,
        instructions: str | None = None,
    ) -> None:
        log_event(
            "generation started",
            speech_id=speech_handle.id,
            step_index=speech_handle.step_index,
        )

        audio_output = self._agent.output.audio
        text_output = self._agent.output.text
        chat_ctx = chat_ctx.copy()
        fnc_ctx = fnc_ctx.copy()

        if user_input is not None:
            chat_ctx.add_message(role="user", content=user_input)

        if instructions is not None:
            try:
                update_instructions(
                    chat_ctx, instructions=instructions, add_if_missing=True
                )
            except ValueError:
                logger.exception("failed to update the instructions")

        tasks = []
        llm_task, llm_gen_data = perform_llm_inference(
            node=self._agent_task.llm_node,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
        )
        tasks.append(llm_task)
        tts_text_input, llm_output = utils.aio.itertools.tee(llm_gen_data.text_ch)

        tts_task: asyncio.Task | None = None
        tts_gen_data: _TTSGenerationData | None = None
        if audio_output is not None:
            tts_task, tts_gen_data = perform_tts_inference(
                node=self._agent_task.tts_node, input=tts_text_input
            )
            tasks.append(tts_task)

        await speech_handle.wait_if_not_interrupted(
            [asyncio.ensure_future(speech_handle._wait_for_authorization())]
        )

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*tasks)
            return

        forward_task, text_out = perform_text_forwarding(
            text_output=text_output, llm_output=llm_output
        )
        tasks.append(forward_task)

        if audio_output is not None:
            assert tts_gen_data is not None
            # TODO(theomonnom): should the audio be added to the chat_context too?
            forward_task, _ = perform_audio_forwarding(
                audio_output=audio_output, tts_output=tts_gen_data.audio_ch
            )
            tasks.append(forward_task)

        # start to execute tools (only after play())
        exe_task, fnc_outputs = perform_tool_executions(
            agent_ctx=AgentContext(self._agent),
            fnc_ctx=fnc_ctx,
            speech_handle=speech_handle,
            function_stream=llm_gen_data.function_ch,
        )
        tasks.append(exe_task)

        await speech_handle.wait_if_not_interrupted([*tasks])

        # wait for the end of the playout if the audio is enabled
        if audio_output is not None:
            await speech_handle.wait_if_not_interrupted(
                [asyncio.ensure_future(audio_output.wait_for_playout())]
            )

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*tasks)

            # if the audio playout was enabled, clear the buffer
            if audio_output is not None:
                audio_output.clear_buffer()
                playback_ev = await audio_output.wait_for_playout()

                log_event(
                    "playout interrupted",
                    playback_position=playback_ev.playback_position,
                    speech_id=speech_handle.id,
                )

                # TODO(theomonnom): calculate the played text based on playback_ev.playback_position
                msg = chat_ctx.add_message(role="assistant", content=text_out.text)
                self._agent_task._chat_ctx.items.append(msg)

            return

        if text_out.text:
            msg = chat_ctx.add_message(role="assistant", content=text_out.text)
            self._agent_task._chat_ctx.items.append(msg)

        log_event("playout completed", speech_id=speech_handle.id)

        if len(fnc_outputs) > 0:
            if speech_handle.step_index >= self._agent.options.max_fnc_steps:
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
            new_agent_task: AgentTask | None = None
            ignore_task_switch = False
            for fnc_call, fnc_output, agent_task in fnc_outputs:
                if fnc_output is not None:
                    new_calls.append(fnc_call)
                    new_fnc_outputs.append(fnc_output)

                if new_agent_task is not None and agent_task is not None:
                    logger.error(
                        "expected to receive only one new task from the tool executions",
                    )
                    ignore_task_switch = True

                new_agent_task = agent_task

            if len(new_fnc_outputs) > 0:
                chat_ctx.items.extend(new_calls)
                chat_ctx.items.extend(new_fnc_outputs)

                handle = SpeechHandle.create(
                    allow_interruptions=self._agent.options.allow_interruptions,
                    step_index=speech_handle.step_index + 1,
                )
                task = asyncio.create_task(
                    self._pipeline_reply_task(
                        speech_handle=handle,
                        chat_ctx=chat_ctx,
                        fnc_ctx=fnc_ctx,
                    ),
                    name="_pipeline_reply_task",
                )
                self._tasks.append(task)
                task.add_done_callback(lambda _: handle._mark_playout_done())
                task.add_done_callback(lambda _: self._tasks.remove(task))
                self._schedule_speech(handle, SpeechHandle.SPEECH_PRIORITY_NORMAL)

            if not ignore_task_switch and new_agent_task is not None:
                self._agent.update_task(new_agent_task)

        log_event("playout completed", speech_id=speech_handle.id)

    @utils.log_exceptions(logger=logger)
    async def _realtime_reply_task(
        self,
        *,
        speech_handle: SpeechHandle,
        user_input: str | None,
        instructions: str | None,
    ) -> None:
        assert self._rt_session is not None, "rt_session is not available"

        if user_input is not None:
            chat_ctx = self._rt_session.chat_ctx.copy()
            chat_ctx.add_message(role="user", content=user_input)
            await self._rt_session.update_chat_ctx(chat_ctx)

        generation_ev = await self._rt_session.generate_reply(
            instructions=instructions or NOT_GIVEN
        )

        await self._realtime_generation_task(
            speech_handle=speech_handle,
            generation_ev=generation_ev,
        )

    @utils.log_exceptions(logger=logger)
    async def _realtime_generation_task(
        self,
        *,
        speech_handle: SpeechHandle,
        generation_ev: llm.GenerationCreatedEvent,
    ) -> None:
        assert self._rt_session is not None, "rt_session is not available"

        log_event(
            "generation started",
            speech_id=speech_handle.id,
            step_index=speech_handle.step_index,
            realtime=True,
        )

        audio_output = self._agent.output.audio
        text_output = self._agent.output.text

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
                        text_output=text_output, llm_output=msg.text_stream
                    )
                    forward_tasks.append(forward_task)

                if audio_output is not None:
                    forward_task, audio_out = perform_audio_forwarding(
                        audio_output=audio_output, tts_output=msg.audio_stream
                    )
                    forward_tasks.append(forward_task)

                outputs.append((text_out, audio_out))

            try:
                await asyncio.gather(*forward_tasks)
            finally:
                await utils.aio.cancel_and_wait(*forward_tasks)

        message_outputs: list[tuple[_TextOutput | None, _AudioOutput | None]] = []
        tasks = [
            asyncio.create_task(
                _read_messages(message_outputs),
                name="_realtime_generation_task.read_messages",
            )
        ]

        exe_task, fnc_outputs = perform_tool_executions(
            agent_ctx=AgentContext(self._agent),
            fnc_ctx=self._agent_task._fnc_ctx,
            speech_handle=speech_handle,
            function_stream=generation_ev.function_stream,
        )
        tasks.append(exe_task)

        await speech_handle.wait_if_not_interrupted([*tasks])

        if audio_output is not None:
            await speech_handle.wait_if_not_interrupted(
                [asyncio.ensure_future(audio_output.wait_for_playout())]
            )

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*tasks)

            if audio_output is not None:
                audio_output.clear_buffer()
                playback_ev = await audio_output.wait_for_playout()

                log_event(
                    "playout interrupted",
                    playback_position=playback_ev.playback_position,
                    speech_id=speech_handle.id,
                )

            # TODO(theomonnom): truncate message (+ OAI serverside mesage)
            return

        if len(fnc_outputs) > 0:
            new_fnc_outputs: list[llm.FunctionCallOutput] = []
            new_agent_task: AgentTask | None = None
            ignore_task_switch = False
            for _, fnc_output, agent_task in fnc_outputs:
                if fnc_output is not None:
                    new_fnc_outputs.append(fnc_output)

                if new_agent_task is not None and agent_task is not None:
                    logger.error(
                        "expected to receive only one new task from the tool executions",
                    )
                    ignore_task_switch = True

                new_agent_task = agent_task

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

                self._rt_session.interrupt()
                try:
                    await self._rt_session.generate_reply()
                except llm.RealtimeError as e:
                    logger.warning(
                        "failed to generate the function calls results",
                        extra={"error": str(e)},
                    )

            if not ignore_task_switch and new_agent_task is not None:
                self._agent.update_task(new_agent_task)

        log_event("playout completed", speech_id=speech_handle.id)
