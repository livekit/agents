from __future__ import annotations

import asyncio
import contextlib
import heapq
import time
from typing import (
    TYPE_CHECKING,
    AsyncIterable,
    Optional,
    Union,
)

from livekit import rtc

from .. import debug, llm, multimodal, stt, tokenize, tts, utils, vad
from .._exceptions import APITimeoutError
from ..llm import (
    ChatContext,
    FunctionContext,
    find_ai_functions,
)
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from . import tools
from .audio_recognition import AudioRecognition, RecognitionHooks, _TurnDetector
from .context import AgentContext
from .generation import (
    _AudioOutput,
    _TextOutput,
    _TTSGenerationData,
    perform_audio_forwarding,
    perform_text_forwarding,
)
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from .pipeline_agent import PipelineAgent


class AgentTask:
    def __init__(
        self,
        *,
        instructions: str,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        fnc_ctx: NotGivenOr[llm.FunctionContext] = NOT_GIVEN,
        turn_detector: NotGivenOr[_TurnDetector] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | multimodal.RealtimeModel] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS] = NOT_GIVEN,
    ) -> None:
        if tts and not tts.capabilities.streaming:
            from .. import tts as text_to_speech

            tts = text_to_speech.StreamAdapter(
                tts=tts, sentence_tokenizer=tokenize.basic.SentenceTokenizer()
            )

        if stt and not stt.capabilities.streaming:
            from .. import stt as speech_to_text

            if not is_given(vad):
                raise ValueError(
                    "VAD is required when streaming is not supported by the STT"
                )

            stt = speech_to_text.StreamAdapter(
                stt=stt,
                vad=vad,
            )

        self._instructions = instructions
        self._chat_ctx = chat_ctx or ChatContext.empty()
        self._fnc_ctx = fnc_ctx or FunctionContext.empty()
        self._fnc_ctx.update_ai_functions(
            list(self._fnc_ctx.ai_functions.values()) + find_ai_functions(self)
        )
        self._turn_detector = turn_detector or None
        self._stt = stt or None
        self._llm = llm or None
        self._tts = tts or None
        self._vad = vad or None

    @property
    def instructions(self) -> str:
        return self._instructions

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def fnc_ctx(self) -> llm.FunctionContext:
        return self._fnc_ctx

    @property
    def turn_detector(self) -> _TurnDetector | None:
        return self._turn_detector

    @property
    def stt(self) -> stt.STT | None:
        return self._stt

    @property
    def llm(self) -> llm.LLM | multimodal.RealtimeModel | None:
        return self._llm

    @property
    def tts(self) -> tts.TTS | None:
        return self._tts

    @property
    def vad(self) -> vad.VAD | None:
        return self._vad

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame]
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        assert self._stt is not None, "stt_node called but no STT node is available"

        async with self._stt.stream() as stream:

            async def _forward_input():
                async for frame in audio:
                    stream.push_frame(frame)

            forward_task = asyncio.create_task(_forward_input())
            try:
                async for event in stream:
                    yield event
            finally:
                await utils.aio.cancel_and_wait(forward_task)

    async def llm_node(
        self, chat_ctx: llm.ChatContext, fnc_ctx: llm.FunctionContext | None
    ) -> Union[
        Optional[AsyncIterable[llm.ChatChunk]],
        Optional[AsyncIterable[str]],
        Optional[str],
    ]:
        assert self._llm is not None, "llm_node called but no LLM node is available"
        assert isinstance(self._llm, llm.LLM), (
            "llm_node should only be used with LLM (non-multimodal/realtime APIs) nodes"
        )

        async with self._llm.chat(chat_ctx=chat_ctx, fnc_ctx=fnc_ctx) as stream:
            async for chunk in stream:
                yield chunk

    async def tts_node(
        self, text: AsyncIterable[str]
    ) -> Optional[AsyncIterable[rtc.AudioFrame]]:
        assert self._tts is not None, "tts_node called but no TTS node is available"

        async with self._tts.stream() as stream:

            async def _forward_input():
                async for chunk in text:
                    stream.push_text(chunk)

                stream.end_input()

            forward_task = asyncio.create_task(_forward_input())
            try:
                async for ev in stream:
                    yield ev.frame
            finally:
                await utils.aio.cancel_and_wait(forward_task)

    def _create_activity(self, agent: PipelineAgent) -> TaskActivity:
        return TaskActivity(task=self, agent=agent)


class TaskActivity(RecognitionHooks):
    def __init__(self, task: AgentTask, agent: PipelineAgent) -> None:
        self._task, self._agent = task, agent
        self._rt_session: multimodal.RealtimeSession | None = None
        self._audio_recognition: AudioRecognition | None = None
        self._lock = asyncio.Lock()

        self._done_fut = asyncio.Future()
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

    async def drain(self) -> None:
        self._speech_q_changed.set()  # TODO(theomonnom): we shouldn't need this here
        self._draining = True

        if self._main_atask is not None:
            await asyncio.shield(self._main_atask)

    async def start(self) -> None:
        async with self._lock:
            self._main_atask = asyncio.create_task(self._main_task(), name="_main_task")

            self._audio_recognition = AudioRecognition(
                hooks=self,
                stt=self._task.stt_node,
                vad=self._task.vad,
                turn_detector=self._task.turn_detector,
                min_endpointing_delay=self._agent.options.min_endpointing_delay,
            )
            self._audio_recognition.start()

            if isinstance(self._task.llm, multimodal.RealtimeModel):
                self._rt_session = self._task.llm.session()
                self._rt_session.on("generation_created", self._on_generation_created)
                self._rt_session.on(
                    "input_speech_started", self._on_input_speech_started
                )
                self._rt_session.on(
                    "input_speech_stopped", self._on_input_speech_stopped
                )
                await self._rt_session.update_instructions(self._task.instructions)
                await self._rt_session.update_chat_ctx(self._task.chat_ctx)
                await self._rt_session.update_fnc_ctx(self._task.fnc_ctx)

            self._started = True

    async def aclose(self) -> None:
        async with self._lock:
            if self._rt_session is not None:
                await self._rt_session.aclose()

            if self._audio_recognition is not None:
                await self._audio_recognition.aclose()

            if self._main_atask is not None:
                await utils.aio.cancel_and_wait(self._main_atask)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if not self._started:
            return

        if self._rt_session is not None:
            self._rt_session.push_audio(frame)

        if self._audio_recognition is not None:
            self._audio_recognition.push_audio(frame)

    def generate_reply(self, user_input: str) -> SpeechHandle:
        if (
            self._agent.current_speech is not None
            and not self._agent.current_speech.interrupted
        ):
            raise ValueError("another reply is already in progress")

        debug.Tracing.log_event("generate_reply", {"user_input": user_input})

        # TODO(theomonnom): move to _generate_pipeline_reply_task
        # self._chat_ctx.items.append(
        #     llm.ChatItem.create(
        #         [llm.ChatMessage.create(role="user", content=user_input)]
        #     )
        # )

        handle = SpeechHandle.create(
            allow_interruptions=self._agent.options.allow_interruptions
        )
        task = asyncio.create_task(
            self._pipeline_reply_task(
                handle=handle,
                chat_ctx=self._task.chat_ctx,
                fnc_ctx=self._task.fnc_ctx,
            ),
            name="_pipeline_reply_task",
        )
        self._tasks.append(task)
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
            await asyncio.gather(*self._tasks)
            debug.Tracing.log_event(f"task done, waiting for {len(self._tasks)} tasks")
            debug.Tracing.log_event("marking agent task as done")
            self._done_fut.set_result(None)

    # -- Realtime Session events --

    def _on_input_speech_started(self, _: multimodal.InputSpeechStartedEvent) -> None:
        debug.Tracing.log_event("input_speech_started")
        self.interrupt()

    def _on_input_speech_stopped(self, _: multimodal.InputSpeechStoppedEvent) -> None:
        debug.Tracing.log_event("input_speech_stopped")

    def _on_generation_created(self, ev: multimodal.GenerationCreatedEvent) -> None:
        if self.draining:
            logger.warning("skipping new generation, task is draining")
            debug.Tracing.log_event("skipping new generation, task is draining")
            return

        handle = SpeechHandle.create(
            allow_interruptions=self._agent.options.allow_interruptions
        )
        task = asyncio.create_task(
            self._realtime_reply_task(
                speech_handle=handle,
                generation_ev=ev,
            ),
        )
        self._tasks.append(task)
        task.add_done_callback(lambda _: self._tasks.remove(task))
        self._schedule_speech(handle, SpeechHandle.SPEECH_PRIORITY_NORMAL)

    # -- Recognition Hooks --

    def on_start_of_speech(self, ev: vad.VADEvent) -> None:
        pass
        # self.emit("user_started_speaking", events.UserStartedSpeakingEvent())

    def on_end_of_speech(self, ev: vad.VADEvent) -> None:
        pass
        # self.emit("user_stopped_speaking", events.UserStoppedSpeakingEvent())

    def on_vad_inference_done(self, ev: vad.VADEvent) -> None:
        if ev.speech_duration > self._agent.options.min_interruption_duration:
            if (
                self._agent.current_speech is not None
                and not self._agent.current_speech.interrupted
                and self._agent.current_speech.allow_interruptions
            ):
                debug.Tracing.log_event(
                    "speech interrupted by vad",
                    {"speech_id": self._agent.current_speech.id},
                )
                self._agent.current_speech.interrupt()

    def on_interim_transcript(self, ev: stt.SpeechEvent) -> None:
        pass

    def on_final_transcript(self, ev: stt.SpeechEvent) -> None:
        pass

    def on_end_of_turn(self, new_transcript: str) -> None:
        # When the audio recognition detects the end of a user turn:
        #  - check if there is no current generation happening
        #  - cancel the current generation if it allows interruptions (otherwise skip this current
        #  turn)
        #  - generate a reply to the user input

        if self._agent.current_speech is not None:
            if self._agent.current_speech.allow_interruptions:
                logger.warning(
                    "skipping user input, current speech generation cannot be interrupted",
                    extra={"user_input": new_transcript},
                )
                return

            debug.Tracing.log_event(
                "speech interrupted, new user turn detected",
                {"speech_id": self._agent.current_speech.id},
            )
            self._agent.current_speech.interrupt()

        if self.draining:
            logger.warning(
                "skipping user input, task is draining",
                extra={"user_input": new_transcript},
            )
            debug.Tracing.log_event(
                "skipping user input, task is draining",
                {"user_input": new_transcript},
            )
            return

        self.generate_reply(new_transcript)

    def retrieve_chat_ctx(self) -> llm.ChatContext:
        return self._task.chat_ctx

    # ---

    @utils.log_exceptions(logger=logger)
    async def _pipeline_reply_task(
        self,
        *,
        speech_handle: SpeechHandle,
    ) -> None:
        debug.Tracing.log_event(
            "generation started",
            {"speech_id": speech_handle.id, "step_index": speech_handle.step_index},
        )

        wg = utils.aio.WaitGroup()
        tasks = []
        llm_task, llm_gen_data = do_llm_inference(
            node=self.llm_node,
            chat_ctx=self._chat_ctx,
            fnc_ctx=(
                self._fnc_ctx
                if speech_handle.step_index < self._agent.options.max_fnc_steps - 1
                and speech_handle.step_index >= 2
                else None
            ),
        )
        tasks.append(llm_task)
        wg.add(1)
        llm_task.add_done_callback(lambda _: wg.done())
        tts_text_input, llm_output = utils.aio.itertools.tee(llm_gen_data.text_ch)

        tts_task: asyncio.Task | None = None
        tts_gen_data: _TTSGenerationData | None = None
        if self._agent.output.audio is not None:
            tts_task, tts_gen_data = do_tts_inference(
                node=self.tts_node, input=tts_text_input
            )
            tasks.append(tts_task)
            wg.add(1)
            tts_task.add_done_callback(lambda _: wg.done())

        # wait for the play() method to be called
        await asyncio.wait(
            [
                speech_handle._play_fut,
                speech_handle._interrupt_fut,
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*tasks)
            speech_handle._mark_done()
            return  # return directly (the generated output wasn't used)

        # forward tasks are started after the play() method is called
        # they redirect the generated text/audio to the output channels
        forward_llm_task = asyncio.create_task(
            _forward_llm_text(llm_output),
            name="_generate_reply_task.forward_llm_text",
        )
        tasks.append(forward_llm_task)
        wg.add(1)
        forward_llm_task.add_done_callback(lambda _: wg.done())

        forward_tts_task: asyncio.Task | None = None
        if tts_gen_data is not None:
            forward_tts_task = asyncio.create_task(
                _forward_tts_audio(tts_gen_data.audio_ch),
                name="_generate_reply_task.forward_tts_audio",
            )
            tasks.append(forward_tts_task)
            wg.add(1)
            forward_tts_task.add_done_callback(lambda _: wg.done())

        # start to execute tools (only after play())
        called_functions: set[llm.CalledFunction] = set()
        tools_task = asyncio.create_task(
            _execute_tools(llm_gen_data.tools_ch, called_functions),
            name="_generate_reply_task.execute_tools",
        )
        tasks.append(tools_task)
        wg.add(1)
        tools_task.add_done_callback(lambda _: wg.done())

        # wait for the tasks to finish
        await asyncio.wait(
            [
                wg.wait(),
                speech_handle._interrupt_fut,
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # wait for the end of the playout if the audio is enabled
        if forward_tts_task is not None and self._agent.output.audio is not None:
            await asyncio.wait(
                [
                    self._agent.output.audio.wait_for_playout(),
                    speech_handle._interrupt_fut,
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*tasks)

            if len(called_functions) > 0:
                functions = [
                    cfnc.call_info.function_info.name for cfnc in called_functions
                ]
                logger.debug(
                    "speech interrupted, ignoring generation of the function calls results",
                    extra={"speech_id": speech_handle.id, "functions": functions},
                )
                debug.Tracing.log_event(
                    "speech interrupted, ignoring generation of the function calls results",
                    {"speech_id": speech_handle.id, "functions": functions},
                )

            # if the audio playout was enabled, clear the buffer
            if forward_tts_task is not None and self._agent.output.audio is not None:
                self._agent.output.audio.clear_buffer()
                playback_ev = await self._agent.output.audio.wait_for_playout()

                debug.Tracing.log_event(
                    "playout interrupted",
                    {
                        "playback_position": playback_ev.playback_position,
                        "speech_id": speech_handle.id,
                    },
                )

                # TODO(theomonnom): calculate the played text based on playback_ev.playback_position

            speech_handle._mark_playout_done()
            speech_handle._mark_done()
            return

        speech_handle._mark_playout_done()
        debug.Tracing.log_event("playout completed", {"speech_id": speech_handle.id})

        if len(called_functions) > 0:
            if speech_handle.step_index >= self._agent.options.max_fnc_steps:
                logger.warning(
                    "maximum number of function calls steps reached",
                    extra={"speech_id": speech_handle.id},
                )
                debug.Tracing.log_event(
                    "maximum number of function calls steps reached",
                    {"speech_id": speech_handle.id},
                )
                speech_handle._mark_done()
                return

            # create a new SpeechHandle to generate the result of the function calls
            speech_handle = SpeechHandle.create(
                allow_interruptions=speech_handle.allow_interruptions,
                step_index=speech_handle.step_index + 1,
            )
            task = asyncio.create_task(
                self._pipeline_reply_task(
                    handle=speech_handle,
                    chat_ctx=self._chat_ctx,
                    fnc_ctx=self._fnc_ctx,
                ),
                name="_pipeline_fnc_reply_task",
            )
            self._agent._schedule_speech(
                speech_handle, SpeechHandle.SPEECH_PRIORITY_NORMAL
            )

        speech_handle._mark_done()

    @utils.log_exceptions(logger=logger)
    async def _realtime_reply_task(
        self,
        *,
        speech_handle: SpeechHandle,
        generation_ev: multimodal.GenerationCreatedEvent,
    ) -> None:
        assert self._rt_session is not None, "rt_session is not available"

        debug.Tracing.log_event(
            "realtime generation started",
            {"speech_id": speech_handle.id, "step_index": speech_handle.step_index},
        )

        audio_output = self._agent.output.audio
        text_output = self._agent.output.text

        await speech_handle.wait_if_not_interrupted(
            [asyncio.ensure_future(speech_handle._wait_for_authorization())]
        )

        if speech_handle.interrupted:
            # TODO(theomonnom): remove the message from the serverside history
            speech_handle._mark_playout_done()
            return

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
                name="_realtime_reply_task.read_messages",
            )
        ]

        exe_task, fnc_outputs = tools.perform_tool_executions(
            agent_ctx=AgentContext(self._agent),
            fnc_ctx=self._task.fnc_ctx,
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

                debug.Tracing.log_event(
                    "playout interrupted",
                    {
                        "playback_position": playback_ev.playback_position,
                        "speech_id": speech_handle.id,
                    },
                )

            speech_handle._mark_playout_done()
            # TODO(theomonnom): truncate message (+ OAI serverside mesage)
            return

        if len(fnc_outputs) > 0:
            new_fnc_outputs: list[llm.FunctionCallOutput] = []
            new_agent_task: AgentTask | None = None
            ignore_task_switch = False
            for fnc_output, agent_task in fnc_outputs:
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
                except multimodal.RealtimeError as e:
                    logger.warning(
                        "failed to update chat context before generating the function calls results",
                        extra={"error": str(e)},
                    )

                self._rt_session.interrupt()
                try:
                    await self._rt_session.generate_reply()
                except multimodal.RealtimeError as e:
                    logger.warning(
                        "failed to generate the function calls results",
                        extra={"error": str(e)},
                    )

            if not ignore_task_switch and new_agent_task is not None:
                self._agent.update_task(new_agent_task)

        debug.Tracing.log_event("playout completed", {"speech_id": speech_handle.id})
        speech_handle._mark_playout_done()
