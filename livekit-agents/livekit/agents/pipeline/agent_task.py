from __future__ import annotations

import asyncio
from typing import (
    AsyncIterable,
    Optional,
    Union,
)

from livekit import rtc

from .. import llm, multimodal, stt, tokenize, tts, utils, vad, debug
from ..llm import ChatContext, FunctionContext, find_ai_functions
from ..log import logger
from .agent_task import AgentTask
from .audio_recognition import AudioRecognition, _TurnDetector
from .pipeline_agent import PipelineAgent, SpeechHandle
from .generation import (
    do_llm_inference,
    do_tts_inference,
    _TTSGenerationData,
    _LLMGenerationData,
)


class AgentTask:
    def __init__(
        self,
        *,
        instructions: str,
        chat_ctx: llm.ChatContext | None = None,
        fnc_ctx: llm.FunctionContext | None = None,
        turn_detector: _TurnDetector | None = None,
        stt: stt.STT | None = None,
        vad: vad.VAD | None = None,
        llm: llm.LLM | multimodal.RealtimeModel | None = None,
        tts: tts.TTS | None = None,
    ) -> None:
        if tts and not tts.capabilities.streaming:
            from .. import tts as text_to_speech

            tts = text_to_speech.StreamAdapter(
                tts=tts, sentence_tokenizer=tokenize.basic.SentenceTokenizer()
            )

        if stt and not stt.capabilities.streaming:
            from .. import stt as speech_to_text

            if vad is None:
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
            list(self._fnc_ctx.ai_functions.values())
            + find_ai_functions(self.__class__)
        )
        self._turn_detector = turn_detector
        self._stt, self._llm, self._tts, self._vad = stt, llm, tts, vad

        self._agent: PipelineAgent | None = None
        self._rt_session: multimodal.RealtimeSession | None = None
        self._audio_recognition: AudioRecognition | None = None

    @property
    def instructions(self) -> str:
        return self._instructions

    @instructions.setter
    def instructions(self, instructions: str) -> None:
        self._instructions = instructions

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

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
                await utils.aio.gracefully_cancel(forward_task)

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
                await utils.aio.gracefully_cancel(forward_task)

    async def _on_start(self, agent: PipelineAgent) -> None:
        if self._rt_session is not None:
            logger.warning("starting a new task while rt_session is not None")

        self._audio_recognition = AudioRecognition(
            task=self,
            stt=self.stt_node,
            vad=self._vad,
            turn_detector=self._turn_detector,
            min_endpointing_delay=agent.options.min_endpointing_delay,
        )
        self._audio_recognition.start()

        if isinstance(self._llm, multimodal.RealtimeModel):
            self._rt_session = self._llm.session()
            self._rt_session.on("generation_created", self._on_generation_created)
            await self._rt_session.update_chat_ctx(self._chat_ctx)

    async def _on_close(self) -> None:
        if self._rt_session is not None:
            await self._rt_session.aclose()

        if self._audio_recognition is not None:
            await self._audio_recognition.aclose()

    def _on_generation_created(self, ev: multimodal.GenerationCreatedEvent) -> None:
        pass

    def _on_input_audio_frame(self, frame: rtc.AudioFrame) -> None:
        if self._rt_session is not None:
            self._rt_session.push_audio(frame)

        if self._audio_recognition is not None:
            self._audio_recognition.push_audio(frame)

    def _on_audio_end_of_turn(self, new_transcript: str) -> None:
        # When the audio recognition detects the end of a user turn:
        #  - check if there is no current generation happening
        #  - cancel the current generation if it allows interruptions (otherwise skip this current
        #  turn)
        #  - generate a reply to the user input

        if self._current_speech is not None:
            if self._current_speech.allow_interruptions:
                logger.warning(
                    "skipping user input, current speech generation cannot be interrupted",
                    extra={"user_input": new_transcript},
                )
                return

            debug.Tracing.log_event(
                "speech interrupted, new user turn detected",
                {"speech_id": self._current_speech.id},
            )
            self._current_speech.interrupt()

        self.generate_reply(new_transcript)

    def _on_vad_inference_done(self, ev: vad.VADEvent) -> None:
        if ev.speech_duration > self._opts.min_interruption_duration:
            if (
                self._current_speech is not None
                and not self._current_speech.interrupted
                and self._current_speech.allow_interruptions
            ):
                debug.Tracing.log_event(
                    "speech interrupted by vad",
                    {"speech_id": self._current_speech.id},
                )
                self._current_speech.interrupt()

    def _on_start_of_speech(self, _: vad.VADEvent) -> None:
        self.emit("user_started_speaking", events.UserStartedSpeakingEvent())

    def _on_end_of_speech(self, _: vad.VADEvent) -> None:
        self.emit("user_stopped_speaking", events.UserStoppedSpeakingEvent())



    @utils.log_exceptions(logger=logger)
    async def _generate_pipeline_reply_task(
        self,
        *,
        speech_handle: SpeechHandle,
    ) -> None:
        assert self._agent is not None, "agent is not set"
        agent = self._agent

        @utils.log_exceptions(logger=logger)
        async def _forward_llm_text(llm_output: AsyncIterable[str]) -> None:
            """collect and forward the generated text to the current agent output"""
            try:
                async for delta in llm_output:
                    if agent.output.text is None:
                        break

                    await agent.output.text.capture_text(delta)
            finally:
                if agent.output.text is not None:
                    agent.output.text.flush()

        @utils.log_exceptions(logger=logger)
        async def _forward_tts_audio(tts_output: AsyncIterable[rtc.AudioFrame]) -> None:
            """collect and forward the generated audio to the current agent output (generally playout)"""
            try:
                async for frame in tts_output:
                    if agent.output.audio is None:
                        break
                    await agent.output.audio.capture_frame(frame)
            finally:
                if agent.output.audio is not None:
                    agent.output.audio.flush()  # always flush (even if the task is interrupted)

        @utils.log_exceptions(logger=logger)
        async def _execute_tools(
            tools_ch: utils.aio.Chan[llm.FunctionCallInfo],
            called_functions: set[llm.CalledFunction],
        ) -> None:
            """execute tools, when cancelled, stop executing new tools but wait for the pending ones"""
            try:
                async for tool in tools_ch:
                    logger.debug(
                        "executing tool",
                        extra={
                            "function": tool.function_info.name,
                            "speech_id": speech_handle.id,
                        },
                    )
                    debug.Tracing.log_event(
                        "executing tool",
                        {
                            "function": tool.function_info.name,
                            "speech_id": speech_handle.id,
                        },
                    )
                    cfnc = tool.execute()
                    called_functions.add(cfnc)
            except asyncio.CancelledError:
                # don't allow to cancel running function calla if they're still running
                pending_tools = [cfn for cfn in called_functions if not cfn.task.done()]

                if pending_tools:
                    names = [cfn.call_info.function_info.name for cfn in pending_tools]

                    logger.debug(
                        "waiting for function call to finish before cancelling",
                        extra={
                            "functions": names,
                            "speech_id": speech_handle.id,
                        },
                    )
                    debug.Tracing.log_event(
                        "waiting for function call to finish before cancelling",
                        {
                            "functions": names,
                            "speech_id": speech_handle.id,
                        },
                    )
                    await asyncio.gather(*[cfn.task for cfn in pending_tools])
            finally:
                if len(called_functions) > 0:
                    logger.debug(
                        "tools execution completed",
                        extra={"speech_id": speech_handle.id},
                    )
                    debug.Tracing.log_event(
                        "tools execution completed",
                        {"speech_id": speech_handle.id},
                    )

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
            await utils.aio.gracefully_cancel(*tasks)
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
        if forward_llm_task is not None and self._agent.output.audio is not None:
            await asyncio.wait(
                [
                    self._agent.output.audio.wait_for_playout(),
                    speech_handle._interrupt_fut,
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

        if speech_handle.interrupted:
            await utils.aio.gracefully_cancel(*tasks)

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

                speech_handle._mark_playout_done()
                # TODO(theomonnom): calculate the played text based on playback_ev.playback_position

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
                self._generate_pipeline_reply_task(
                    handle=speech_handle,
                    chat_ctx=self._chat_ctx,
                    fnc_ctx=self._fnc_ctx,
                ),
                name="_generate_pipeline_reply",
            )
            self._agent._schedule_speech(
                speech_handle, task, PipelineAgent.SPEECH_PRIORITY_NORMAL
            )

        speech_handle._mark_done()
