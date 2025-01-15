from __future__ import annotations, print_function

import asyncio
import contextlib
import heapq
from dataclasses import dataclass
from typing import (
    AsyncIterable,
    Literal,
    Optional,
    Tuple,
    Union,
)

from livekit import rtc

from .. import debug, llm, stt, tokenize, tts, utils, vad
from ..llm import ChatContext, FunctionContext
from ..log import logger
from . import events, io
from .audio_recognition import AudioRecognition, _TurnDetector
from .generation import (
    _TTSGenerationData,
    do_llm_inference,
    do_tts_inference,
)


class AgentContext:
    pass


class SpeechHandle:
    def __init__(
        self, *, speech_id: str, allow_interruptions: bool, step_index: int
    ) -> None:
        self._id = speech_id
        self._step_index = step_index
        self._allow_interruptions = allow_interruptions
        self._interrupt_fut = asyncio.Future()
        self._done_fut = asyncio.Future()
        self._play_fut = asyncio.Future()
        self._playout_done_fut = asyncio.Future()

    @staticmethod
    def create(allow_interruptions: bool = True, step_index: int = 0) -> SpeechHandle:
        return SpeechHandle(
            speech_id=utils.shortuuid("speech_"),
            allow_interruptions=allow_interruptions,
            step_index=step_index,
        )

    @property
    def id(self) -> str:
        return self._id

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def interrupted(self) -> bool:
        return self._interrupt_fut.done()

    @property
    def allow_interruptions(self) -> bool:
        return self._allow_interruptions

    def play(self) -> None:
        self._play_fut.set_result(None)

    def done(self) -> bool:
        return self._done_fut.done()

    def interrupt(self) -> None:
        if not self._allow_interruptions:
            raise ValueError("This generation handle does not allow interruptions")

        if self.done():
            return

        self._done_fut.set_result(None)
        self._interrupt_fut.set_result(None)

    async def wait_for_playout(self) -> None:
        await asyncio.shield(self._playout_done_fut)

    def _mark_playout_done(self) -> None:
        self._playout_done_fut.set_result(None)

    def _mark_done(self) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            # will raise InvalidStateError if the future is already done (interrupted)
            self._done_fut.set_result(None)


EventTypes = Literal[
    "user_started_speaking",
    "user_stopped_speaking",
    "agent_started_speaking",
    "agent_stopped_speaking",
    "user_message_committed",
    "agent_message_committed",
    "agent_message_interrupted",
]


@dataclass
class _PipelineOptions:
    language: str | None
    allow_interruptions: bool
    min_interruption_duration: float
    min_endpointing_delay: float
    max_fnc_steps: int


class PipelineAgent(rtc.EventEmitter[EventTypes]):
    SPEECH_PRIORITY_LOW = 0
    """Priority for messages that should be played after all other messages in the queue"""
    SPEECH_PRIORITY_NORMAL = 5
    """Every speech generates by the PipelineAgent defaults to this priority."""
    SPEECH_PRIORITY_HIGH = 10
    """Priority for important messages that should be played before others."""

    def __init__(
        self,
        *,
        llm: llm.LLM | None = None,
        vad: vad.VAD | None = None,
        stt: stt.STT | None = None,
        tts: tts.TTS | None = None,
        turn_detector: _TurnDetector | None = None,
        language: str | None = None,
        chat_ctx: ChatContext | None = None,
        fnc_ctx: FunctionContext | None = None,
        allow_interruptions: bool = True,
        min_interruption_duration: float = 0.5,
        min_endpointing_delay: float = 0.5,
        max_fnc_steps: int = 5,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__()
        self._loop = loop or asyncio.get_event_loop()

        self._chat_ctx = chat_ctx or ChatContext()
        self._fnc_ctx = fnc_ctx

        self._stt, self._vad, self._llm, self._tts = stt, vad, llm, tts

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

        self._turn_detector = turn_detector
        self._audio_recognition = AudioRecognition(
            agent=self,
            stt=self.stt_node,
            vad=vad,
            turn_detector=turn_detector,
            min_endpointing_delay=min_endpointing_delay,
            chat_ctx=self._chat_ctx,
            loop=self._loop,
        )

        self._opts = _PipelineOptions(
            language=language,
            allow_interruptions=allow_interruptions,
            min_interruption_duration=min_interruption_duration,
            min_endpointing_delay=min_endpointing_delay,
            max_fnc_steps=max_fnc_steps,
        )

        self._audio_recognition.on("end_of_turn", self._on_audio_end_of_turn)
        self._audio_recognition.on("start_of_speech", self._on_start_of_speech)
        self._audio_recognition.on("end_of_speech", self._on_end_of_speech)
        self._audio_recognition.on("vad_inference_done", self._on_vad_inference_done)

        # configurable IO
        self._input = io.AgentInput(
            self._on_video_input_changed, self._on_audio_input_changed
        )
        self._output = io.AgentOutput(
            self._on_video_output_changed,
            self._on_audio_output_changed,
            self._on_text_output_changed,
        )

        self._current_speech: SpeechHandle | None = None
        self._speech_q: list[Tuple[int, SpeechHandle]] = []
        self._speech_q_changed = asyncio.Event()
        self._speech_tasks = []

        self._speech_scheduler_task: asyncio.Task | None = None

    # -- Pipeline nodes --
    # They can all be overriden by subclasses, by default they use the STT/LLM/TTS specified in the
    # constructor of the PipelineAgent

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

    def start(self) -> None:
        self._audio_recognition.start()
        self._speech_scheduler_task = asyncio.create_task(
            self._playout_scheduler(), name="_playout_scheduler"
        )

    async def aclose(self) -> None:
        await self._audio_recognition.aclose()

    def emit(self, event: EventTypes, *args) -> None:
        debug.Tracing.log_event(f'agent.on("{event}")')
        return super().emit(event, *args)

    @property
    def input(self) -> io.AgentInput:
        return self._input

    @property
    def output(self) -> io.AgentOutput:
        return self._output

    # TODO(theomonnom): find a better name than `generation`
    @property
    def current_speech(self) -> SpeechHandle | None:
        return self._current_speech

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    def update_options(self) -> None:
        pass

    def say(self, text: str | AsyncIterable[str]) -> SpeechHandle:
        pass

    def generate_reply(self, user_input: str) -> SpeechHandle:
        if self._current_speech is not None and not self._current_speech.interrupted:
            raise ValueError("another reply is already in progress")

        debug.Tracing.log_event("generate_reply", {"user_input": user_input})
        self._chat_ctx.append(role="user", text=user_input)  # TODO(theomonnom) Remove

        handle = SpeechHandle.create(allow_interruptions=self._opts.allow_interruptions)
        task = asyncio.create_task(
            self._generate_pipeline_reply_task(
                handle=handle,
                chat_ctx=self._chat_ctx,
                fnc_ctx=self._fnc_ctx,
            ),
            name="_generate_pipeline_reply",
        )
        self._schedule_speech(handle, task, self.SPEECH_PRIORITY_NORMAL)
        return handle

    # -- Main generation task --

    def _schedule_speech(
        self, speech: SpeechHandle, task: asyncio.Task, priority: int
    ) -> None:
        self._speech_tasks.append(task)
        task.add_done_callback(lambda _: self._speech_tasks.remove(task))

        heapq.heappush(self._speech_q, (priority, speech))
        self._speech_q_changed.set()

    @utils.log_exceptions(logger=logger)
    async def _playout_scheduler(self) -> None:
        while True:
            await self._speech_q_changed.wait()

            while self._speech_q:
                _, speech = heapq.heappop(self._speech_q)
                self._current_speech = speech
                speech.play()
                await speech.wait_for_playout()
                self._current_speech = None

            self._speech_q_changed.clear()

    @utils.log_exceptions(logger=logger)
    async def _generate_pipeline_reply_task(
        self,
        *,
        handle: SpeechHandle,
        chat_ctx: ChatContext,
        fnc_ctx: FunctionContext | None,
    ) -> None:
        @utils.log_exceptions(logger=logger)
        async def _forward_llm_text(llm_output: AsyncIterable[str]) -> None:
            """collect and forward the generated text to the current agent output"""
            if self.output.text is None:
                return

            try:
                async for delta in llm_output:
                    await self.output.text.capture_text(delta)
            finally:
                self.output.text.flush()

        @utils.log_exceptions(logger=logger)
        async def _forward_tts_audio(tts_output: AsyncIterable[rtc.AudioFrame]) -> None:
            """collect and forward the generated audio to the current agent output (generally playout)"""
            if self.output.audio is None:
                return

            try:
                async for frame in tts_output:
                    await self.output.audio.capture_frame(frame)
            finally:
                self.output.audio.flush()  # always flush (even if the task is interrupted)

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
                            "speech_id": handle.id,
                        },
                    )
                    debug.Tracing.log_event(
                        "executing tool",
                        {
                            "function": tool.function_info.name,
                            "speech_id": handle.id,
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
                            "speech_id": handle.id,
                        },
                    )
                    debug.Tracing.log_event(
                        "waiting for function call to finish before cancelling",
                        {
                            "functions": names,
                            "speech_id": handle.id,
                        },
                    )
                    await asyncio.gather(*[cfn.task for cfn in pending_tools])
            finally:
                if len(called_functions) > 0:
                    logger.debug(
                        "tools execution completed",
                        extra={"speech_id": handle.id},
                    )
                    debug.Tracing.log_event(
                        "tools execution completed",
                        {"speech_id": handle.id},
                    )

        debug.Tracing.log_event(
            "generation started",
            {"speech_id": handle.id, "step_index": handle.step_index},
        )

        wg = utils.aio.WaitGroup()
        tasks = []
        llm_task, llm_gen_data = do_llm_inference(
            node=self.llm_node,
            chat_ctx=chat_ctx,
            fnc_ctx=(
                fnc_ctx
                if handle.step_index < self._opts.max_fnc_steps - 1
                and handle.step_index >= 2
                else None
            ),
        )
        tasks.append(llm_task)
        wg.add(1)
        llm_task.add_done_callback(lambda _: wg.done())
        tts_text_input, llm_output = utils.aio.itertools.tee(llm_gen_data.text_ch)

        tts_task: asyncio.Task | None = None
        tts_gen_data: _TTSGenerationData | None = None
        if self._output.audio is not None:
            tts_task, tts_gen_data = do_tts_inference(
                node=self.tts_node, input=tts_text_input
            )
            tasks.append(tts_task)
            wg.add(1)
            tts_task.add_done_callback(lambda _: wg.done())

        # wait for the play() method to be called
        await asyncio.wait(
            [
                handle._play_fut,
                handle._interrupt_fut,
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if handle.interrupted:
            await utils.aio.gracefully_cancel(*tasks)
            handle._mark_done()
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
                handle._interrupt_fut,
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # wait for the end of the playout if the audio is enabled
        if forward_llm_task is not None:
            assert self._output.audio is not None
            await asyncio.wait(
                [
                    self._output.audio.wait_for_playout(),
                    handle._interrupt_fut,
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

        if handle.interrupted:
            await utils.aio.gracefully_cancel(*tasks)

            if len(called_functions) > 0:
                functions = [
                    cfnc.call_info.function_info.name for cfnc in called_functions
                ]
                logger.debug(
                    "speech interrupted, ignoring generation of the function calls results",
                    extra={"speech_id": handle.id, "functions": functions},
                )
                debug.Tracing.log_event(
                    "speech interrupted, ignoring generation of the function calls results",
                    {"speech_id": handle.id, "functions": functions},
                )

            # if the audio playout was enabled, clear the buffer
            if forward_tts_task is not None:
                assert self._output.audio is not None

                self._output.audio.clear_buffer()
                playback_ev = await self._output.audio.wait_for_playout()

                debug.Tracing.log_event(
                    "playout interrupted",
                    {
                        "playback_position": playback_ev.playback_position,
                        "speech_id": handle.id,
                    },
                )

                handle._mark_playout_done()
                # TODO(theomonnom): calculate the played text based on playback_ev.playback_position

            handle._mark_done()
            return

        handle._mark_playout_done()
        debug.Tracing.log_event("playout completed", {"speech_id": handle.id})

        if len(called_functions) > 0:
            if handle.step_index >= self._opts.max_fnc_steps:
                logger.warning(
                    "maximum number of function calls steps reached",
                    extra={"speech_id": handle.id},
                )
                debug.Tracing.log_event(
                    "maximum number of function calls steps reached",
                    {"speech_id": handle.id},
                )
                handle._mark_done()
                return

            # create a new SpeechHandle to generate the result of the function calls
            handle = SpeechHandle.create(
                allow_interruptions=self._opts.allow_interruptions,
                step_index=handle.step_index + 1,
            )
            task = asyncio.create_task(
                self._generate_pipeline_reply_task(
                    handle=handle,
                    chat_ctx=chat_ctx,
                    fnc_ctx=fnc_ctx,
                ),
                name="_generate_pipeline_reply",
            )
            self._schedule_speech(handle, task, self.SPEECH_PRIORITY_NORMAL)

        handle._mark_done()

    # -- Audio recognition --

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

    # ---

    # -- User changed input/output streams/sinks --

    def _on_video_input_changed(self) -> None:
        pass

    def _on_audio_input_changed(self) -> None:
        self._audio_recognition.audio_input = self._input.audio

    def _on_video_output_changed(self) -> None:
        pass

    def _on_audio_output_changed(self) -> None:
        pass

    def _on_text_output_changed(self) -> None:
        pass

    # ---
