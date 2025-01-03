from __future__ import annotations, print_function

import asyncio
import contextlib

from dataclasses import dataclass
from typing import (
    AsyncIterable,
    Callable,
    Literal,
    Optional,
    Union,
)

from livekit import rtc

from .. import llm, stt, tts, utils, vad
from ..llm import ChatContext, FunctionContext
from ..log import logger
from . import io
from .audio_recognition import AudioRecognition, _TurnDetector
from .generation import (
    do_llm_inference,
    do_tts_inference,
)


class AgentInput:
    def __init__(self, video_changed: Callable, audio_changed: Callable) -> None:
        self._video_stream: io.VideoStream | None = None
        self._audio_stream: io.AudioStream | None = None
        self._video_changed = video_changed
        self._audio_changed = audio_changed

    @property
    def video(self) -> io.VideoStream | None:
        return self._video_stream

    @video.setter
    def video(self, stream: io.VideoStream | None) -> None:
        self._video_stream = stream
        self._video_changed()

    @property
    def audio(self) -> io.AudioStream | None:
        return self._audio_stream

    @audio.setter
    def audio(self, stream: io.AudioStream | None) -> None:
        self._audio_stream = stream
        self._audio_changed()


class AgentOutput:
    def __init__(
        self, video_changed: Callable, audio_changed: Callable, text_changed: Callable
    ) -> None:
        self._video_sink: io.VideoSink | None = None
        self._audio_sink: io.AudioSink | None = None
        self._text_sink: io.TextSink | None = None
        self._video_changed = video_changed
        self._audio_changed = audio_changed
        self._text_changed = text_changed

    @property
    def video(self) -> io.VideoSink | None:
        return self._video_sink

    @video.setter
    def video(self, sink: io.VideoSink | None) -> None:
        self._video_sink = sink
        self._video_changed()

    @property
    def audio(self) -> io.AudioSink | None:
        return self._audio_sink

    @audio.setter
    def audio(self, sink: io.AudioSink | None) -> None:
        self._audio_sink = sink
        self._audio_changed()

    @property
    def text(self) -> io.TextSink | None:
        return self._text_sink

    @text.setter
    def text(self, sink: io.TextSink | None) -> None:
        self._text_sink = sink
        self._text_changed()


class GenerationHandle:
    def __init__(
        self, *, speech_id: str, allow_interruptions: bool, task: asyncio.Task
    ) -> None:
        self._id = speech_id
        self._allow_interruptions = allow_interruptions
        self._interrupted = False
        self._task = task

    @staticmethod
    def from_task(
        task: asyncio.Task, *, allow_interruptions: bool = True
    ) -> GenerationHandle:
        return GenerationHandle(
            speech_id=utils.shortuuid("gen_"),
            allow_interruptions=allow_interruptions,
            task=task,
        )

    @property
    def id(self) -> str:
        return self._id

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    @property
    def allow_interruptions(self) -> bool:
        return self._allow_interruptions

    def interrupt(self) -> None:
        if not self._allow_interruptions:
            raise ValueError("This generation handle does not allow interruptions")

        if self._task.done():
            return

        self._interrupted = True
        self._task.cancel()


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


class PipelineAgent(rtc.EventEmitter[EventTypes]):
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
        min_interruption_duration: float = 1.0,
        min_endpointing_delay: float = 0.5,
        max_fnc_steps: int = 5,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__()
        self._loop = loop or asyncio.get_event_loop()

        self._chat_ctx = chat_ctx or ChatContext()
        self._fnc_ctx = fnc_ctx

        self._stt, self._vad, self._llm, self._tts = stt, vad, llm, tts
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
        )

        self._max_fnc_steps = max_fnc_steps
        self._audio_recognition.on("end_of_turn", self._on_audio_end_of_turn)
        self._audio_recognition.on("vad_inference_done", self._on_vad_inference_done)

        # configurable IO
        self._input = AgentInput(
            self._on_video_input_changed, self._on_audio_input_changed
        )
        self._output = AgentOutput(
            self._on_video_output_changed,
            self._on_audio_output_changed,
            self._on_text_output_changed,
        )

        # current generation happening (including all function calls & steps)
        self._current_generation: GenerationHandle | None = None

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
                forward_task.cancel()

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

    async def aclose(self) -> None:
        await self._audio_recognition.aclose()

    @property
    def input(self) -> AgentInput:
        return self._input

    @property
    def output(self) -> AgentOutput:
        return self._output

    # TODO(theomonnom): find a better name than `generation`
    @property
    def current_generation(self) -> GenerationHandle | None:
        return self._current_generation

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    def update_options(self) -> None:
        pass

    def say(self, text: str | AsyncIterable[str]) -> GenerationHandle:
        pass

    def generate_reply(self, user_input: str) -> GenerationHandle:
        if (
            self._current_generation is not None
            and not self._current_generation.interrupted
        ):
            raise ValueError("another reply is already in progress")

        self._chat_ctx.append(role="user", text=user_input)

        # TODO(theomonnom): Use the agent task chat_ctx
        task = asyncio.create_task(
            self._generate_reply_task(chat_ctx=self._chat_ctx, fnc_ctx=self._fnc_ctx)
        )
        gen_handle = GenerationHandle.from_task(task)
        return gen_handle

    # -- Main generation task --

    @utils.log_exceptions(logger=logger)
    async def _generate_reply_task(
        self, *, chat_ctx: ChatContext, fnc_ctx: FunctionContext | None
    ) -> None:
        @utils.log_exceptions(logger=logger)
        async def _forward_llm_text(llm_output: AsyncIterable[str]) -> None:
            """collect and forward the generated text to the current agent output"""
            if self.output.text is None:
                return

            async for delta in llm_output:
                await self.output.text.capture_text(delta)

            self.output.text.flush()

        @utils.log_exceptions(logger=logger)
        async def _forward_tts_audio(
            tts_output: AsyncIterable[rtc.AudioFrame], wait_for_playout: bool = True
        ) -> None:
            """collect and forward the generated audio to the current agent output"""
            if self.output.audio is None:
                return

            async for frame in tts_output:
                await self.output.audio.capture_frame(frame)

            self.output.audio.flush()

        # new messages generated during the generation (including function calls)
        new_messages: list[llm.ChatMessage] = []

        # TODO(theomonnom): how nested fnc calls is going to work with realtime API?
        for i in range(
            self._max_fnc_steps + 1
        ):  # +1 to ignore the first step that doesn't contain any tools
            # if i >= 2, the LLM supports having multiple steps
            fnc_ctx = fnc_ctx if i < self._max_fnc_steps - 1 and i >= 2 else None
            chat_ctx = chat_ctx

            llm_task, llm_gen_data = do_llm_inference(
                node=self.llm_node, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx
            )
            tts_text_input, llm_output = utils.aio.itertools.tee(llm_gen_data.text_ch)
            forward_llm_task = asyncio.create_task(
                _forward_llm_text(llm_output),
                name="_generate_reply_task.forward_llm_text",
            )

            tts_task: asyncio.Task | None = None
            forward_tts_task: asyncio.Task | None = None
            if self._output.audio is not None:
                tts_task, tts_gen_data = do_tts_inference(
                    node=self.tts_node, input=tts_text_input
                )
                forward_tts_task = asyncio.create_task(
                    _forward_tts_audio(tts_gen_data.audio_ch),
                    name="_generate_reply_task.forward_tts_audio",
                )

            tools: list[llm.FunctionCallInfo] = []
            async for tool in llm_gen_data.tools_ch:
                tools.append(tool)  # TODO(theomonnom): optimize function calls response

            await asyncio.gather(llm_task, forward_llm_task)

            # TODO(theomonnom): Simplify this
            if tts_task is not None and forward_tts_task is not None:
                assert self._output.audio is not None
                await asyncio.gather(tts_task, forward_tts_task)
                playback_ev = await self._output.audio.wait_for_playout()

            generated_text = llm_gen_data.generated_text
            if len(generated_text) > 0:
                new_messages.append(
                    llm.ChatMessage(role="assistant", content=generated_text)
                )

            if len(tools) == 0:
                break  # no more fnc step needed

    # -- Audio recognition --

    def _on_audio_end_of_turn(self, new_transcript: str) -> None:
        # When the audio recognition detects the end of a user turn:
        #  - check if there is no current generation happening
        #  - cancel the current generation if it allows interruptions (otherwise skip this current
        #  turn)
        #  - generate a reply to the user input

        if self._current_generation is not None:
            if self._current_generation.allow_interruptions:
                logger.warning(
                    "skipping user input, current speech generation cannot be interrupted",
                    extra={"user_input": new_transcript},
                )
                return

            self._current_generation.interrupt()

        self.generate_reply(new_transcript)

    def _on_vad_inference_done(self, ev: vad.VADEvent) -> None:
        if ev.speech_duration > self._opts.min_interruption_duration:
            if self._current_generation is not None:
                self._current_generation.interrupt()

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
