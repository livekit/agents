from __future__ import annotations

import asyncio
import contextvars
import time
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Literal,
    Optional,
    Union,
)

from livekit import rtc

from .. import metrics, stt, tokenize, tts, utils, vad
from .._constants import ATTRIBUTE_AGENT_STATE
from .._types import AgentState
from ..llm import LLM, ChatContext, ChatMessage, FunctionContext, LLMStream
from .agent_output import AgentOutput, SpeechSource, SynthesisHandle
from .agent_playout import AgentPlayout
from .human_input import HumanInput
from .log import logger
from .plotter import AssistantPlotter
from .speech_handle import SpeechHandle

BeforeLLMCallback = Callable[
    ["VoicePipelineAgent", ChatContext],
    Union[
        Optional[LLMStream],
        Awaitable[Optional[LLMStream]],
        Literal[False],
        Awaitable[Literal[False]],
    ],
]

WillSynthesizeAssistantReply = BeforeLLMCallback

BeforeTTSCallback = Callable[
    ["VoicePipelineAgent", Union[str, AsyncIterable[str]]],
    SpeechSource,
]


EventTypes = Literal[
    "user_started_speaking",
    "user_stopped_speaking",
    "agent_started_speaking",
    "agent_stopped_speaking",
    "user_speech_committed",
    "agent_speech_committed",
    "agent_speech_interrupted",
    "function_calls_collected",
    "function_calls_finished",
    "metrics_collected",
]

_CallContextVar = contextvars.ContextVar["AgentCallContext"](
    "voice_assistant_contextvar"
)


class AgentCallContext:
    def __init__(self, assistant: "VoicePipelineAgent", llm_stream: LLMStream) -> None:
        self._assistant = assistant
        self._metadata = dict[str, Any]()
        self._llm_stream = llm_stream

    @staticmethod
    def get_current() -> "AgentCallContext":
        return _CallContextVar.get()

    @property
    def agent(self) -> "VoicePipelineAgent":
        return self._assistant

    def store_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    def llm_stream(self) -> LLMStream:
        return self._llm_stream


def _default_before_llm_cb(
    agent: VoicePipelineAgent, chat_ctx: ChatContext
) -> LLMStream:
    return agent.llm.chat(
        chat_ctx=chat_ctx,
        fnc_ctx=agent.fnc_ctx,
    )


@dataclass
class SpeechData:
    sequence_id: str


SpeechDataContextVar = contextvars.ContextVar[SpeechData]("voice_assistant_speech_data")


def _default_before_tts_cb(
    agent: VoicePipelineAgent, text: str | AsyncIterable[str]
) -> str | AsyncIterable[str]:
    return text


@dataclass(frozen=True)
class _ImplOptions:
    allow_interruptions: bool
    int_speech_duration: float
    int_min_words: int
    min_endpointing_delay: float
    max_nested_fnc_calls: int
    preemptive_synthesis: bool
    before_llm_cb: BeforeLLMCallback
    before_tts_cb: BeforeTTSCallback
    plotting: bool
    transcription: AgentTranscriptionOptions


@dataclass(frozen=True)
class AgentTranscriptionOptions:
    user_transcription: bool = True
    """Whether to forward the user transcription to the client"""
    agent_transcription: bool = True
    """Whether to forward the agent transcription to the client"""
    agent_transcription_speed: float = 1.0
    """The speed at which the agent's speech transcription is forwarded to the client.
    We try to mimic the agent's speech speed by adjusting the transcription speed."""
    sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer()
    """The tokenizer used to split the speech into sentences.
    This is used to decide when to mark a transcript as final for the agent transcription."""
    word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(
        ignore_punctuation=False
    )
    """The tokenizer used to split the speech into words.
    This is used to simulate the "interim results" of the agent transcription."""
    hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word
    """A function that takes a string (word) as input and returns a list of strings,
    representing the hyphenated parts of the word."""


class VoicePipelineAgent(utils.EventEmitter[EventTypes]):
    """
    A pipeline agent (VAD + STT + LLM + TTS) implementation.
    """

    MIN_TIME_PLAYED_FOR_COMMIT = 1.5
    """Minimum time played for the user speech to be committed to the chat context"""

    def __init__(
        self,
        *,
        vad: vad.VAD,
        stt: stt.STT,
        llm: LLM,
        tts: tts.TTS,
        chat_ctx: ChatContext | None = None,
        fnc_ctx: FunctionContext | None = None,
        allow_interruptions: bool = True,
        interrupt_speech_duration: float = 0.5,
        interrupt_min_words: int = 0,
        min_endpointing_delay: float = 0.5,
        max_nested_fnc_calls: int = 1,
        preemptive_synthesis: bool = False,
        transcription: AgentTranscriptionOptions = AgentTranscriptionOptions(),
        before_llm_cb: BeforeLLMCallback = _default_before_llm_cb,
        before_tts_cb: BeforeTTSCallback = _default_before_tts_cb,
        plotting: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        # backward compatibility
        will_synthesize_assistant_reply: WillSynthesizeAssistantReply | None = None,
    ) -> None:
        """
        Create a new VoicePipelineAgent.

        Args:
            vad: Voice Activity Detection (VAD) instance.
            stt: Speech-to-Text (STT) instance.
            llm: Large Language Model (LLM) instance.
            tts: Text-to-Speech (TTS) instance.
            chat_ctx: Chat context for the assistant.
            fnc_ctx: Function context for the assistant.
            allow_interruptions: Whether to allow the user to interrupt the assistant.
            interrupt_speech_duration: Minimum duration of speech to consider for interruption.
            interrupt_min_words: Minimum number of words to consider for interruption.
                Defaults to 0 as this may increase the latency depending on the STT.
            min_endpointing_delay: Delay to wait before considering the user finished speaking.
            max_nested_fnc_calls: Maximum number of nested function calls allowed for chaining
                function calls (e.g functions that depend on each other).
            preemptive_synthesis: Whether to preemptively synthesize responses.
            transcription: Options for assistant transcription.
            before_llm_cb: Callback called when the assistant is about to synthesize a reply.
                This can be used to customize the reply (e.g: inject context/RAG).

                Returning None will create a default LLM stream. You can also return your own llm
                stream by calling the llm.chat() method.

                Returning False will cancel the synthesis of the reply.
            before_tts_cb: Callback called when the assistant is about to
                synthesize a speech. This can be used to customize text before the speech synthesis.
                (e.g: editing the pronunciation of a word).
            plotting: Whether to enable plotting for debugging. matplotlib must be installed.
            loop: Event loop to use. Default to asyncio.get_event_loop().
        """
        super().__init__()
        self._loop = loop or asyncio.get_event_loop()

        if will_synthesize_assistant_reply is not None:
            logger.warning(
                "will_synthesize_assistant_reply is deprecated and will be removed in 1.5.0, use before_llm_cb instead",
            )
            before_llm_cb = will_synthesize_assistant_reply

        self._opts = _ImplOptions(
            plotting=plotting,
            allow_interruptions=allow_interruptions,
            int_speech_duration=interrupt_speech_duration,
            int_min_words=interrupt_min_words,
            min_endpointing_delay=min_endpointing_delay,
            max_nested_fnc_calls=max_nested_fnc_calls,
            preemptive_synthesis=preemptive_synthesis,
            transcription=transcription,
            before_llm_cb=before_llm_cb,
            before_tts_cb=before_tts_cb,
        )
        self._plotter = AssistantPlotter(self._loop)

        # wrap with StreamAdapter automatically when streaming is not supported on a specific TTS/STT.
        # To override StreamAdapter options, create the adapter manually.

        if not tts.capabilities.streaming:
            from .. import tts as text_to_speech

            tts = text_to_speech.StreamAdapter(
                tts=tts, sentence_tokenizer=tokenize.basic.SentenceTokenizer()
            )

        if not stt.capabilities.streaming:
            from .. import stt as speech_to_text

            stt = speech_to_text.StreamAdapter(
                stt=stt,
                vad=vad,
            )

        self._stt, self._vad, self._llm, self._tts = stt, vad, llm, tts
        self._chat_ctx = chat_ctx or ChatContext()
        self._fnc_ctx = fnc_ctx
        self._started, self._closed = False, False

        self._human_input: HumanInput | None = None
        self._agent_output: AgentOutput | None = None

        # done when the agent output track is published
        self._track_published_fut = asyncio.Future[None]()

        self._pending_agent_reply: SpeechHandle | None = None
        self._agent_reply_task: asyncio.Task[None] | None = None

        self._playing_speech: SpeechHandle | None = None
        self._transcribed_text, self._transcribed_interim_text = "", ""

        self._deferred_validation = _DeferredReplyValidation(
            self._validate_reply_if_possible,
            self._opts.min_endpointing_delay,
            loop=self._loop,
        )

        self._speech_q: list[SpeechHandle] = []
        self._speech_q_changed = asyncio.Event()

        self._update_state_task: asyncio.Task | None = None

        self._last_final_transcript_time: float | None = None
        self._last_speech_time: float | None = None

    @property
    def fnc_ctx(self) -> FunctionContext | None:
        return self._fnc_ctx

    @fnc_ctx.setter
    def fnc_ctx(self, fnc_ctx: FunctionContext | None) -> None:
        self._fnc_ctx = fnc_ctx

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def llm(self) -> LLM:
        return self._llm

    @property
    def tts(self) -> tts.TTS:
        return self._tts

    @property
    def stt(self) -> stt.STT:
        return self._stt

    @property
    def vad(self) -> vad.VAD:
        return self._vad

    def start(
        self, room: rtc.Room, participant: rtc.RemoteParticipant | str | None = None
    ) -> None:
        """Start the voice assistant

        Args:
            room: the room to use
            participant: the participant to listen to, can either be a participant or a participant identity
                If None, the first participant found in the room will be selected
        """
        if self._started:
            raise RuntimeError("voice assistant already started")

        @self._stt.on("metrics_collected")
        def _on_stt_metrics(stt_metrics: metrics.STTMetrics) -> None:
            self.emit(
                "metrics_collected",
                metrics.PipelineSTTMetrics(
                    **stt_metrics.__dict__,
                ),
            )

        @self._tts.on("metrics_collected")
        def _on_tts_metrics(tts_metrics: metrics.TTSMetrics) -> None:
            speech_data = SpeechDataContextVar.get(None)
            if speech_data is None:
                return

            self.emit(
                "metrics_collected",
                metrics.PipelineTTSMetrics(
                    **tts_metrics.__dict__,
                    sequence_id=speech_data.sequence_id,
                ),
            )

        @self._llm.on("metrics_collected")
        def _on_llm_metrics(llm_metrics: metrics.LLMMetrics) -> None:
            speech_data = SpeechDataContextVar.get(None)
            if speech_data is None:
                return
            self.emit(
                "metrics_collected",
                metrics.PipelineLLMMetrics(
                    **llm_metrics.__dict__,
                    sequence_id=speech_data.sequence_id,
                ),
            )

        @self._vad.on("metrics_collected")
        def _on_vad_metrics(vad_metrics: vad.VADMetrics) -> None:
            self.emit(
                "metrics_collected", metrics.PipelineVADMetrics(**vad_metrics.__dict__)
            )

        room.on("participant_connected", self._on_participant_connected)
        self._room, self._participant = room, participant

        if participant is not None:
            if isinstance(participant, rtc.RemoteParticipant):
                self._link_participant(participant.identity)
            else:
                self._link_participant(participant)
        else:
            # no participant provided, try to find the first participant in the room
            for participant in self._room.remote_participants.values():
                self._link_participant(participant.identity)
                break

        self._main_atask = asyncio.create_task(self._main_task())

    def on(self, event: EventTypes, callback: Callable[[Any], None] | None = None):
        """Register a callback for an event

        Args:
            event: the event to listen to (see EventTypes)
                - user_started_speaking: the user started speaking
                - user_stopped_speaking: the user stopped speaking
                - agent_started_speaking: the agent started speaking
                - agent_stopped_speaking: the agent stopped speaking
                - user_speech_committed: the user speech was committed to the chat context
                - agent_speech_committed: the agent speech was committed to the chat context
                - agent_speech_interrupted: the agent speech was interrupted
                - function_calls_collected: received the complete set of functions to be executed
                - function_calls_finished: all function calls have been completed
            callback: the callback to call when the event is emitted
        """
        return super().on(event, callback)

    async def say(
        self,
        source: str | LLMStream | AsyncIterable[str],
        *,
        allow_interruptions: bool = True,
        add_to_chat_ctx: bool = True,
    ) -> None:
        """
        Play a speech source through the voice assistant.

        Args:
            source: The source of the speech to play.
                It can be a string, an LLMStream, or an asynchronous iterable of strings.
            allow_interruptions: Whether to allow interruptions during the speech playback.
            add_to_chat_ctx: Whether to add the speech to the chat context.
        """
        await self._track_published_fut

        new_handle = SpeechHandle.create_assistant_speech(
            allow_interruptions=allow_interruptions, add_to_chat_ctx=add_to_chat_ctx
        )
        synthesis_handle = self._synthesize_agent_speech(new_handle.id, source)
        new_handle.initialize(source=source, synthesis_handle=synthesis_handle)
        self._add_speech_for_playout(new_handle)

    def _update_state(self, state: AgentState, delay: float = 0.0):
        """Set the current state of the agent"""

        @utils.log_exceptions(logger=logger)
        async def _run_task(delay: float) -> None:
            await asyncio.sleep(delay)

            if self._room.isconnected():
                await self._room.local_participant.set_attributes(
                    {ATTRIBUTE_AGENT_STATE: state}
                )

        if self._update_state_task is not None:
            self._update_state_task.cancel()

        self._update_state_task = asyncio.create_task(_run_task(delay))

    async def aclose(self) -> None:
        """Close the voice assistant"""
        if not self._started:
            return

        self._room.off("participant_connected", self._on_participant_connected)
        await self._deferred_validation.aclose()

    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        if self._human_input is not None:
            return

        self._link_participant(participant.identity)

    def _link_participant(self, identity: str) -> None:
        participant = self._room.remote_participants.get(identity)
        if participant is None:
            logger.error("_link_participant must be called with a valid identity")
            return

        self._human_input = HumanInput(
            room=self._room,
            vad=self._vad,
            stt=self._stt,
            participant=participant,
            transcription=self._opts.transcription.user_transcription,
        )

        def _on_start_of_speech(ev: vad.VADEvent) -> None:
            self._plotter.plot_event("user_started_speaking")
            self.emit("user_started_speaking")
            self._deferred_validation.on_human_start_of_speech(ev)

        def _on_vad_inference_done(ev: vad.VADEvent) -> None:
            if not self._track_published_fut.done():
                return

            assert self._agent_output is not None

            tv = 1.0
            if self._opts.allow_interruptions:
                tv = max(0.0, 1.0 - ev.probability)
                self._agent_output.playout.target_volume = tv

            smoothed_tv = self._agent_output.playout.smoothed_volume

            self._plotter.plot_value("raw_vol", tv)
            self._plotter.plot_value("smoothed_vol", smoothed_tv)
            self._plotter.plot_value("vad_probability", ev.probability)

            if ev.speech_duration >= self._opts.int_speech_duration:
                self._interrupt_if_possible()

            if ev.raw_accumulated_speech > 0.0:
                self._last_speech_time = (
                    time.perf_counter() - ev.raw_accumulated_silence
                )

        def _on_end_of_speech(ev: vad.VADEvent) -> None:
            self._plotter.plot_event("user_stopped_speaking")
            self.emit("user_stopped_speaking")
            self._deferred_validation.on_human_end_of_speech(ev)

        def _on_interim_transcript(ev: stt.SpeechEvent) -> None:
            self._transcribed_interim_text = ev.alternatives[0].text

        def _on_final_transcript(ev: stt.SpeechEvent) -> None:
            new_transcript = ev.alternatives[0].text
            if not new_transcript:
                return

            logger.debug(
                "received user transcript",
                extra={"user_transcript": new_transcript},
            )

            self._last_final_transcript_time = time.perf_counter()

            self._transcribed_text += (
                " " if self._transcribed_text else ""
            ) + new_transcript

            if self._opts.preemptive_synthesis:
                if (
                    self._playing_speech is None
                    or self._playing_speech.allow_interruptions
                ):
                    self._synthesize_agent_reply()

            self._deferred_validation.on_human_final_transcript(new_transcript)

            words = self._opts.transcription.word_tokenizer.tokenize(
                text=new_transcript
            )
            if len(words) >= 3:
                # VAD can sometimes not detect that the human is speaking
                # to make the interruption more reliable, we also interrupt on the final transcript.
                self._interrupt_if_possible()

        self._human_input.on("start_of_speech", _on_start_of_speech)
        self._human_input.on("vad_inference_done", _on_vad_inference_done)
        self._human_input.on("end_of_speech", _on_end_of_speech)
        self._human_input.on("interim_transcript", _on_interim_transcript)
        self._human_input.on("final_transcript", _on_final_transcript)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        if self._opts.plotting:
            await self._plotter.start()

        self._update_state("initializing")
        audio_source = rtc.AudioSource(self._tts.sample_rate, self._tts.num_channels)
        track = rtc.LocalAudioTrack.create_audio_track("assistant_voice", audio_source)
        self._agent_publication = await self._room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )

        agent_playout = AgentPlayout(audio_source=audio_source)
        self._agent_output = AgentOutput(
            room=self._room,
            agent_playout=agent_playout,
            llm=self._llm,
            tts=self._tts,
        )

        def _on_playout_started() -> None:
            self._plotter.plot_event("agent_started_speaking")
            self.emit("agent_started_speaking")
            self._update_state("speaking")

        def _on_playout_stopped(interrupted: bool) -> None:
            self._plotter.plot_event("agent_stopped_speaking")
            self.emit("agent_stopped_speaking")
            self._update_state("listening")

        agent_playout.on("playout_started", _on_playout_started)
        agent_playout.on("playout_stopped", _on_playout_stopped)

        self._track_published_fut.set_result(None)

        while True:
            await self._speech_q_changed.wait()

            while self._speech_q:
                speech = self._speech_q[0]
                self._playing_speech = speech
                await self._play_speech(speech)
                self._speech_q.pop(0)  # Remove the element only after playing
                self._playing_speech = None

            self._speech_q_changed.clear()

    def _synthesize_agent_reply(self):
        """Synthesize the agent reply to the user question, also make sure only one reply
        is synthesized/played at a time"""

        if self._pending_agent_reply is not None:
            self._pending_agent_reply.cancel()

        if self._human_input is not None and not self._human_input.speaking:
            self._update_state("thinking", 0.2)

        self._pending_agent_reply = new_handle = SpeechHandle.create_assistant_reply(
            allow_interruptions=self._opts.allow_interruptions,
            add_to_chat_ctx=True,
            user_question=self._transcribed_text,
        )

        self._agent_reply_task = asyncio.create_task(
            self._synthesize_answer_task(self._agent_reply_task, new_handle)
        )

    @utils.log_exceptions(logger=logger)
    async def _synthesize_answer_task(
        self, old_task: asyncio.Task[None], handle: SpeechHandle
    ) -> None:
        if old_task is not None:
            await utils.aio.gracefully_cancel(old_task)

        copied_ctx = self._chat_ctx.copy()
        playing_speech = self._playing_speech
        if playing_speech is not None and playing_speech.initialized:
            if (
                not playing_speech.user_question or playing_speech.user_committed
            ) and not playing_speech.speech_committed:
                # the speech is playing but not committed yet, add it to the chat context for this new reply synthesis
                copied_ctx.messages.append(
                    ChatMessage.create(
                        text=playing_speech.synthesis_handle.tts_forwarder.played_text,
                        role="assistant",
                    )
                )

        copied_ctx.messages.append(
            ChatMessage.create(text=handle.user_question, role="user")
        )

        tk = SpeechDataContextVar.set(SpeechData(sequence_id=handle.id))
        try:
            llm_stream = self._opts.before_llm_cb(self, copied_ctx)
            if asyncio.iscoroutine(llm_stream):
                llm_stream = await llm_stream

            if llm_stream is False:
                handle.cancel()
                return

            # fallback to default impl if no custom/user stream is returned
            if not isinstance(llm_stream, LLMStream):
                llm_stream = _default_before_llm_cb(self, chat_ctx=copied_ctx)

            if handle.interrupted:
                return

            synthesis_handle = self._synthesize_agent_speech(handle.id, llm_stream)
            handle.initialize(source=llm_stream, synthesis_handle=synthesis_handle)
        finally:
            SpeechDataContextVar.reset(tk)

    async def _play_speech(self, speech_handle: SpeechHandle) -> None:
        try:
            await speech_handle.wait_for_initialization()
        except asyncio.CancelledError:
            return

        await self._agent_publication.wait_for_subscription()

        synthesis_handle = speech_handle.synthesis_handle
        if synthesis_handle.interrupted:
            return

        user_question = speech_handle.user_question

        play_handle = synthesis_handle.play()
        join_fut = play_handle.join()

        def _commit_user_question_if_needed() -> None:
            if (
                not user_question
                or synthesis_handle.interrupted
                or speech_handle.user_committed
            ):
                return

            is_using_tools = isinstance(speech_handle.source, LLMStream) and len(
                speech_handle.source.function_calls
            )

            # make sure at least some speech was played before committing the user message
            # since we try to validate as fast as possible it is possible the agent gets interrupted
            # really quickly (barely audible), we don't want to mark this question as "answered".
            if (
                speech_handle.allow_interruptions
                and not is_using_tools
                and (
                    play_handle.time_played < self.MIN_TIME_PLAYED_FOR_COMMIT
                    and not join_fut.done()
                )
            ):
                return

            user_msg = ChatMessage.create(text=user_question, role="user")
            self._chat_ctx.messages.append(user_msg)
            self.emit("user_speech_committed", user_msg)

            self._transcribed_text = self._transcribed_text[len(user_question) :]
            speech_handle.mark_user_committed()

        # wait for the play_handle to finish and check every 1s if the user question should be committed
        _commit_user_question_if_needed()

        while not join_fut.done():
            await asyncio.wait(
                [join_fut], return_when=asyncio.FIRST_COMPLETED, timeout=0.2
            )

            _commit_user_question_if_needed()

            if speech_handle.interrupted:
                break

        _commit_user_question_if_needed()

        collected_text = speech_handle.synthesis_handle.tts_forwarder.played_text
        interrupted = speech_handle.interrupted
        is_using_tools = isinstance(speech_handle.source, LLMStream) and len(
            speech_handle.source.function_calls
        )

        extra_tools_messages = []  # additional messages from the functions to add to the context if needed

        # if the answer is using tools, execute the functions and automatically generate
        # a response to the user question from the returned values
        if is_using_tools and not interrupted:
            assert isinstance(speech_handle.source, LLMStream)
            assert (
                not user_question or speech_handle.user_committed
            ), "user speech should have been committed before using tools"

            llm_stream = speech_handle.source

            if collected_text:
                msg = ChatMessage.create(text=collected_text, role="assistant")
                self._chat_ctx.messages.append(msg)

                speech_handle.mark_speech_committed()
                self.emit("agent_speech_committed", msg)

            # execute functions
            call_ctx = AgentCallContext(self, llm_stream)
            tk = _CallContextVar.set(call_ctx)

            new_function_calls = llm_stream.function_calls

            for i in range(self._opts.max_nested_fnc_calls):
                self.emit("function_calls_collected", new_function_calls)

                called_fncs = []
                for fnc in new_function_calls:
                    called_fnc = fnc.execute()
                    called_fncs.append(called_fnc)
                    logger.debug(
                        "executing ai function",
                        extra={
                            "function": fnc.function_info.name,
                            "speech_id": speech_handle.id,
                        },
                    )
                    try:
                        await called_fnc.task
                    except Exception as e:
                        logger.exception(
                            "error executing ai function",
                            extra={
                                "function": fnc.function_info.name,
                                "speech_id": speech_handle.id,
                            },
                            exc_info=e,
                        )

                tool_calls_info = []
                tool_calls_results = []

                for called_fnc in called_fncs:
                    # ignore the function calls that returns None
                    if called_fnc.result is None and called_fnc.exception is None:
                        continue

                    tool_calls_info.append(called_fnc.call_info)
                    tool_calls_results.append(
                        ChatMessage.create_tool_from_called_function(called_fnc)
                    )

                if not tool_calls_info:
                    break

                # generate an answer from the tool calls
                extra_tools_messages.append(
                    ChatMessage.create_tool_calls(tool_calls_info, text=collected_text)
                )
                extra_tools_messages.extend(tool_calls_results)

                chat_ctx = speech_handle.source.chat_ctx.copy()
                chat_ctx.messages.extend(extra_tools_messages)

                answer_llm_stream = self._llm.chat(
                    chat_ctx=chat_ctx, fnc_ctx=self.fnc_ctx
                )
                answer_synthesis = self._synthesize_agent_speech(
                    speech_handle.id, answer_llm_stream
                )
                # replace the synthesis handle with the new one to allow interruption
                speech_handle.synthesis_handle = answer_synthesis
                play_handle = answer_synthesis.play()
                await play_handle.join()

                collected_text = answer_synthesis.tts_forwarder.played_text
                interrupted = answer_synthesis.interrupted
                new_function_calls = answer_llm_stream.function_calls

                self.emit("function_calls_finished", called_fncs)

                if not new_function_calls:
                    break

            _CallContextVar.reset(tk)

        if speech_handle.add_to_chat_ctx and (
            not user_question or speech_handle.user_committed
        ):
            self._chat_ctx.messages.extend(extra_tools_messages)

            if interrupted:
                collected_text += "..."

            msg = ChatMessage.create(text=collected_text, role="assistant")
            self._chat_ctx.messages.append(msg)

            speech_handle.mark_speech_committed()

            if interrupted:
                self.emit("agent_speech_interrupted", msg)
            else:
                self.emit("agent_speech_committed", msg)

            logger.debug(
                "committed agent speech",
                extra={
                    "agent_transcript": collected_text,
                    "interrupted": interrupted,
                    "speech_id": speech_handle.id,
                },
            )

    def _synthesize_agent_speech(
        self,
        speech_id: str,
        source: str | LLMStream | AsyncIterable[str],
    ) -> SynthesisHandle:
        assert (
            self._agent_output is not None
        ), "agent output should be initialized when ready"

        tk = SpeechDataContextVar.set(SpeechData(speech_id))

        async def _llm_stream_to_str_generator(
            stream: LLMStream,
        ) -> AsyncGenerator[str]:
            try:
                async for chunk in stream:
                    if not chunk.choices:
                        continue

                    content = chunk.choices[0].delta.content
                    if content is None:
                        continue

                    yield content
            finally:
                await stream.aclose()

        if isinstance(source, LLMStream):
            source = _llm_stream_to_str_generator(source)

        og_source = source
        transcript_source = source
        if isinstance(og_source, AsyncIterable):
            og_source, transcript_source = utils.aio.itertools.tee(og_source, 2)

        tts_source = self._opts.before_tts_cb(self, og_source)
        if tts_source is None:
            raise ValueError("before_tts_cb must return str or AsyncIterable[str]")

        try:
            return self._agent_output.synthesize(
                speech_id=speech_id,
                tts_source=tts_source,
                transcript_source=transcript_source,
                transcription=self._opts.transcription.agent_transcription,
                transcription_speed=self._opts.transcription.agent_transcription_speed,
                sentence_tokenizer=self._opts.transcription.sentence_tokenizer,
                word_tokenizer=self._opts.transcription.word_tokenizer,
                hyphenate_word=self._opts.transcription.hyphenate_word,
            )
        finally:
            SpeechDataContextVar.reset(tk)

    def _validate_reply_if_possible(self) -> None:
        """Check if the new agent speech should be played"""

        if self._playing_speech is not None:
            should_ignore_input = False
            if not self._playing_speech.allow_interruptions:
                should_ignore_input = True
                logger.debug(
                    "skipping validation, agent is speaking and does not allow interruptions",
                    extra={"speech_id": self._playing_speech.id},
                )
            elif not self._should_interrupt():
                should_ignore_input = True
                logger.debug(
                    "interrupt threshold is not met",
                    extra={"speech_id": self._playing_speech.id},
                )
            if should_ignore_input:
                self._transcribed_text = ""
                return

        if self._pending_agent_reply is None:
            if self._opts.preemptive_synthesis or not self._transcribed_text:
                return

            self._synthesize_agent_reply()

        assert self._pending_agent_reply is not None

        # in some bad timing, we could end up with two pushed agent replies inside the speech queue.
        # so make sure we directly interrupt every reply when validating a new one
        for speech in self._speech_q:
            if not speech.is_reply:
                continue

            if speech.allow_interruptions:
                speech.interrupt()

        logger.debug(
            "validated agent reply",
            extra={"speech_id": self._pending_agent_reply.id},
        )

        if self._last_speech_time is not None:
            time_since_last_speech = time.perf_counter() - self._last_speech_time
            transcription_delay = max(
                (self._last_final_transcript_time or 0) - self._last_speech_time, 0
            )

            eou_metrics = metrics.PipelineEOUMetrics(
                timestamp=time.time(),
                sequence_id=self._pending_agent_reply.id,
                end_of_utterance_delay=time_since_last_speech,
                transcription_delay=transcription_delay,
            )
            self.emit("metrics_collected", eou_metrics)

        self._add_speech_for_playout(self._pending_agent_reply)
        self._pending_agent_reply = None
        self._transcribed_interim_text = ""
        # self._transcribed_text is reset after MIN_TIME_PLAYED_FOR_COMMIT, see self._play_speech

    def _interrupt_if_possible(self) -> None:
        """Check whether the current assistant speech should be interrupted"""
        if self._playing_speech and self._should_interrupt():
            self._playing_speech.interrupt()

    def _should_interrupt(self) -> bool:
        if self._playing_speech is None:
            return True

        if (
            not self._playing_speech.allow_interruptions
            or self._playing_speech.interrupted
        ):
            return False

        if self._opts.int_min_words != 0:
            text = self._transcribed_interim_text or self._transcribed_text
            interim_words = self._opts.transcription.word_tokenizer.tokenize(text=text)
            if len(interim_words) < self._opts.int_min_words:
                return False

        return True

    def _add_speech_for_playout(self, speech_handle: SpeechHandle) -> None:
        self._speech_q.append(speech_handle)
        self._speech_q_changed.set()


class _DeferredReplyValidation:
    """This class is used to try to find the best time to validate the agent reply."""

    # if the STT gives us punctuation, we can try validate the reply faster.
    PUNCTUATION = ".!?"
    PUNCTUATION_REDUCE_FACTOR = 0.75

    LATE_TRANSCRIPT_TOLERANCE = 1.5  # late compared to end of speech

    def __init__(
        self,
        validate_fnc: Callable[[], None],
        min_endpointing_delay: float,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._validate_fnc = validate_fnc
        self._validating_task: asyncio.Task | None = None
        self._last_final_transcript: str = ""
        self._last_recv_end_of_speech_time: float = 0.0
        self._speaking = False

        self._end_of_speech_delay = min_endpointing_delay
        self._final_transcript_delay = min_endpointing_delay + 1.0

    @property
    def validating(self) -> bool:
        return self._validating_task is not None and not self._validating_task.done()

    def on_human_final_transcript(self, transcript: str) -> None:
        self._last_final_transcript = transcript.strip()  # type: ignore

        if self._speaking:
            return

        has_recent_end_of_speech = (
            time.time() - self._last_recv_end_of_speech_time
            < self.LATE_TRANSCRIPT_TOLERANCE
        )
        delay = (
            self._end_of_speech_delay
            if has_recent_end_of_speech
            else self._final_transcript_delay
        )
        delay = delay * (
            self.PUNCTUATION_REDUCE_FACTOR if self._end_with_punctuation() else 1.0
        )

        self._run(delay)

    def on_human_start_of_speech(self, ev: vad.VADEvent) -> None:
        self._speaking = True
        if self.validating:
            assert self._validating_task is not None
            self._validating_task.cancel()

    def on_human_end_of_speech(self, ev: vad.VADEvent) -> None:
        self._speaking = False
        self._last_recv_end_of_speech_time = time.time()

        if self._last_final_transcript:
            delay = self._end_of_speech_delay * (
                self.PUNCTUATION_REDUCE_FACTOR if self._end_with_punctuation() else 1.0
            )
            self._run(delay)

    async def aclose(self) -> None:
        if self._validating_task is not None:
            await utils.aio.gracefully_cancel(self._validating_task)

    def _end_with_punctuation(self) -> bool:
        return (
            len(self._last_final_transcript) > 0
            and self._last_final_transcript[-1] in self.PUNCTUATION
        )

    def _reset_states(self) -> None:
        self._last_final_transcript = ""
        self._last_recv_end_of_speech_time = 0.0

    def _run(self, delay: float) -> None:
        @utils.log_exceptions(logger=logger)
        async def _run_task(delay: float) -> None:
            await asyncio.sleep(delay)
            self._reset_states()
            self._validate_fnc()

        if self._validating_task is not None:
            self._validating_task.cancel()

        self._validating_task = asyncio.create_task(_run_task(delay))
