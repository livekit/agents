from __future__ import annotations

import asyncio
import contextvars
import time
from dataclasses import dataclass
from typing import Any, AsyncIterable, Awaitable, Callable, Literal, Optional, Union

from livekit import rtc

from .. import stt, tokenize, tts, utils, vad
from ..llm import LLM, ChatContext, ChatMessage, FunctionContext, LLMStream
from .agent_output import AgentOutput, SpeechSource, SynthesisHandle
from .cancellable_source import CancellableAudioSource
from .human_input import HumanInput
from .log import logger
from .plotter import AssistantPlotter


@dataclass
class _SpeechInfo:
    source: str | LLMStream | AsyncIterable[str]
    user_question: str  # empty when the speech isn't an answer
    allow_interruptions: bool
    add_to_chat_ctx: bool
    synthesis_handle: SynthesisHandle


WillSynthesizeAssistantReply = Callable[
    ["VoiceAssistant", ChatContext],
    Union[Optional[LLMStream], Awaitable[Optional[LLMStream]]],
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
]


_CallContextVar = contextvars.ContextVar["AssistantCallContext"](
    "voice_assistant_contextvar"
)


class AssistantCallContext:
    def __init__(self, assistant: "VoiceAssistant", llm_stream: LLMStream) -> None:
        self._assistant = assistant
        self._metadata = dict[str, Any]()
        self._llm_stream = llm_stream

    @staticmethod
    def get_current() -> "AssistantCallContext":
        return _CallContextVar.get()

    @property
    def assistant(self) -> "VoiceAssistant":
        return self._assistant

    def store_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    def llm_stream(self) -> LLMStream:
        return self._llm_stream


def _default_will_synthesize_assistant_reply(
    assistant: VoiceAssistant, chat_ctx: ChatContext
) -> LLMStream:
    return assistant.llm.chat(chat_ctx=chat_ctx, fnc_ctx=assistant.fnc_ctx)


@dataclass(frozen=True)
class _ImplOptions:
    allow_interruptions: bool
    int_speech_duration: float
    int_min_words: int
    preemptive_synthesis: bool
    will_synthesize_assistant_reply: WillSynthesizeAssistantReply
    plotting: bool
    transcription: AssistantTranscriptionOptions


@dataclass(frozen=True)
class AssistantTranscriptionOptions:
    user_transcription: bool = True
    """Whether to forward the user transcription to the client"""
    agent_transcription: bool = True
    """Whether to forward the agent transcription to the client"""
    agent_transcription_speed: float = 1.0
    """The speed at which the agent's speech transcription is forwarded to the client.
    We try to mimic the agent's speech speed by adjusting the transcription speed."""
    sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer()
    """The tokenizer used to split the speech into sentences. 
    This is used to device when to mark a transcript as final for the agent transcription."""
    word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer()
    """The tokenizer used to split the speech into words.
    This is used to simulate the "interim results" of the agent transcription."""
    hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word
    """A function that takes a string (word) as input and returns a list of strings,
    representing the hyphenated parts of the word."""


class VoiceAssistant(utils.EventEmitter[EventTypes]):
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
        interrupt_speech_duration: float = 0.6,
        interrupt_min_words: int = 0,
        preemptive_synthesis: bool = True,
        transcription: AssistantTranscriptionOptions = AssistantTranscriptionOptions(),
        will_synthesize_assistant_reply: WillSynthesizeAssistantReply = _default_will_synthesize_assistant_reply,
        plotting: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__()
        self._loop = loop or asyncio.get_event_loop()
        self._opts = _ImplOptions(
            plotting=plotting,
            allow_interruptions=allow_interruptions,
            int_speech_duration=interrupt_speech_duration,
            int_min_words=interrupt_min_words,
            preemptive_synthesis=preemptive_synthesis,
            transcription=transcription,
            will_synthesize_assistant_reply=will_synthesize_assistant_reply,
        )
        self._plotter = AssistantPlotter(self._loop)

        # wrap with StreamAdapter automatically when streaming is not supported on a specific TTS
        # to override StreamAdapter options, create the adapter manually

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
        self._track_published_fut = asyncio.Future[None]()

        self._agent_answer_speech: _SpeechInfo | None = None
        self._agent_playing_speech: _SpeechInfo | None = None
        self._agent_answer_atask: asyncio.Task[None] | None = None
        self._playout_ch = utils.aio.Chan[_SpeechInfo]()

        self._transcribed_text, self._transcribed_interim_text = "", ""

        self._deferred_validation = _DeferredAnswerValidation(
            self._validate_answer_if_needed, loop=self._loop
        )

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

        room.on("participant_connected", self._on_participant_connected)
        self._room, self._participant = room, participant

        if participant is not None:
            if isinstance(participant, rtc.RemoteParticipant):
                self._link_participant(participant.identity)
            else:
                self._link_participant(participant)
        else:
            # no participant provided, try to find the first in the room
            for participant in self._room.remote_participants.values():
                self._link_participant(participant.identity)
                break

        self._main_atask = asyncio.create_task(self._main_task())

    async def say(
        self,
        source: str | LLMStream | AsyncIterable[str],
        *,
        allow_interruptions: bool = True,
        add_to_chat_ctx: bool = True,
    ) -> None:
        """
        Make the assistant say something.
        The source can be a string, an LLMStream or an AsyncIterable[str]

        Args:
            source: the source of the speech
            allow_interruptions: whether the speech can be interrupted
            add_to_chat_ctx: whether to add the speech to the chat context
        """
        await self._track_published_fut
        assert (
            self._agent_output is not None
        ), "agent output should be initialized when ready"

        speech_source = source
        if isinstance(speech_source, LLMStream):
            speech_source = _llm_stream_to_str_iterable(speech_source)

        synthesis_handle = self._agent_synthesize(transcript=speech_source)
        speech = _SpeechInfo(
            source=source,
            user_question="",
            allow_interruptions=allow_interruptions,
            add_to_chat_ctx=add_to_chat_ctx,
            synthesis_handle=synthesis_handle,
        )
        self._playout_ch.send_nowait(speech)

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

        def _on_vad_updated(ev: vad.VADEvent) -> None:
            if not self._track_published_fut.done():
                return

            assert self._agent_output is not None

            tv = 1.0
            if self._opts.allow_interruptions:
                tv = max(0.0, 1.0 - ev.probability)
                self._agent_output.audio_source.target_volume = tv

            smoothed_tv = self._agent_output.audio_source.smoothed_volume

            self._plotter.plot_value("raw_vol", tv)
            self._plotter.plot_value("smoothed_vol", smoothed_tv)
            self._plotter.plot_value("vad_probability", ev.probability)

            if ev.speech_duration >= self._opts.int_speech_duration:
                self._interrupt_if_needed()

        def _on_end_of_speech(ev: vad.VADEvent) -> None:
            self._plotter.plot_event("user_stopped_speaking")
            self.emit("user_stopped_speaking")
            self._deferred_validation.on_human_end_of_speech(ev)

        def _on_interim_transcript(ev: stt.SpeechEvent) -> None:
            self._transcribed_interim_text = ev.alternatives[0].text

        def _on_final_transcript(ev: stt.SpeechEvent) -> None:
            self._transcribed_text += ev.alternatives[0].text

            if self._opts.preemptive_synthesis:
                self._synthesize_answer(
                    user_transcript=self._transcribed_text, force_play=False
                )

        self._human_input.on("start_of_speech", _on_start_of_speech)
        self._human_input.on("vad_inference_done", _on_vad_updated)
        self._human_input.on("end_of_speech", _on_end_of_speech)
        self._human_input.on("interim_transcript", _on_interim_transcript)
        self._human_input.on("final_transcript", _on_final_transcript)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        if self._opts.plotting:
            self._plotter.start()

        audio_source = rtc.AudioSource(self._tts.sample_rate, self._tts.num_channels)
        track = rtc.LocalAudioTrack.create_audio_track("assistant_voice", audio_source)
        self._agent_publication = await self._room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )

        cancellable_audio_source = CancellableAudioSource(source=audio_source)
        self._agent_output = AgentOutput(
            room=self._room,
            source=cancellable_audio_source,
            llm=self._llm,
            tts=self._tts,
        )

        def _on_playout_started() -> None:
            self._plotter.plot_event("agent_started_speaking")
            self.emit("agent_started_speaking")

        def _on_playout_stopped(cancelled: bool) -> None:
            self._plotter.plot_event("agent_stopped_speaking")
            self.emit("agent_stopped_speaking")

        cancellable_audio_source.on("playout_started", _on_playout_started)
        cancellable_audio_source.on("playout_stopped", _on_playout_stopped)

        self._track_published_fut.set_result(None)

        # play validated speeches
        async for speech in self._playout_ch:
            self._agent_playing_speech = speech
            await self._play_speech(speech)
            self._agent_playing_speech = None

    def _validate_answer_if_needed(self) -> None:
        """
        Check if the user speech should be validated/played
        """
        if (
            self._agent_answer_speech is not None
            and not self._agent_answer_speech.synthesis_handle.interrupted
        ):
            self._playout_ch.send_nowait(self._agent_answer_speech)
            self._agent_answer_speech = None
        elif not self._opts.preemptive_synthesis and self._transcribed_text:
            self._synthesize_answer(
                user_transcript=self._transcribed_text, force_play=True
            )

    def _interrupt_if_needed(self) -> None:
        """
        Check whether the current assistant speech should be interrupted
        """
        if (
            self._agent_playing_speech is None
            or not self._agent_playing_speech.allow_interruptions
            or self._agent_playing_speech.synthesis_handle.interrupted
        ):
            return

        if self._opts.int_min_words != 0:
            # check the final/interim transcribed text for the minimum word count
            # to interrupt the agent speech
            interim_words = self._opts.transcription.word_tokenizer.tokenize(
                text=self._transcribed_interim_text
            )
            if len(interim_words) < self._opts.int_min_words:
                return

        self._agent_playing_speech.synthesis_handle.interrupt()

    def _synthesize_answer(self, *, user_transcript: str, force_play: bool) -> None:
        """
        Synthesize the answer to the user question and make sure
        only one answer is synthesized at a time
        """

        @utils.log_exceptions(logger=logger)
        async def _synthesize_answer_task(old_task: asyncio.Task[None]) -> None:
            # Use an async task to synthesize the agent answer to
            # allow users to execute async code inside the will_create_llm_stream callback
            assert (
                self._agent_output is not None
            ), "agent output should be initialized when ready"

            if old_task is not None:
                await utils.aio.gracefully_cancel(old_task)

            user_msg = ChatMessage.create(text=user_transcript, role="user")
            copied_ctx = self._chat_ctx.copy()
            copied_ctx.messages.append(user_msg)

            llm_stream = self._opts.will_synthesize_assistant_reply(self, copied_ctx)
            if asyncio.iscoroutine(llm_stream):
                llm_stream = await llm_stream

            # fallback to default impl if no custom/user stream is returned
            if not isinstance(llm_stream, LLMStream):
                llm_stream = _default_will_synthesize_assistant_reply(
                    self, chat_ctx=copied_ctx
                )

            synthesis = self._agent_synthesize(
                transcript=_llm_stream_to_str_iterable(llm_stream)
            )
            self._agent_answer_speech = _SpeechInfo(
                source=llm_stream,
                user_question=user_transcript,
                allow_interruptions=self._opts.allow_interruptions,
                add_to_chat_ctx=True,
                synthesis_handle=synthesis,
            )
            self._deferred_validation.on_new_synthesis(user_transcript)

            if force_play:
                self._playout_ch.send_nowait(self._agent_answer_speech)

        if self._agent_answer_speech is not None:
            self._agent_answer_speech.synthesis_handle.interrupt()

        self._agent_answer_speech = None
        old_task = self._agent_answer_atask

        self._agent_answer_atask = asyncio.create_task(
            _synthesize_answer_task(old_task)
        )

    async def _play_speech(self, speech_info: _SpeechInfo) -> None:
        logger.debug("VoiceAssistant._play_speech started")

        assert self._agent_playing_speech is not None

        MIN_TIME_PLAYED_FOR_COMMIT = 1.5

        assert (
            self._agent_output is not None
        ), "agent output should be initialized when ready"

        synthesis_handle = speech_info.synthesis_handle
        if synthesis_handle.interrupted:
            return

        user_question = speech_info.user_question
        user_speech_commited = False

        play_handle = synthesis_handle.play()
        join_fut = play_handle.join()

        def _commit_user_message_if_needed() -> None:
            nonlocal user_speech_commited

            if (
                not user_question
                or synthesis_handle.interrupted
                or user_speech_commited
            ):
                return

            is_using_tools = isinstance(speech_info.source, LLMStream) and len(
                speech_info.source.function_calls
            )

            # make sure at least some speech was played before committing the user message
            # since we try to validate as fast as possible it is possible the agent gets interrupted
            # really quickly (barely audible), we don't want to mark this question as "answered".
            if not is_using_tools and (
                play_handle.time_played < MIN_TIME_PLAYED_FOR_COMMIT
                and not join_fut.done()
            ):
                return

            user_msg = ChatMessage.create(text=user_question, role="user")
            self._chat_ctx.messages.append(user_msg)
            self.emit("user_speech_committed", user_msg)

            self._transcribed_text = self._transcribed_text[len(user_question) :]
            user_speech_commited = True

        # wait for the play_handle to finish and check every 1s if the user question should be committed
        while not join_fut.done():
            await asyncio.wait(
                [join_fut], return_when=asyncio.FIRST_COMPLETED, timeout=1.0
            )

            _commit_user_message_if_needed()

        _commit_user_message_if_needed()

        collected_text = speech_info.synthesis_handle.collected_text
        interrupted = speech_info.synthesis_handle.interrupted
        is_using_tools = isinstance(speech_info.source, LLMStream) and len(
            speech_info.source.function_calls
        )

        extra_tools_messages = []  # additional messages from the functions to add to the context if needed

        # if the answer is using tools, execute the functions and automatically generate
        # a response to the user question from the returned values
        if is_using_tools and not interrupted:
            assert isinstance(speech_info.source, LLMStream)
            assert (
                user_speech_commited
            ), "user speech should be committed before using tools"

            # execute functions
            call_ctx = AssistantCallContext(self, speech_info.source)
            tk = _CallContextVar.set(call_ctx)
            self.emit("function_calls_collected", speech_info.source.function_calls)
            called_fncs = speech_info.source.execute_functions()
            tasks = [called_fnc.task for called_fnc in called_fncs]
            await asyncio.gather(*tasks, return_exceptions=True)
            self.emit("function_calls_finished", called_fncs)
            _CallContextVar.reset(tk)

            tool_calls = []
            tool_calls_results_msg = []

            for called_fnc in called_fncs:
                # ignore the function calls that returns None
                if called_fnc.result is None:
                    continue

                tool_calls.append(called_fnc.call_info)
                tool_calls_results_msg.append(
                    ChatMessage.create_tool_from_called_function(called_fnc)
                )

            if tool_calls:
                extra_tools_messages.append(ChatMessage.create_tool_calls(tool_calls))
                extra_tools_messages.extend(tool_calls_results_msg)

                chat_ctx = speech_info.source.chat_ctx.copy()
                chat_ctx.messages.extend(extra_tools_messages)

                answer_stream = self._llm.chat(chat_ctx=chat_ctx, fnc_ctx=self._fnc_ctx)
                answer_synthesis = self._agent_synthesize(
                    transcript=_llm_stream_to_str_iterable(answer_stream)
                )
                # make sure users can interrupt the fnc calls answer
                # TODO(theomonnom): maybe we should add a new fnc_call_answer field to _SpeechInfo?
                self._agent_playing_speech.synthesis_handle = answer_synthesis
                play_handle = answer_synthesis.play()
                await play_handle.join()

                collected_text = answer_synthesis.collected_text
                interrupted = answer_synthesis.interrupted

        if speech_info.add_to_chat_ctx and (not user_question or user_speech_commited):
            self._chat_ctx.messages.extend(extra_tools_messages)

            msg = ChatMessage.create(text=collected_text, role="assistant")
            self._chat_ctx.messages.append(msg)

            if interrupted:
                self.emit("agent_speech_interrupted", msg)
            else:
                self.emit("agent_speech_committed", msg)

        logger.debug("VoiceAssistant._play_speech ended")

    def _agent_synthesize(self, *, transcript: SpeechSource) -> SynthesisHandle:
        assert (
            self._agent_output is not None
        ), "agent output should be initialized when ready"

        return self._agent_output.synthesize(
            transcript=transcript,
            transcription=self._opts.transcription.agent_transcription,
            transcription_speed=self._opts.transcription.agent_transcription_speed,
            sentence_tokenizer=self._opts.transcription.sentence_tokenizer,
            word_tokenizer=self._opts.transcription.word_tokenizer,
            hyphenate_word=self._opts.transcription.hyphenate_word,
        )


async def _llm_stream_to_str_iterable(stream: LLMStream) -> AsyncIterable[str]:
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is None:
            continue

        yield content


class _DeferredAnswerValidation:
    # if the STT gives us punctuation, we can validate faster, we can be more confident
    # about the end of the sentence (naive way to increase the default DEFER_DELAY to allow the user
    # to say longer sentences without being interrupted by the assistant)
    PUNCTUATION = ".!?"
    DEFER_DELAY_WITH_PUNCTUATION = 0.15
    DEFER_DELAY = 0.2
    LATE_TRANSCRIPT_TOLERANCE = 5

    def __init__(
        self, validate_fnc: Callable[[], None], loop: asyncio.AbstractEventLoop
    ) -> None:
        self._validate_fnc = validate_fnc
        self._tasks_set = utils.aio.TaskSet(loop=loop)

        self._validating_task: asyncio.Task | None = None
        self._last_final_transcript: str = ""
        self._last_recv_end_of_speech_time: float = 0.0

    @property
    def validating(self) -> bool:
        return self._validating_task is not None and not self._validating_task.done()

    def _get_defer_delay(self) -> float:
        if (
            self._last_final_transcript
            and self._last_final_transcript[-1] in self.PUNCTUATION
        ):
            return self.DEFER_DELAY_WITH_PUNCTUATION

        return self.DEFER_DELAY

    def _reset_states(self) -> None:
        self._last_final_transcript = ""
        self._last_recv_end_of_speech_time = 0.0

    def on_new_synthesis(self, user_msg: str) -> None:
        self._last_final_transcript = user_msg.strip()  # type: ignore

        if self.validating:
            self._run(self._get_defer_delay())  # debounce
        elif (
            self._last_recv_end_of_speech_time
            and time.time() - self._last_recv_end_of_speech_time
            < self.LATE_TRANSCRIPT_TOLERANCE
        ):
            # final transcript was received after human stopped speaking
            self._run(self._get_defer_delay())

    def on_human_start_of_speech(self, ev: vad.VADEvent) -> None:
        if self.validating:
            assert self._validating_task is not None
            self._validating_task.cancel()

    def on_human_end_of_speech(self, ev: vad.VADEvent) -> None:
        self._last_recv_end_of_speech_time = time.time()

        if self._last_final_transcript:
            self._run(self._get_defer_delay())

    async def aclose(self) -> None:
        if self._validating_task is not None:
            self._validating_task.cancel()

        await self._tasks_set.aclose()

    @utils.log_exceptions(logger=logger)
    async def _run_task(self, delay: float) -> None:
        await asyncio.sleep(delay)
        self._last_final_transcript = ""
        self._received_end_of_speech = False
        self._validate_fnc()
        logger.debug("_DeferredAnswerValidation speech validated")

    def _run(self, delay: float) -> None:
        if self._validating_task is not None:
            self._validating_task.cancel()

        self._validating = self._tasks_set.create_task(self._run_task(delay))
