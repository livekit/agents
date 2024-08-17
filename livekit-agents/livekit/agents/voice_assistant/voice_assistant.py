from __future__ import annotations

import asyncio
import contextvars
import time
from dataclasses import dataclass
from typing import Any, AsyncIterable, Awaitable, Callable, Literal, Optional, Union

from livekit import rtc

from .. import stt, tokenize, tts, utils, vad
from ..llm import LLM, ChatContext, ChatMessage, FunctionContext, LLMStream
from .agent_output import AgentOutput, SynthesisHandle
from .agent_playout import AgentPlayout
from .human_input import HumanInput
from .log import logger
from .plotter import AssistantPlotter
from .speech_handle import SpeechHandle

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
    This is used to decide when to mark a transcript as final for the agent transcription."""
    word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer()
    """The tokenizer used to split the speech into words.
    This is used to simulate the "interim results" of the agent transcription."""
    hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word
    """A function that takes a string (word) as input and returns a list of strings,
    representing the hyphenated parts of the word."""


class VoiceAssistant(utils.EventEmitter[EventTypes]):
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
        preemptive_synthesis: bool = True,
        transcription: AssistantTranscriptionOptions = AssistantTranscriptionOptions(),
        will_synthesize_assistant_reply: WillSynthesizeAssistantReply = _default_will_synthesize_assistant_reply,
        plotting: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """
        Create a new VoiceAssistant.

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
            preemptive_synthesis: Whether to preemptively synthesize responses.
            transcription: Options for assistant transcription.
            will_synthesize_assistant_reply: Callback called when the assistant is about to synthesize a reply.
                This can be used to customize the reply (e.g: inject context/RAG).
            plotting: Whether to enable plotting for debugging. matplotlib must be installed.
            loop: Event loop to use. Default to asyncio.get_event_loop().
        """
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
            self._validate_reply_if_possible, loop=self._loop
        )

        self._speech_q: list[SpeechHandle] = []
        self._speech_q_changed = asyncio.Event()

        self._last_end_of_speech_time: float | None = None

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
                self._agent_output.playout.target_volume = tv

            smoothed_tv = self._agent_output.playout.smoothed_volume

            self._plotter.plot_value("raw_vol", tv)
            self._plotter.plot_value("smoothed_vol", smoothed_tv)
            self._plotter.plot_value("vad_probability", ev.probability)

            if ev.speech_duration >= self._opts.int_speech_duration:
                self._interrupt_if_possible()

        def _on_end_of_speech(ev: vad.VADEvent) -> None:
            self._plotter.plot_event("user_stopped_speaking")
            self.emit("user_stopped_speaking")
            self._deferred_validation.on_human_end_of_speech(ev)
            self._last_end_of_speech_time = time.time()

        def _on_interim_transcript(ev: stt.SpeechEvent) -> None:
            self._transcribed_interim_text = ev.alternatives[0].text

        def _on_final_transcript(ev: stt.SpeechEvent) -> None:
            new_transcript = ev.alternatives[0].text
            self._transcribed_text += (
                " " if self._transcribed_text else ""
            ) + new_transcript

            if self._opts.preemptive_synthesis:
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
        self._human_input.on("vad_inference_done", _on_vad_updated)
        self._human_input.on("end_of_speech", _on_end_of_speech)
        self._human_input.on("interim_transcript", _on_interim_transcript)
        self._human_input.on("final_transcript", _on_final_transcript)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        if self._opts.plotting:
            await self._plotter.start()

        audio_source = rtc.AudioSource(self._tts.sample_rate, self._tts.num_channels)
        track = rtc.LocalAudioTrack.create_audio_track("assistant_voice", audio_source)
        self._agent_publication = await self._room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )

        agent_playout = AgentPlayout(source=audio_source)
        self._agent_output = AgentOutput(
            room=self._room,
            agent_playout=agent_playout,
            llm=self._llm,
            tts=self._tts,
        )

        def _on_playout_started() -> None:
            self._plotter.plot_event("agent_started_speaking")
            self.emit("agent_started_speaking")

        def _on_playout_stopped(interrupted: bool) -> None:
            self._plotter.plot_event("agent_stopped_speaking")
            self.emit("agent_stopped_speaking")

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

    def _synthesize_agent_reply(self) -> None:
        """Synthesize the agent reply to the user question, also make sure only one reply
        is synthesized/played at a time"""

        if self._pending_agent_reply is not None:
            self._pending_agent_reply.interrupt()

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

        user_msg = ChatMessage.create(text=handle.user_question, role="user")
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

        synthesis_handle = self._synthesize_agent_speech(handle.id, llm_stream)
        handle.initialize(source=llm_stream, synthesis_handle=synthesis_handle)

        # TODO(theomonnom): Find a more reliable way to get the elapsed time from the last end of speech
        # (VAD could not have detected any speech - maybe unlikely?)
        if self._last_end_of_speech_time is not None:
            elapsed = round(time.time() - self._last_end_of_speech_time, 3)
        else:
            elapsed = -1.0

        logger.debug(
            "synthesizing agent reply",
            extra={
                "user_transcript": handle.user_question,
                "speech_id": handle.id,
                "elapsed": elapsed,
            },
        )

    async def _play_speech(self, speech_handle: SpeechHandle) -> None:
        try:
            await speech_handle.wait_for_initialization()
        except asyncio.CancelledError:
            return

        synthesis_handle = speech_handle.synthesis_handle
        if synthesis_handle.interrupted:
            return

        user_question = speech_handle.user_question
        user_speech_committed = False

        play_handle = synthesis_handle.play()
        join_fut = play_handle.join()

        def _commit_user_question_if_needed() -> None:
            nonlocal user_speech_committed

            if (
                not user_question
                or synthesis_handle.interrupted
                or user_speech_committed
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

            logger.debug(
                "committed user transcript", extra={"user_transcript": user_question}
            )
            user_msg = ChatMessage.create(text=user_question, role="user")
            self._chat_ctx.messages.append(user_msg)
            self.emit("user_speech_committed", user_msg)

            self._transcribed_text = self._transcribed_text[len(user_question) :]
            user_speech_committed = True

        # wait for the play_handle to finish and check every 1s if the user question should be committed
        _commit_user_question_if_needed()

        while not join_fut.done():
            await asyncio.wait(
                [join_fut], return_when=asyncio.FIRST_COMPLETED, timeout=0.5
            )

            _commit_user_question_if_needed()

        _commit_user_question_if_needed()

        collected_text = speech_handle.synthesis_handle.tts_forwarder.played_text
        interrupted = speech_handle.synthesis_handle.interrupted
        is_using_tools = isinstance(speech_handle.source, LLMStream) and len(
            speech_handle.source.function_calls
        )

        extra_tools_messages = []  # additional messages from the functions to add to the context if needed

        # if the answer is using tools, execute the functions and automatically generate
        # a response to the user question from the returned values
        if is_using_tools and not interrupted:
            assert isinstance(speech_handle.source, LLMStream)
            assert (
                not user_question or user_speech_committed
            ), "user speech should have been committed before using tools"

            # execute functions
            call_ctx = AssistantCallContext(self, speech_handle.source)
            tk = _CallContextVar.set(call_ctx)
            self.emit("function_calls_collected", speech_handle.source.function_calls)
            called_fncs_info = speech_handle.source.function_calls

            called_fncs = []
            for fnc in called_fncs_info:
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
                except Exception:
                    pass

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

                chat_ctx = speech_handle.source.chat_ctx.copy()
                chat_ctx.messages.extend(extra_tools_messages)

                answer_llm_stream = self._llm.chat(
                    chat_ctx=chat_ctx, fnc_ctx=self._fnc_ctx
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

        if speech_handle.add_to_chat_ctx and (
            not user_question or user_speech_committed
        ):
            self._chat_ctx.messages.extend(extra_tools_messages)

            if interrupted:
                collected_text += "..."

            msg = ChatMessage.create(text=collected_text, role="assistant")
            self._chat_ctx.messages.append(msg)

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

        if isinstance(source, LLMStream):
            source = _llm_stream_to_str_iterable(speech_id, source)

        return self._agent_output.synthesize(
            speech_id=speech_id,
            transcript=source,
            transcription=self._opts.transcription.agent_transcription,
            transcription_speed=self._opts.transcription.agent_transcription_speed,
            sentence_tokenizer=self._opts.transcription.sentence_tokenizer,
            word_tokenizer=self._opts.transcription.word_tokenizer,
            hyphenate_word=self._opts.transcription.hyphenate_word,
        )

    def _validate_reply_if_possible(self) -> None:
        """Check if the new agent speech should be played"""

        if self._pending_agent_reply is None:
            if self._opts.preemptive_synthesis or not self._transcribed_text:
                return

            self._synthesize_agent_reply()  # this will populate self._pending_agent_reply
        else:
            # in some timing, we could end up with two pushed agent replies inside the speech queue.
            # so make sure we directly interrupt every reply when pushing a new one
            for speech in self._speech_q:
                if speech.allow_interruptions and speech.is_reply:
                    speech.interrupt()

        assert self._pending_agent_reply is not None

        logger.debug(
            "validated agent reply",
            extra={"speech_id": self._pending_agent_reply.id},
        )

        self._add_speech_for_playout(self._pending_agent_reply)
        self._pending_agent_reply = None
        self._transcribed_interim_text = ""
        # self._transcribed_text is reset after MIN_TIME_PLAYED_FOR_COMMIT, see self._play_speech

    def _interrupt_if_possible(self) -> None:
        """Check whether the current assistant speech should be interrupted"""
        if (
            self._playing_speech is None
            or not self._playing_speech.allow_interruptions
            or self._playing_speech.synthesis_handle.interrupted
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

        self._playing_speech.interrupt()

    def _add_speech_for_playout(self, speech_handle: SpeechHandle) -> None:
        self._speech_q.append(speech_handle)
        self._speech_q_changed.set()


async def _llm_stream_to_str_iterable(
    speech_id: str, stream: LLMStream
) -> AsyncIterable[str]:
    start_time = time.time()
    first_frame = True
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is None:
            continue

        if first_frame:
            first_frame = False
            logger.debug(
                "first LLM token",
                extra={
                    "speech_id": speech_id,
                    "elapsed": round(time.time() - start_time, 3),
                },
            )

        yield content


class _DeferredReplyValidation:
    """This class is used to try to find the best time to validate the agent reply."""

    # if the STT gives us punctuation, we can try validate the reply faster.
    PUNCTUATION = ".!?"
    PUNCTUATION_REDUCE_FACTOR = 0.5

    DEFER_DELAY_END_OF_SPEECH = 0.2
    DEFER_DELAY_FINAL_TRANSCRIPT = 1.0
    LATE_TRANSCRIPT_TOLERANCE = 1.5  # late compared to end of speech

    def __init__(
        self, validate_fnc: Callable[[], None], loop: asyncio.AbstractEventLoop
    ) -> None:
        self._validate_fnc = validate_fnc
        self._validating_task: asyncio.Task | None = None
        self._last_final_transcript: str = ""
        self._last_recv_end_of_speech_time: float = 0.0
        self._speaking = False

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
            self.DEFER_DELAY_END_OF_SPEECH
            if has_recent_end_of_speech
            else self.DEFER_DELAY_FINAL_TRANSCRIPT
        )
        delay = (
            delay * self.PUNCTUATION_REDUCE_FACTOR
            if self._end_with_punctuation()
            else 1.0
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
            delay = (
                self.DEFER_DELAY_END_OF_SPEECH * self.PUNCTUATION_REDUCE_FACTOR
                if self._end_with_punctuation()
                else 1.0
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
