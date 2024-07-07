from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any, AsyncIterable, Awaitable, Callable, Literal, Optional, Union

from livekit import rtc

from .. import llm, stt, tokenize, tts, utils, vad
from .agent_output import AgentOutput, SynthesisHandle
from .cancellable_source import CancellableAudioSource
from .human_input import HumanInput
from .log import logger
from .plotter import AssistantPlotter


@dataclass
class _SpeechInfo:
    source: str | llm.LLMStream | AsyncIterable[str]
    allow_interruptions: bool
    add_to_chat_ctx: bool
    synthesis_handle: SynthesisHandle


WillSynthesizeAssistantReply = Callable[
    ["VoiceAssistant", llm.ChatContext],
    Union[Optional[llm.LLMStream], Awaitable[Optional[llm.LLMStream]]],
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


def _default_will_synthesize_assistant_reply(
    assistant: VoiceAssistant, chat_ctx: llm.ChatContext
) -> llm.LLMStream:
    return assistant.llm.chat(chat_ctx=chat_ctx, fnc_ctx=assistant.fnc_ctx)


@dataclass(frozen=True)
class _ImplOptions:
    allow_interruptions: bool
    int_speech_duration: float
    int_min_words: int
    preemptive_synthesis: bool
    will_synthesize_assistant_reply: WillSynthesizeAssistantReply
    plotting: bool

    # transcription & transcript analysis
    transcription: bool
    word_tokenizer: tokenize.WordTokenizer
    sentence_tokenizer: tokenize.SentenceTokenizer
    hyphenate_word: Callable[[str], list[str]]
    transcription_speed: float


class VoiceAssistant(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        vad: vad.VAD,
        stt: stt.STT,
        llm: llm.LLM,
        tts: tts.TTS,
        chat_ctx: llm.ChatContext = llm.ChatContext(),
        fnc_ctx: llm.FunctionContext | None = None,
        allow_interruptions: bool = True,
        interrupt_speech_duration: float = 0.65,
        interrupt_min_words: int = 2,
        preemptive_synthesis: bool = True,
        transcription: bool = True,
        will_synthesize_assistant_reply: WillSynthesizeAssistantReply = _default_will_synthesize_assistant_reply,
        sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(),
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(),
        hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word,
        transcription_speed: float = 3.83,
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
            sentence_tokenizer=sentence_tokenizer,
            word_tokenizer=word_tokenizer,
            hyphenate_word=hyphenate_word,
            transcription_speed=transcription_speed,
            will_synthesize_assistant_reply=will_synthesize_assistant_reply,
        )
        self._plotter = AssistantPlotter(self._loop)

        # wrap with StreamAdapter automatically when streaming is not supported on a specific TTS
        # to override StreamAdapter options, create the adapter manually
        if not tts.streaming_supported:
            from .. import tts as text_to_speech

            tts = text_to_speech.StreamAdapter(
                tts=tts, sentence_tokenizer=tokenize.basic.SentenceTokenizer()
            )

        self._stt, self._vad, self._llm, self._tts = stt, vad, llm, tts
        self._chat_ctx, self._fnc_ctx = chat_ctx, fnc_ctx
        self._started, self._closed = False, False

        self._human_input: HumanInput | None = None
        self._agent_output: AgentOutput | None = None
        self._track_published_fut = asyncio.Future()

        self._agent_answer_speech: _SpeechInfo | None = None
        self._agent_playing_speech: _SpeechInfo | None = None
        self._agent_answer_atask: asyncio.Task[None] | None = None
        self._playout_ch = utils.aio.Chan[_SpeechInfo]()

        self._transcribed_text, self._transcribed_interim_text = "", ""

        self._deferred_validation = _DeferredAnswerValidation(
            self._validate_answer_if_needed, loop=self._loop
        )

    @property
    def fnc_ctx(self) -> llm.FunctionContext | None:
        return self._fnc_ctx

    @fnc_ctx.setter
    def fnc_ctx(self, fnc_ctx: llm.FunctionContext | None) -> None:
        self._fnc_ctx = fnc_ctx

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def llm(self) -> llm.LLM:
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
            for participant in self._room.participants.values():
                self._link_participant(participant.identity)
                break

        self._main_atask = asyncio.create_task(self._main_task())

    async def say(
        self,
        source: str | llm.LLMStream | AsyncIterable[str],
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
        if isinstance(speech_source, llm.LLMStream):
            speech_source = _llm_stream_to_str_iterable(speech_source)

        synthesis_handle = self._agent_output.synthesize(transcript=speech_source)
        speech = _SpeechInfo(
            source=source,
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
        """
        Close the voice assistant
        """
        if not self._started:
            return

        self._room.off("participant_connected", self._on_participant_connected)
        await self._deferred_validation.aclose()

    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        if self._human_input is not None:
            return

        self._link_participant(participant.identity)

    def _link_participant(self, identity: str) -> None:
        participant = self._room.participants_by_identity.get(identity)
        if participant is None:
            logger.error("_link_participant must be called with a valid identity")
            return

        self._human_input = HumanInput(
            room=self._room,
            vad=self._vad,
            stt=self._stt,
            participant=participant,
        )

        def _on_start_of_speech(ev: vad.VADEvent) -> None:
            self._plotter.plot_event("user_started_speaking")
            self.emit("user_started_speaking")
            self._deferred_validation.on_human_start_of_speech(ev)

        def _on_vad_updated(ev: vad.VADEvent) -> None:
            if not self._track_published_fut.done():
                return

            assert self._agent_output is not None
            tv = max(0, 1 - ev.probability)
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
            self._synthesize_answer(user_transcript=self._transcribed_text)

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
        if self._agent_answer_speech is None or self._human_input is None:
            return

        if self._agent_answer_speech.synthesis_handle.interrupted:
            return

        # validate the answer & queue it for playout, also add the user question to the chat context
        user_msg = llm.ChatMessage.create(text=self._transcribed_text, role="user")
        self._chat_ctx.messages.append(user_msg)
        self.emit("user_speech_committed", self._chat_ctx, user_msg)

        self._agent_playing_synthesis = self._agent_answer_speech
        self._agent_answer_speech = None
        self._transcribed_text, self._transcribed_interim_text = "", ""
        self._playout_ch.send_nowait(self._agent_playing_synthesis)

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
            final_words = self._opts.word_tokenizer.tokenize(
                text=self._transcribed_text
            )
            interim_words = self._opts.word_tokenizer.tokenize(
                text=self._transcribed_interim_text
            )
            if (
                len(final_words) <= self._opts.int_min_words
                and len(interim_words) <= self._opts.int_min_words
            ):
                return

        self._agent_playing_speech.synthesis_handle.interrupt()

    def _synthesize_answer(self, *, user_transcript: str):
        """
        Synthesize the answer to the user question and make sure
        only one answer is synthesized at a time
        """
        if self._agent_answer_speech is not None:
            self._agent_answer_speech.synthesis_handle.interrupt()

        self._agent_answer_speech = None

        @utils.log_exceptions(logger=logger)
        async def _synthesize_answer_task(old_task: asyncio.Task[None]) -> None:
            # Use an async task to synthesize the agent answer to
            # allow users to execute async code inside the will_create_llm_stream callback
            assert (
                self._agent_output is not None
            ), "agent output should be initialized when ready"

            if old_task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    old_task.cancel()
                    await old_task

            user_msg = llm.ChatMessage.create(text=user_transcript, role="user")
            copied_ctx = self._chat_ctx.copy()
            copied_ctx.messages.append(user_msg)

            llm_stream = self._opts.will_synthesize_assistant_reply(self, copied_ctx)
            if asyncio.iscoroutine(llm_stream):
                llm_stream = await llm_stream

            # fallback to default impl if no custom/user stream is returned
            if llm_stream is None:
                llm_stream = self._llm.chat(chat_ctx=copied_ctx, fnc_ctx=self._fnc_ctx)

            assert isinstance(
                llm_stream, llm.LLMStream
            ), "will_create_llm_stream should be a LLMStream"

            synthesis = self._agent_output.synthesize(
                transcript=_llm_stream_to_str_iterable(llm_stream)
            )
            self._agent_answer_speech = _SpeechInfo(
                source=llm_stream,
                allow_interruptions=self._opts.allow_interruptions,
                add_to_chat_ctx=True,
                synthesis_handle=synthesis,
            )
            self._deferred_validation.on_new_synthesis(user_msg)

        old_task = self._agent_answer_atask
        self._agent_answer_atask = asyncio.create_task(
            _synthesize_answer_task(old_task)
        )

    async def _play_speech(self, speech_info: _SpeechInfo) -> None:
        assert (
            self._agent_output is not None
        ), "agent output should be initialized when ready"

        if speech_info.synthesis_handle.interrupted:
            return

        self._playing_synthesis = speech_info.synthesis_handle
        play_handle = speech_info.synthesis_handle.play()

        # Wait for the playout of the speech to finish (interrupted or done)
        # When the LLM is calling a tool, it doesn't generate any "speech"/"text" to play
        # so awaiting the play_handle will end immediately.
        await play_handle

        collected_text = speech_info.synthesis_handle.collected_text
        interrupted = speech_info.synthesis_handle.interrupted
        if (
            isinstance(speech_info.source, llm.LLMStream)
            and len(speech_info.source.function_calls) > 0
            and not interrupted
        ):
            self.emit("function_calls_collected", speech_info.source.function_calls)

            # run the user functions and automatically generate the LLM answer for it
            # when they're all completed
            called_fncs = speech_info.source.execute_functions()
            tasks = [called_fnc.task for called_fnc in called_fncs]
            await asyncio.gather(*tasks, return_exceptions=True)

            self.emit("function_calls_finished", called_fncs)

            tool_calls = []
            tool_calls_results = []

            for called_fnc in called_fncs:
                # ignore the function calls that returns None
                if called_fnc.result is None:
                    continue

                tool_calls.append(called_fnc.call_info)
                tool_calls_results.append(
                    llm.ChatMessage.create_tool_from_called_function(called_fnc)
                )

            chat_ctx = speech_info.source.chat_ctx.copy()
            chat_ctx.messages.extend(tool_calls)
            chat_ctx.messages.extend(tool_calls_results)

            answer_stream = self._llm.chat(chat_ctx=chat_ctx, fnc_ctx=self._fnc_ctx)
            answer_synthesis = self._agent_output.synthesize(
                transcript=_llm_stream_to_str_iterable(answer_stream)
            )
            await answer_synthesis.play()

            collected_text = answer_synthesis.collected_text
            interrupted = answer_synthesis.interrupted

        if speech_info.add_to_chat_ctx:
            msg = llm.ChatMessage.create(text=collected_text, role="assistant")
            self._chat_ctx.messages.append(msg)

            if interrupted:
                self.emit("agent_speech_interrupted", self._chat_ctx, msg)
            else:
                self.emit("agent_speech_committed", self._chat_ctx, msg)


async def _llm_stream_to_str_iterable(stream: llm.LLMStream) -> AsyncIterable[str]:
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is None:
            continue

        yield content


class _DeferredAnswerValidation:
    DEFER_DELAY = 0.05  # 50ms

    def __init__(
        self, validate_fnc: Callable[[], None], loop: asyncio.AbstractEventLoop
    ) -> None:
        self._validate_fnc = validate_fnc
        self._ts = utils.aio.TaskSet(loop=loop)

        self._validating_task: asyncio.Task | None = None
        self._last_final_transcript = ""
        self._received_end_of_speech = False

    @property
    def validating(self) -> bool:
        return self._validating_task is not None and not self._validating_task.done()

    def on_new_synthesis(self, user_msg: llm.ChatMessage) -> None:
        self._last_final_transcript = user_msg.content

        if self.validating:
            self._run(self.DEFER_DELAY)  # debounce
        elif self._received_end_of_speech:
            # final transcript was received too late
            self._run(self.DEFER_DELAY)
            self._received_end_of_speech = False

    def on_human_start_of_speech(self, ev: vad.VADEvent) -> None:
        if self.validating:
            assert self._validating_task is not None
            self._validating_task.cancel()

    def on_human_end_of_speech(self, ev: vad.VADEvent) -> None:
        self._received_end_of_speech = True

        if self._last_final_transcript:
            self._run(self.DEFER_DELAY)

    async def aclose(self) -> None:
        if self._validating_task is not None:
            self._validating_task.cancel()

        await self._ts.aclose()

    @utils.log_exceptions(logger=logger)
    async def _run_task(self, delay: float) -> None:
        await asyncio.sleep(delay)
        self._last_final_transcript = ""
        self._received_end_of_speech = False
        self._validate_fnc()

    def _run(self, delay: float) -> None:
        if self._validating_task is not None:
            self._validating_task.cancel()

        self._validating = self._ts.create_task(self._run_task(delay))
