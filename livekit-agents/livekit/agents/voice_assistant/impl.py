from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import AsyncIterable, Awaitable, Callable, Optional, Union

from livekit import rtc

from .. import aio, llm, stt, tokenize, tts, utils, vad
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


WillCreateLLMStream = Callable[
    ["AssistantImpl", llm.ChatContext],
    Union[Optional[llm.LLMStream], Awaitable[Optional[llm.LLMStream]]],
]


@dataclass(frozen=True)
class ImplOptions:
    allow_interruptions: bool
    int_speech_duration: float
    int_min_words: int
    preemptive_synthesis: bool
    will_synthesize_assistant_answer: WillCreateLLMStream
    plotting: bool
    debug: bool

    # transcription & transcript analysis
    transcription: bool
    word_tokenizer: tokenize.WordTokenizer
    sentence_tokenizer: tokenize.SentenceTokenizer
    hyphenate_word: Callable[[str], list[str]]
    transcription_speed: float


class AssistantImpl:
    UPDATE_INTERVAL_S = 0.5  # 2tps
    PLOT_INTERVAL_S = 0.05  # 20tps

    def __init__(
        self,
        *,
        vad: vad.VAD,
        stt: stt.STT,
        llm: llm.LLM,
        tts: tts.TTS,
        emitter: utils.EventEmitter,
        options: ImplOptions,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._stt, self._vad, self._llm, self._tts = stt, vad, llm, tts
        self._emitter = emitter
        self._opts = options
        self._loop = loop

        self._chat_ctx, self._fnc_ctx = chat_ctx, fnc_ctx
        self._started, self._closed = False, False
        self._plotter = AssistantPlotter(self._loop)

        self._human_input: HumanInput | None = None
        self._agent_output: AgentOutput | None = None
        self._ready_future = asyncio.Future()

        self._agent_answer_speech: _SpeechInfo | None = None
        self._agent_answer_atask: asyncio.Task[None] | None = None
        self._agent_playing_speech: _SpeechInfo | None = (
            None  # speech currently being played
        )
        self._queued_playouts: list[_SpeechInfo] = []

        self._transcribed_text, self._transcribed_interim_text = "", ""

    def start(
        self, room: rtc.Room, participant: rtc.RemoteParticipant | str | None = None
    ) -> None:
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

    async def aclose(self) -> None:
        if not self._started:
            return

        self._room.off("participant_connected", self._on_participant_connected)

    async def say(
        self,
        source: str | llm.LLMStream | AsyncIterable[str],
        *,
        allow_interruptions: bool = True,
        add_to_chat_ctx: bool = True,
    ) -> None:
        await self._ready_future
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
        self._queued_playouts.append(speech)

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

        def _on_human_start_of_speech(ev: vad.VADEvent) -> None:
            self._plotter.plot_event("user_started_speaking")
            self._emitter.emit("user_started_speaking")

        def _on_human_vad_updated(ev: vad.VADEvent) -> None:
            tv = max(0, 1 - ev.probability)
            self._audio_source.target_volume = tv

            self._plotter.plot_value("raw_vol", tv)
            self._plotter.plot_value("vad_probability", ev.probability)

            if ev.duration >= self._opts.int_speech_duration:
                self._interrupt_if_needed()

        def _on_human_end_of_speech(ev: vad.VADEvent) -> None:
            self._validate_answer_if_needed()
            self._plotter.plot_event("user_started_speaking")
            self._emitter.emit("user_stopped_speaking")

        def _on_human_interim_transcript(ev: stt.SpeechEvent) -> None:
            self._transcribed_interim_text = ev.alternatives[0].text

        def _on_human_final_transcript(ev: stt.SpeechEvent) -> None:
            self._transcribed_text += ev.alternatives[0].text
            self._synthesize_answer(user_transcript=self._transcribed_text)

        self._human_input.on("start_of_speech", _on_human_start_of_speech)
        self._human_input.on("vad_inference_done", _on_human_vad_updated)
        self._human_input.on("end_of_speech", _on_human_end_of_speech)
        self._human_input.on("interim_transcript", _on_human_interim_transcript)
        self._human_input.on("final_transcript", _on_human_final_transcript)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        audio_source = rtc.AudioSource(self._tts.sample_rate, self._tts.num_channels)
        track = rtc.LocalAudioTrack.create_audio_track("assistant_voice", audio_source)
        self._agent_publication = await self._room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )

        self._audio_source = CancellableAudioSource(source=audio_source)
        self._agent_output = AgentOutput(
            room=self._room,
            source=self._audio_source,
            llm=self._llm,
            tts=self._tts,
        )

        self._ready_future.set_result(None)

        async def _update_loop_co():
            interval_s = AssistantImpl.UPDATE_INTERVAL_S
            interval = aio.interval(interval_s)
            while True:
                await interval.tick()

                if len(self._queued_playouts) > 0:
                    speech = self._queued_playouts.pop()
                    self._agent_playing_speech = speech
                    await self._play_speech(speech)
                    self._agent_playing_speech = None

        async def _plotter_co():
            # plot volume and vad probability

            interval_s = AssistantImpl.UPDATE_INTERVAL_S
            interval = aio.interval(interval_s)
            while True:
                await interval.tick()

        coros = []
        coros.append(_update_loop_co())
        if self._opts.plotting:
            coros.append(_plotter_co())

        await asyncio.gather(*coros)

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

    def _validate_answer_if_needed(self) -> None:
        """
        Check if the user speech should be validated/played
        """
        if self._agent_answer_speech is None or self._human_input is None:
            return

        if (
            self._human_input.speaking
            or self._agent_answer_speech.synthesis_handle.interrupted
        ):
            return

        # validate the answer & queue it for playout, also add the user question to the chat context
        user_msg = llm.ChatMessage.create(text=self._transcribed_text, role="user")
        self._chat_ctx.messages.append(user_msg)
        self._emitter.emit("user_speech_committed", self._chat_ctx, user_msg)

        self._agent_playing_synthesis = self._agent_answer_speech
        self._agent_answer_speech = None
        self._transcribed_text, self._transcribed_interim_text = "", ""
        self._queued_playouts.append(self._agent_playing_synthesis)

    def _synthesize_answer(self, *, user_transcript: str):
        """
        Synthesize the answer to the user question and make sure only one answer is synthesized at a time
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

            llm_stream = self._opts.will_synthesize_assistant_answer(self, copied_ctx)
            if asyncio.iscoroutine(llm_stream):
                llm_stream = await llm_stream

            # fallback to default impl if no custom/user stream is returned
            if llm_stream is None:
                llm_stream = self._llm.chat(chat_ctx=copied_ctx, fnc_ctx=self._fnc_ctx)

            assert isinstance(
                llm_stream, llm.LLMStream
            ), "will_create_llm_stream should be a LLMStream"

            source = _llm_stream_to_str_iterable(llm_stream)
            synthesis = self._agent_output.synthesize(transcript=source)
            self._agent_answer_speech = _SpeechInfo(
                source=llm_stream,
                allow_interruptions=self._opts.allow_interruptions,
                add_to_chat_ctx=True,
                synthesis_handle=synthesis,
            )

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
            # TODO(theomonnom): emit function calls events & add call context

            # run the user functions and automatically generate the LLM answer for it
            # when they're all completed
            called_fncs = speech_info.source.execute_functions()
            tasks = [called_fnc.task for called_fnc in called_fncs]
            await asyncio.gather(*tasks, return_exceptions=True)

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
                self._emitter.emit("agent_speech_interrupted", self._chat_ctx, msg)
            else:
                self._emitter.emit("agent_speech_committed", self._chat_ctx, msg)


async def _llm_stream_to_str_iterable(stream: llm.LLMStream) -> AsyncIterable[str]:
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is not None:
            yield content
