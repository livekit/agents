from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterable, Callable, Literal

from livekit import rtc

from .. import aio, tokenize, transcription, utils
from .. import llm as allm
from .. import stt as astt
from .. import tts as atts
from .. import vad as avad
from . import plotter

logger = logging.getLogger("livekit.agents.voice_assistant")


@dataclass
class _SpeechData:
    source: str | allm.LLMStream | AsyncIterable[str]
    allow_interruptions: bool
    add_to_ctx: bool  # should this synthesis be added to the chat context
    validation_future: asyncio.Future[None]  # validate the speech for playout
    validated: bool = False
    interrupted: bool = False
    user_question: str | None = None
    collected_text: str = ""

    def validate_speech(self) -> None:
        self.validated = True
        with contextlib.suppress(asyncio.InvalidStateError):
            self.validation_future.set_result(None)


@dataclass(frozen=True)
class _AssistantOptions:
    plotting: bool
    debug: bool
    allow_interruptions: bool
    int_speech_duration: float
    int_min_words: int
    base_volume: float
    transcription: bool
    preemptive_synthesis: bool
    word_tokenizer: tokenize.WordTokenizer
    sentence_tokenizer: tokenize.SentenceTokenizer
    hyphenate_word: Callable[[str], list[str]]
    transcription_speed: float


@dataclass(frozen=True)
class _StartArgs:
    room: rtc.Room
    participant: rtc.RemoteParticipant | str | None


_ContextVar = contextvars.ContextVar("voice_assistant_contextvar")


class AssistantContext:
    def __init__(self, assistant: "VoiceAssistant", llm_stream: allm.LLMStream) -> None:
        self._assistant = assistant
        self._metadata = dict()
        self._llm_stream = llm_stream

    @staticmethod
    def get_current() -> "AssistantContext":
        return _ContextVar.get()

    @property
    def assistant(self) -> "VoiceAssistant":
        return self._assistant

    def store_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    def llm_stream(self) -> allm.LLMStream:
        return self._llm_stream


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


class VoiceAssistant(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        vad: avad.VAD,
        stt: astt.STT,
        llm: allm.LLM,
        tts: atts.TTS,
        chat_ctx: allm.ChatContext | None = None,
        fnc_ctx: allm.FunctionContext | None = None,
        allow_interruptions: bool = True,
        interrupt_speech_duration: float = 0.65,
        interrupt_min_words: int = 3,
        base_volume: float = 1.0,
        debug: bool = False,
        plotting: bool = False,
        preemptive_synthesis: bool = True,
        loop: asyncio.AbstractEventLoop | None = None,
        transcription: bool = True,
        sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(),
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(),
        hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word,
        transcription_speed: float = 3.83,
    ) -> None:
        super().__init__()
        self._loop = loop or asyncio.get_event_loop()
        self._opts = _AssistantOptions(
            plotting=plotting,
            debug=debug,
            allow_interruptions=allow_interruptions,
            int_speech_duration=interrupt_speech_duration,
            int_min_words=interrupt_min_words,
            base_volume=base_volume,
            preemptive_synthesis=preemptive_synthesis,
            transcription=transcription,
            sentence_tokenizer=sentence_tokenizer,
            word_tokenizer=word_tokenizer,
            hyphenate_word=hyphenate_word,
            transcription_speed=transcription_speed,
        )

        # wrap with adapter automatically with default options
        # to override StreamAdapter options, create the adapter manually
        if not tts.streaming_supported:
            tts = atts.StreamAdapter(
                tts=tts, sentence_tokenizer=tokenize.basic.SentenceTokenizer()
            )

        self._vad, self._tts, self._llm, self._stt = vad, tts, llm, stt
        self._fnc_ctx = fnc_ctx
        self._chat_ctx = chat_ctx or allm.ChatContext()
        self._plotter = plotter.AssistantPlotter(self._loop)

        self._audio_source: rtc.AudioSource | None = None  # published agent audiotrack
        self._user_track: rtc.RemoteAudioTrack | None = None  # user microphone track
        self._user_identity: str | None = None  # linked participant identity

        self._started = False
        self._start_speech_lock = asyncio.Lock()
        self._pending_validation = False

        # tasks
        self._recognize_atask: asyncio.Task | None = None
        self._play_atask: asyncio.Task | None = None
        self._tasks = set[asyncio.Task]()

        # playout state
        self._maybe_answer_task: asyncio.Task | None = None
        self._validated_speech: _SpeechData | None = None
        self._answer_speech: _SpeechData | None = None
        self._playout_start_time: float | None = None

        # synthesis state
        self._speech_playing: _SpeechData | None = None  # validated and playing speech
        self._user_speaking, self._agent_speaking = False, False

        self._target_volume = self._opts.base_volume
        self._vol_filter = utils.ExpFilter(0.9, max_val=self._opts.base_volume)
        self._vol_filter.apply(1.0, self._opts.base_volume)
        self._speech_prob = 0.0
        self._transcribed_text, self._interim_text = "", ""
        self._ready_future = asyncio.Future()

    @property
    def chat_context(self) -> allm.ChatContext:
        return self._chat_ctx

    @property
    def started(self) -> bool:
        return self._started

    def start(
        self, room: rtc.Room, participant: rtc.RemoteParticipant | str | None = None
    ) -> None:
        """Start the voice assistant

        Args:
            room: the room to use
            participant: the participant to listen to, can either be a participant or a participant identity
                If None, the first participant found in the room will be selected
        """
        if self.started:
            raise RuntimeError("voice assistant already started")

        self._started = True
        self._start_args = _StartArgs(room=room, participant=participant)

        room.on("track_published", self._on_track_published)
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)
        room.on("participant_connected", self._on_participant_connected)

        self._main_atask = asyncio.create_task(self._main_task())

    async def say(
        self,
        source: str | allm.LLMStream | AsyncIterable[str],
        *,
        allow_interruptions: bool = True,
        add_to_chat_context: bool = True,
    ) -> None:
        """
        Make the assistant say something.
        The source can be a string, an LLMStream or an AsyncIterable[str]

        Args:
            source: the source of the speech
            allow_interruptions: whether the speech can be interrupted
            add_to_chat_context: whether to add the speech to the chat context
        """
        await self._wait_ready()

        data = _SpeechData(
            source=source,
            allow_interruptions=allow_interruptions,
            add_to_ctx=add_to_chat_context,
            validation_future=asyncio.Future(),
        )
        data.validate_speech()

        await self._start_speech(data, interrupt_current_if_possible=False)

        assert self._play_atask is not None
        await self._play_atask

    def on(self, event: EventTypes, callback: Callable | None = None) -> Callable:
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

    async def aclose(self, wait: bool = True) -> None:
        """
        Close the voice assistant

        Args:
            wait: whether to wait for the current speech to finish before closing
        """
        if not self.started:
            return

        self._ready_future.cancel()

        self._start_args.room.off("track_published", self._on_track_published)
        self._start_args.room.off("track_subscribed", self._on_track_subscribed)
        self._start_args.room.off("track_unsubscribed", self._on_track_unsubscribed)
        self._start_args.room.off(
            "participant_connected", self._on_participant_connected
        )

        self._plotter.terminate()

        with contextlib.suppress(asyncio.CancelledError):
            self._main_atask.cancel()
            await self._main_atask

        if self._recognize_atask is not None:
            self._recognize_atask.cancel()

        if not wait:
            if self._play_atask is not None:
                self._play_atask.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            if self._play_atask is not None:
                await self._play_atask

            if self._recognize_atask is not None:
                await self._recognize_atask

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        """
        Main task is publising the agent audio track and run the update loop
        """
        if self._opts.plotting:
            self._plotter.start()

        if self._start_args.participant is not None:
            if isinstance(self._start_args.participant, rtc.RemoteParticipant):
                self._link_participant(self._start_args.participant.identity)
            else:
                self._link_participant(self._start_args.participant)
        else:
            # no participant provided, try to find the first in the room
            for participant in self._start_args.room.participants.values():
                self._link_participant(participant.identity)
                break

        self._audio_source = rtc.AudioSource(
            self._tts.sample_rate, self._tts.num_channels
        )

        track = rtc.LocalAudioTrack.create_audio_track(
            "assistant_voice", self._audio_source
        )
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        self._pub = await self._start_args.room.local_participant.publish_track(
            track, options
        )

        self._ready_future.set_result(None)

        # Loop running each 10ms to do the following:
        #  - Update the volume based on the user speech probability
        #  - Decide when to interrupt the agent speech
        #  - Decide when to validate the user speech (starting the agent answer)
        speech_prob_avg = utils.MovingAverage(100)
        speaking_avg_validation = utils.MovingAverage(150)
        interruption_speaking_avg = utils.MovingAverage(
            int(self._opts.int_speech_duration * 100)
        )

        interval_10ms = aio.interval(0.01)

        vad_pw = 2.4  # TODO(theomonnom): should this be exposed?
        while True:
            await interval_10ms.tick()

            speech_prob_avg.add_sample(self._speech_prob)
            speaking_avg_validation.add_sample(int(self._user_speaking))
            interruption_speaking_avg.add_sample(int(self._user_speaking))

            bvol = self._opts.base_volume
            self._target_volume = max(0, 1 - speech_prob_avg.get_avg() * vad_pw) * bvol

            if self._validated_speech:
                if not self._validated_speech.allow_interruptions:
                    # avoid volume to go to 0 even if speech probability is high
                    self._target_volume = max(self._target_volume, bvol * 0.5)

                if self._validated_speech.interrupted:
                    # the current speech is interrupted, target volume should be 0
                    self._target_volume = 0

            if self._user_speaking:
                # if the user has been speaking int_speed_duration, interrupt the agent speech
                # (this currently allows 10% of noise in the VAD)
                if interruption_speaking_avg.get_avg() >= 0.1:
                    self._interrupt_if_needed()
            elif self._pending_validation:
                if speaking_avg_validation.get_avg() <= 0.05:
                    self._validate_answer_if_needed()

            self._plotter.plot_value("raw_vol", self._target_volume)
            self._plotter.plot_value("vad_probability", self._speech_prob)

    def _link_participant(self, identity: str) -> None:
        p = self._start_args.room.participants_by_identity.get(identity)
        assert p is not None, "_link_participant should be called with a valid identity"

        # set self._user_identity before calling _on_track_published or _on_track_subscribed
        self._user_identity = identity
        self._log_debug(f"linking participant {identity}")

        for pub in p.tracks.values():
            if pub.subscribed:
                self._on_track_subscribed(pub.track, pub, p)  # type: ignore
            else:
                self._on_track_published(pub, p)

    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        if not self._user_identity:
            self._link_participant(participant.identity)

    def _on_track_published(
        self, pub: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        if (
            participant.identity != self._user_identity
            or pub.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return

        if not pub.subscribed:
            pub.set_subscribed(True)

    def _on_track_subscribed(
        self,
        track: rtc.RemoteTrack,
        pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if (
            participant.identity != self._user_identity
            or pub.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return

        self._log_debug("starting listening to user microphone")
        self._user_track = track  # type: ignore
        self._recognize_atask = asyncio.create_task(
            self._recognize_task(rtc.AudioStream(track))
        )

    def _on_track_unsubscribed(
        self,
        track: rtc.RemoteTrack,
        pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if (
            participant.identity != self._user_identity
            or pub.source != rtc.TrackSource.SOURCE_MICROPHONE
            or self._user_track is None
        ):
            return

        # user microphone unsubscribed, (participant disconnected/track unpublished)
        self._log_debug("user microphone not available anymore")
        assert (
            self._recognize_atask is not None
        ), "recognize task should be running when user_track was set"
        self._recognize_atask.cancel()
        self._user_track = None

    @utils.log_exceptions(logger=logger)
    async def _recognize_task(self, audio_stream: rtc.AudioStream) -> None:
        """
        Receive the frames from the user audio stream and do the following:
         - do Voice Activity Detection (VAD)
         - do Speech-to-Text (STT)
        """
        assert (
            self._user_identity is not None
        ), "user identity should be set before recognizing"

        vad_stream = self._vad.stream()
        stt_stream = self._stt.stream()

        stt_forwarder = utils._noop.Nop()
        if self._opts.transcription:
            stt_forwarder = transcription.STTSegmentsForwarder(
                room=self._start_args.room,
                participant=self._user_identity,
                track=self._user_track,
            )

        async def _audio_stream_co() -> None:
            async for ev in audio_stream:
                stt_stream.push_frame(ev.frame)
                vad_stream.push_frame(ev.frame)

        async def _vad_stream_co() -> None:
            async for ev in vad_stream:
                if ev.type == avad.VADEventType.START_OF_SPEECH:
                    self._log_debug("user started speaking")
                    self._plotter.plot_event("user_started_speaking")
                    self._user_speaking = True
                    self.emit("user_started_speaking")
                elif ev.type == avad.VADEventType.INFERENCE_DONE:
                    self._speech_prob = ev.probability
                elif ev.type == avad.VADEventType.END_OF_SPEECH:
                    self._log_debug(f"user stopped speaking {ev.duration:.2f}s")
                    self._plotter.plot_event("user_started_speaking")
                    self._pending_validation = True
                    self._user_speaking = False
                    self.emit("user_stopped_speaking")

        async def _stt_stream_co() -> None:
            async for ev in stt_stream:
                stt_forwarder.update(ev)
                if ev.type == astt.SpeechEventType.FINAL_TRANSCRIPT:
                    self._on_final_transcript(ev.alternatives[0].text)
                elif ev.type == astt.SpeechEventType.INTERIM_TRANSCRIPT:
                    # interim transcript is used in combination with VAD
                    # to interrupt the current speech.
                    # (can be disabled by setting int_min_words to 0)
                    self._interim_text = ev.alternatives[0].text
                elif ev.type == astt.SpeechEventType.END_OF_SPEECH:
                    self._pending_validation = True

        try:
            await asyncio.gather(
                _audio_stream_co(),
                _vad_stream_co(),
                _stt_stream_co(),
            )
        finally:
            await asyncio.gather(
                stt_forwarder.aclose(wait=False),
                stt_stream.aclose(wait=False),
                vad_stream.aclose(),
            )

    def _on_final_transcript(self, text: str) -> None:
        self._transcribed_text += text
        self._log_debug(f"received final transcript: {self._transcribed_text}")

        # to create an llm stream we need an async context
        # setting it to "" and will be updated inside the _answer_task below
        # (this function can't be async because we don't want to block _update_co)
        self._answer_speech = _SpeechData(
            source="",
            allow_interruptions=self._opts.allow_interruptions,
            add_to_ctx=True,
            validation_future=asyncio.Future(),
            user_question=self._transcribed_text,
        )

        # this speech may not be validated, so we create a copy
        # of our context to add the new user message
        copied_ctx = self._chat_ctx.copy()
        copied_ctx.messages.append(
            allm.ChatMessage(
                text=self._transcribed_text,
                role=allm.ChatRole.USER,
            )
        )

        if self._maybe_answer_task is not None:
            self._maybe_answer_task.cancel()

        async def _answer_task(ctx: allm.ChatContext, data: _SpeechData) -> None:
            try:
                data.source = await self._llm.chat(ctx, fnc_ctx=self._fnc_ctx)
                await self._start_speech(data, interrupt_current_if_possible=False)
            except Exception:
                logger.exception("error while answering")

        t = asyncio.create_task(_answer_task(copied_ctx, self._answer_speech))
        self._maybe_answer_task = t
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    def _interrupt_if_needed(self) -> None:
        """
        Check whether the current assistant speech should be interrupted
        """
        if (
            not self._validated_speech
            or not self._opts.allow_interruptions
            or self._validated_speech.interrupted
        ):
            return

        if self._opts.int_min_words != 0:
            txt = self._transcribed_text.strip().split()
            if len(txt) <= self._opts.int_min_words:
                txt = self._interim_text.strip().split()
                if len(txt) <= self._opts.int_min_words:
                    return

        if (
            self._playout_start_time is not None
            and (time.time() - self._playout_start_time) < 1
        ):  # don't interrupt new speech (if they're not older than 1s)
            return

        self._validated_speech.interrupted = True
        self._validate_answer_if_needed()
        self._log_debug("user interrupted assistant speech")

    def _validate_answer_if_needed(self) -> None:
        """
        Check whether the current pending answer to the user should be validated (played)
        """
        if self._answer_speech is None:
            return

        if self._agent_speaking and (
            self._validated_speech and not self._validated_speech.interrupted
        ):
            return

        self._pending_validation = False
        self._transcribed_text = self._interim_text = ""
        self._answer_speech.validate_speech()
        self._log_debug("user speech validated")

    async def _start_speech(
        self, data: _SpeechData, *, interrupt_current_if_possible: bool
    ) -> None:
        await self._wait_ready()

        async with self._start_speech_lock:
            # interrupt the current speech if possible, otherwise wait before playing the new speech
            if self._play_atask is not None:
                if self._validated_speech is not None:
                    if (
                        interrupt_current_if_possible
                        and self._validated_speech.allow_interruptions
                    ):
                        logger.debug("_start_speech - interrupting current speech")
                        self._validated_speech.interrupted = True

                else:
                    # pending speech isn't validated yet, OK to cancel
                    self._play_atask.cancel()

                with contextlib.suppress(asyncio.CancelledError):
                    await self._play_atask

            self._play_atask = asyncio.create_task(
                self._play_speech_if_validated_task(data)
            )

    @utils.log_exceptions(logger=logger)
    async def _play_speech_if_validated_task(self, data: _SpeechData) -> None:
        """
        Start synthesis and playout the speech only if validated
        """
        self._log_debug(f"play_speech_if_validated {data.user_question}")

        # reset volume before starting a new speech
        self._vol_filter.reset()
        playout_tx, playout_rx = aio.channel()  # playout channel

        tts_forwarder = utils._noop.Nop()
        if self._opts.transcription:
            tts_forwarder = transcription.TTSSegmentsForwarder(
                room=self._start_args.room,
                participant=self._start_args.room.local_participant,
                track=self._pub.sid,
                sentence_tokenizer=self._opts.sentence_tokenizer,
                word_tokenizer=self._opts.word_tokenizer,
                hyphenate_word=self._opts.hyphenate_word,
                speed=self._opts.transcription_speed,
            )

        if not self._opts.preemptive_synthesis:
            await data.validation_future

        tts_co = self._synthesize_task(data, playout_tx, tts_forwarder)
        _synthesize_task = asyncio.create_task(tts_co)

        try:
            # wait for speech validation before playout
            await data.validation_future

            # validated!
            self._validated_speech = data
            self._playout_start_time = time.time()

            if data.user_question is not None:
                msg = allm.ChatMessage(text=data.user_question, role=allm.ChatRole.USER)
                self._chat_ctx.messages.append(msg)
                self.emit("user_speech_committed", self._chat_ctx, msg)

            self._log_debug("starting playout")
            await self._playout_co(playout_rx, tts_forwarder)

            msg = allm.ChatMessage(
                text=data.collected_text,
                role=allm.ChatRole.ASSISTANT,
            )

            if data.add_to_ctx:
                self._chat_ctx.messages.append(msg)
                if data.interrupted:
                    self.emit("agent_speech_interrupted", self._chat_ctx, msg)
                else:
                    self.emit("agent_speech_committed", self._chat_ctx, msg)

            self._log_debug("playout finished", extra={"interrupted": data.interrupted})
        finally:
            self._validated_speech = None
            with contextlib.suppress(asyncio.CancelledError):
                _synthesize_task.cancel()
                await _synthesize_task

            # make sure that _synthesize_task is finished before closing the transcription
            # forwarder. pushing text/audio to the forwarder after closing it will raise an exception
            await tts_forwarder.aclose()
            self._log_debug("play_speech_if_validated_task finished")

    async def _synthesize_speech_co(
        self,
        data: _SpeechData,
        playout_tx: aio.ChanSender[rtc.AudioFrame],
        text: str,
        tts_forwarder: transcription.TTSSegmentsForwarder | utils._noop.Nop,
    ) -> None:
        """synthesize speech from a string"""
        data.collected_text += text
        tts_forwarder.push_text(text)
        tts_forwarder.mark_text_segment_end()

        start_time = time.time()
        first_frame = True
        audio_duration = 0.0

        try:
            async for audio in self._tts.synthesize(text):
                if first_frame:
                    first_frame = False
                    dt = time.time() - start_time
                    self._log_debug(f"tts first frame in {dt:.2f}s")

                frame = audio.data
                audio_duration += frame.samples_per_channel / frame.sample_rate

                playout_tx.send_nowait(frame)
                tts_forwarder.push_audio(frame)

        finally:
            tts_forwarder.mark_audio_segment_end()
            playout_tx.close()
            self._log_debug(f"tts finished synthesising {audio_duration:.2f}s of audio")

    async def _synthesize_streamed_speech_co(
        self,
        data: _SpeechData,
        playout_tx: aio.ChanSender[rtc.AudioFrame],
        streamed_text: AsyncIterable[str],
        tts_forwarder: transcription.TTSSegmentsForwarder | utils._noop.Nop,
    ) -> None:
        """synthesize speech from streamed text"""

        async def _read_generated_audio_task():
            start_time = time.time()
            first_frame = True
            audio_duration = 0.0
            async for event in tts_stream:
                if event.type == atts.SynthesisEventType.AUDIO:
                    if first_frame:
                        first_frame = False
                        dt = time.time() - start_time
                        self._log_debug(f"tts first frame in {dt:.2f}s (streamed)")

                    assert event.audio is not None
                    frame = event.audio.data
                    audio_duration += frame.samples_per_channel / frame.sample_rate
                    tts_forwarder.push_audio(frame)
                    playout_tx.send_nowait(frame)

            self._log_debug(
                f"tts finished synthesising {audio_duration:.2f}s audio (streamed)"
            )

        # otherwise, stream the text to the TTS
        tts_stream = self._tts.stream()
        read_task = asyncio.create_task(_read_generated_audio_task())

        try:
            async for seg in streamed_text:
                data.collected_text += seg
                tts_forwarder.push_text(seg)
                tts_stream.push_text(seg)

        finally:
            tts_forwarder.mark_text_segment_end()
            tts_stream.mark_segment_end()

            await tts_stream.aclose()
            await read_task

            tts_forwarder.mark_audio_segment_end()
            playout_tx.close()

    @utils.log_exceptions(logger=logger)
    async def _synthesize_task(
        self,
        data: _SpeechData,
        playout_tx: aio.ChanSender[rtc.AudioFrame],
        tts_forwarder: transcription.TTSSegmentsForwarder | utils._noop.Nop,
    ) -> None:
        """Synthesize speech from the source. Also run LLM inference when needed"""
        if isinstance(data.source, str):
            await self._synthesize_speech_co(
                data, playout_tx, data.source, tts_forwarder
            )
        elif isinstance(data.source, allm.LLMStream):
            llm_stream = data.source
            assistant_ctx = AssistantContext(self, llm_stream)
            token = _ContextVar.set(assistant_ctx)

            async def _forward_llm_chunks():
                async for chunk in llm_stream:
                    alt = chunk.choices[0].delta.content
                    if not alt:
                        continue
                    yield alt

            await self._synthesize_streamed_speech_co(
                data, playout_tx, _forward_llm_chunks(), tts_forwarder
            )

            if len(llm_stream.called_functions) > 0:
                self.emit("function_calls_collected", assistant_ctx)

            await llm_stream.aclose()

            if len(llm_stream.called_functions) > 0:
                self.emit("function_calls_finished", assistant_ctx)

            _ContextVar.reset(token)
        else:
            await self._synthesize_streamed_speech_co(
                data, playout_tx, data.source, tts_forwarder
            )

    async def _playout_co(
        self,
        playout_rx: aio.ChanReceiver[rtc.AudioFrame],
        tts_forwarder: transcription.TTSSegmentsForwarder | utils._noop.Nop,
    ) -> None:
        """
        Playout audio with the current volume.
        The playout_rx is streaming the synthesized speech from the TTS provider to minimize latency
        """
        assert (
            self._audio_source is not None
        ), "audio source should be set before playout"

        def _should_break():
            eps = 1e-6
            assert self._validated_speech is not None
            return (
                self._validated_speech.interrupted
                and self._vol_filter.filtered() <= eps
            )

        first_frame = True
        early_break = False

        async for frame in playout_rx:
            if first_frame:
                self._log_debug("agent started speaking")
                self._plotter.plot_event("agent_started_speaking")
                self._agent_speaking = True
                self.emit("agent_started_speaking")
                tts_forwarder.segment_playout_started()  # we have only one segment
                first_frame = False

            if _should_break():
                early_break = True
                break

            # divide the frame by chunks of 20ms
            ms20 = frame.sample_rate // 50
            i = 0
            while i < len(frame.data):
                if _should_break():
                    break

                rem = min(ms20, len(frame.data) - i)
                data = frame.data[i : i + rem]
                i += rem

                dt = 1 / len(data)
                for si in range(0, len(data)):
                    vol = self._vol_filter.apply(dt, self._target_volume)
                    data[si] = int((data[si] / 32768) * vol * 32768)

                await self._audio_source.capture_frame(
                    rtc.AudioFrame(
                        data=data.tobytes(),
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                        samples_per_channel=rem,
                    )
                )

        if not first_frame:
            self._log_debug("agent stopped speaking")
            if not early_break:
                tts_forwarder.segment_playout_finished()

            self._plotter.plot_event("agent_stopped_speaking")
            self._agent_speaking = False
            self.emit("agent_stopped_speaking")

    def _log_debug(self, msg: str, **kwargs) -> None:
        if self._opts.debug:
            logger.debug(msg, **kwargs)

    async def _wait_ready(self) -> None:
        """Wait for the assistant to be fully started"""
        await self._ready_future
