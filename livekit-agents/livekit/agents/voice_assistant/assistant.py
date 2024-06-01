from __future__ import annotations

import asyncio
import contextlib
import contextvars
import time
from typing import Any, AsyncIterable, Callable, Literal

from attrs import define
from livekit import rtc

from .. import aio, tokenize, transcription, utils
from .. import llm as allm
from .. import stt as astt
from .. import tts as atts
from .. import vad as avad
from ..log import logger
from . import plotter


@define(kw_only=True)
class _SpeechData:
    source: str | allm.LLMStream | AsyncIterable[str] | None = None
    allow_interruptions: bool
    add_to_ctx: bool  # should this synthesis be added to the chat context
    val_ch: aio.Chan[None]  # validate the speech for playout
    validated: bool = False
    interrupted: bool = False
    collected_text: str = (
        ""  # if the source is a stream, this will be the collected text
    )

    answering_user_speech: str | None = None  # the this speech is answering to


def _validate_speech(data: _SpeechData):
    data.validated = True
    data.val_ch.close()


@define(kw_only=True, frozen=True)
class _AssistantOptions:
    plotting: bool
    debug: bool
    allow_interruptions: bool
    int_speech_duration: float
    # some STT doesn't support streaming (e.g Whisper)
    # so it doesn't make sense to wait for a certain amount of words
    # before interrupting the speech. we should set this to 0 in that case
    int_min_words: int
    base_volume: float
    transcription: bool
    word_tokenizer: tokenize.WordTokenizer
    sentence_tokenizer: tokenize.SentenceTokenizer
    hyphenate_word: Callable[[str], list[str]]
    transcription_speed: float


@define(kw_only=True, frozen=True)
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
        interrupt_volume: float = 0.05,
        interrupt_speech_duration: float = 0.65,
        interrupt_min_words: int = 3,
        base_volume: float = 1.0,
        debug: bool = False,
        plotting: bool = False,
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
            transcription=transcription,
            sentence_tokenizer=sentence_tokenizer,
            word_tokenizer=word_tokenizer,
            hyphenate_word=hyphenate_word,
            transcription_speed=transcription_speed,
        )
        self._vad, self._tts, self._llm, self._stt = vad, tts, llm, stt
        self._fnc_ctx = fnc_ctx
        self._chat_ctx = chat_ctx or allm.ChatContext()
        self._speaking, self._user_speaking = False, False
        self._plotter = plotter.AssistantPlotter(self._loop)

        self._audio_source, self._audio_stream = None, None
        self._closed, self._started, self._ready = False, False, False
        self._linked_participant = ""

        self._pending_validation = False

        # tasks
        self._launch_task: asyncio.Task | None = None
        self._recognize_task: asyncio.Task | None = None
        self._update_task: asyncio.Task | None = None
        self._play_task: asyncio.Task | None = None
        self._tasks = set[asyncio.Task]()

        # playout state
        self._maybe_answer_task: asyncio.Task | None = None
        self._playing_speech: _SpeechData | None = None
        self._answer_speech: _SpeechData | None = None
        self._playout_start_time: float | None = None

        # synthesis state
        self._speech_playing: _SpeechData | None = None  # validated and playing speech
        self._user_speaking, self._agent_speaking = False, False

        self._target_volume = self._opts.base_volume
        self._vol_filter = utils.ExpFilter(0.9, max_val=self._opts.base_volume)
        self._vol_filter.apply(1.0, self._opts.base_volume)
        self._speech_prob = 0.0
        self._last_speech_prob = 0.0
        self._transcripted_text, self._interim_text = "", ""
        self._start_future = asyncio.Future()

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

    @property
    def chat_context(self) -> allm.ChatContext:
        return self._chat_ctx

    @property
    def started(self) -> bool:
        return self._started

    async def say(
        self,
        source: str | allm.LLMStream | AsyncIterable[str],
        *,
        allow_interruptions: bool = True,
        add_to_ctx: bool = True,
    ) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            await self._start_future

        data = _SpeechData(
            source=source,
            allow_interruptions=allow_interruptions,
            add_to_ctx=add_to_ctx,
            val_ch=aio.Chan[None](),
        )

        _validate_speech(data)
        await self._start_speech(data, interrupt_current_if_possible=True)
        assert self._play_task is not None
        await self._play_task

    def start(
        self, room: rtc.Room, participant: rtc.RemoteParticipant | str | None = None
    ) -> None:
        """Start the voice assistant in a room

        Args:
            room: the room currently in use
            participant: the participant to listen to, can either be a participant or a participant identity
                If None, the first participant in the room will be used
        """
        if self.started:
            logger.warning("voice assistant already started")
            return

        self._started = True
        self._start_args = _StartArgs(room=room, participant=participant)

        room.on("track_published", self._on_track_published)
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)
        room.on("participant_connected", self._on_participant_connected)

        self._launch_task = asyncio.create_task(self._launch())

    async def aclose(self, wait: bool = True) -> None:
        if not self.started:
            return

        self._closed = True
        self._start_future.cancel()

        self._start_args.room.off("track_published", self._on_track_published)
        self._start_args.room.off("track_subscribed", self._on_track_subscribed)
        self._start_args.room.off("track_unsubscribed", self._on_track_unsubscribed)
        self._start_args.room.off(
            "participant_connected", self._on_participant_connected
        )

        if self._opts.plotting:
            self._plotter.terminate()

        with contextlib.suppress(asyncio.CancelledError):
            if self._launch_task is not None:
                self._launch_task.cancel()
                await self._launch_task

        if self._recognize_task is not None:
            self._recognize_task.cancel()

        if not wait:
            if self._update_task is not None:
                self._update_task.cancel()

            if self._play_task is not None:
                self._play_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            if self._update_task is not None:
                await self._update_task

            if self._recognize_task is not None:
                await self._recognize_task

            if self._play_task is not None:
                await self._play_task

    async def _launch(self):
        self._log_debug("assistant - launching")

        if self._opts.plotting:
            self._plotter.start()

        if self._start_args.participant is not None:
            if isinstance(self._start_args.participant, rtc.RemoteParticipant):
                self._link_participant(self._start_args.participant.identity)
            else:
                self._link_participant(self._start_args.participant)
        else:
            for participant in self._start_args.room.participants.values():
                self._link_participant(participant.identity)
                break

        self._audio_source = rtc.AudioSource(
            self._tts.sample_rate, self._tts.num_channels
        )
        self._update_task = asyncio.create_task(self._update_loop())

        track = rtc.LocalAudioTrack.create_audio_track("voice", self._audio_source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        self._pub = await self._start_args.room.local_participant.publish_track(
            track, options
        )
        self._start_future.set_result(None)

    def _link_participant(self, identity: str):
        p = self._start_args.room.participants_by_identity.get(identity)
        assert p is not None

        # link partcipant before subscribing to tracks to avoid race where
        # _on_track_published or _on_track_subscribed is quickly called before
        # self._linked_participant is set
        self._linked_participant = identity
        self._log_debug(f"assistant - linked participant {identity}")

        for pub in p.tracks.values():
            if pub.subscribed:
                self._on_track_subscribed(pub.track, pub, p)  # type: ignore
            else:
                self._on_track_published(pub, p)

    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        if not self._linked_participant:
            self._link_participant(participant.identity)

    def _on_track_published(
        self, pub: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        if participant.identity != self._linked_participant:
            return

        if pub.source == rtc.TrackSource.SOURCE_MICROPHONE and not pub.subscribed:
            pub.set_subscribed(True)

    def _on_track_subscribed(
        self,
        track: rtc.RemoteTrack,
        pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if participant.identity != self._linked_participant:
            return

        if pub.source == rtc.TrackSource.SOURCE_MICROPHONE:
            self._user_track_ready(track)  # type: ignore

    def _on_track_unsubscribed(
        self,
        track: rtc.RemoteTrack,
        pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if participant.identity != self._linked_participant:
            return

        if pub.source == rtc.TrackSource.SOURCE_MICROPHONE:
            self._user_track_dirty()

    def _user_track_dirty(self):
        """Called when the AudioStream isn't valid anymore (microphone unsubscribed or participant
        disconnected)"""
        if not self._ready:
            logger.warning("assistant already marked as dirty")
            return

        self._log_debug("marking assistant as dirty")
        self._ready = False

        if self._recognize_task is not None:
            self._recognize_task.cancel()

    def _user_track_ready(self, user_track: rtc.AudioTrack):
        """We have everything we need to start the voice assistant (audio sources & audio
        streams)"""
        if self._ready:
            logger.warning("assistant already marked as ready")
            return

        audio_stream = rtc.AudioStream(user_track)

        self._log_debug("marking assistant as ready")
        self._user_track = user_track
        self._audio_stream = audio_stream
        self._ready = True
        self._recognize_task = asyncio.create_task(self._recognize_loop())

    def _recv_final_transcript(self, ev: astt.SpeechEvent):
        self._log_debug(f"assistant - received transcript {ev.alternatives[0].text}")
        self._transcripted_text += ev.alternatives[0].text
        self._maybe_answer(self._transcripted_text)

    def _recv_interim_transcript(self, ev: astt.SpeechEvent):
        self._interim_text = ev.alternatives[0].text

    def _transcript_finished(self, ev: astt.SpeechEvent):
        self._log_debug("assistant - transcript finished")
        self._pending_validation = True

    def _did_vad_inference(self, ev: avad.VADEvent):
        self._plotter.plot_value("vad_raw", ev.raw_inference_prob)
        self._plotter.plot_value("vad_smoothed", ev.probability)
        self._plotter.plot_value("vad_dur", ev.inference_duration * 1000)
        self._speech_prob = ev.raw_inference_prob

    def _user_started_speaking(self):
        self._log_debug("assistant - user started speaking")
        self._plotter.plot_event("user_started_speaking")
        self._user_speaking = True
        self.emit("user_started_speaking")

    def _user_stopped_speaking(self, speech_duration: float):
        self._log_debug(f"assistant - user stopped speaking {speech_duration:.2f}s")
        self._plotter.plot_event("user_started_speaking")
        self._pending_validation = True
        self._user_speaking = False
        self.emit("user_stopped_speaking")

    def _agent_started_speaking(self):
        self._log_debug("assistant - agent started speaking")
        self._plotter.plot_event("agent_started_speaking")
        self._agent_speaking = True
        self.emit("agent_started_speaking")

    def _agent_stopped_speaking(self):
        self._log_debug("assistant - agent stopped speaking")
        self._plotter.plot_event("agent_stopped_speaking")
        self._agent_speaking = False
        self.emit("agent_stopped_speaking")

    async def _recognize_loop(self):
        """Recognize speech from the audio stream and do voice activity detection"""
        assert self._audio_stream is not None

        vad_stream = self._vad.stream()
        stt_stream = self._stt.stream()

        stt_forwarder = utils._noop.Nop()
        if self._opts.transcription:
            stt_forwarder = transcription.STTSegmentsForwarder(
                room=self._start_args.room,
                participant=self._linked_participant,
                track=self._user_track,
            )

        select = aio.select([self._audio_stream, vad_stream, stt_stream])
        try:
            while True:
                s = await select()
                if s.selected is self._audio_stream:
                    audio_event: rtc.AudioFrameEvent = s.result()
                    stt_stream.push_frame(audio_event.frame)
                    vad_stream.push_frame(audio_event.frame)

                if s.selected is vad_stream:
                    vad_event: avad.VADEvent = s.result()
                    if vad_event.type == avad.VADEventType.START_OF_SPEECH:
                        self._user_started_speaking()
                    elif vad_event.type == avad.VADEventType.INFERENCE_DONE:
                        self._did_vad_inference(vad_event)
                    elif vad_event.type == avad.VADEventType.END_OF_SPEECH:
                        self._user_stopped_speaking(vad_event.duration)

                if s.selected is stt_stream:
                    stt_event = s.result()
                    stt_forwarder.update(stt_event)
                    if stt_event.type == astt.SpeechEventType.FINAL_TRANSCRIPT:
                        self._recv_final_transcript(stt_event)
                    elif stt_event.type == astt.SpeechEventType.INTERIM_TRANSCRIPT:
                        self._recv_interim_transcript(stt_event)
                    elif stt_event.type == astt.SpeechEventType.END_OF_SPEECH:
                        self._transcript_finished(stt_event)
        except Exception:
            logger.exception("error in recognize loop")
        finally:
            await asyncio.gather(
                stt_forwarder.aclose(wait=False),
                stt_stream.aclose(wait=False),
                vad_stream.aclose(wait=False),
                select.aclose(),
            )

    async def _update_loop(self):
        """Update the volume every 10ms based on the speech probability, decide whether to interrupt
        and when to validate an answer"""
        speech_prob_avg = utils.MovingAverage(100)
        speaking_avg_validation = utils.MovingAverage(210)
        interruption_speaking_avg = utils.MovingAverage(
            int(self._opts.int_speech_duration * 100)
        )

        interval_10ms = aio.interval(0.01)

        vad_pw = 2.4  # TODO(theomonnom): should this be exposed?
        while not self._closed:
            await interval_10ms.tick()

            speech_prob_avg.add_sample(self._speech_prob)
            speaking_avg_validation.add_sample(int(self._user_speaking))
            interruption_speaking_avg.add_sample(int(self._user_speaking))

            bvol = self._opts.base_volume
            self._target_volume = max(0, 1 - speech_prob_avg.get_avg() * vad_pw) * bvol

            if self._playing_speech:
                if not self._playing_speech.allow_interruptions:
                    # avoid volume to go to 0 even if speech probability is high
                    self._target_volume = max(self._target_volume, bvol * 0.5)

                if self._playing_speech.interrupted:
                    # the current speech is interrupted, target volume should be 0
                    self._target_volume = 0

            if self._user_speaking:
                if (
                    interruption_speaking_avg.get_avg() >= 0.1
                ):  # allow 10% of "noise"/false positives in the VAD?
                    self._interrupt_if_needed()
            elif self._pending_validation:
                if speaking_avg_validation.get_avg() <= 0.05:
                    self._pending_validation = False
                    self._validate_answer_if_needed()

            if self._opts.plotting:
                self._plotter.plot_value("raw_t_vol", self._target_volume)
                self._plotter.plot_value("vol", self._vol_filter.filtered())

    def _interrupt_if_needed(self):
        """Check whether the current speech should be interrupted"""
        if (
            not self._playing_speech
            or not self._opts.allow_interruptions
            or self._playing_speech.interrupted
        ):
            return

        if self._opts.int_min_words != 0:
            txt = self._transcripted_text.strip().split()
            if len(txt) <= self._opts.int_min_words:
                txt = self._interim_text.strip().split()
                if len(txt) <= self._opts.int_min_words:
                    return

        if (
            self._playout_start_time is not None
            and (time.time() - self._playout_start_time) < 1
        ):  # don't interrupt new speech (if they're not older than 1s)
            return

        self._log_debug("assistant - interrupting speech")
        self._playing_speech.interrupted = True
        self._validate_answer_if_needed()

    def _validate_answer_if_needed(self):
        if self._answer_speech is None:
            return

        if self._agent_speaking and (
            self._playing_speech and not self._playing_speech.interrupted
        ):
            return

        self._log_debug("assistant - validating answer")
        self._transcripted_text = self._interim_text = ""
        _validate_speech(self._answer_speech)

    def _maybe_answer(self, text: str) -> None:
        async def _answer_if_validated(
            ctx: allm.ChatContext, data: _SpeechData
        ) -> None:
            try:
                data.source = await self._llm.chat(ctx, fnc_ctx=self._fnc_ctx)
                await self._start_speech(data, interrupt_current_if_possible=False)
            except Exception:
                logger.exception("error while answering")

        self._answer_speech = _SpeechData(
            allow_interruptions=self._opts.allow_interruptions,
            add_to_ctx=True,
            val_ch=aio.Chan[None](),
            answering_user_speech=text,
        )

        messages = self._chat_ctx.messages.copy()
        user_msg = allm.ChatMessage(
            text=text,
            role=allm.ChatRole.USER,
        )
        messages.append(user_msg)
        ctx = allm.ChatContext(messages=messages)

        if self._maybe_answer_task is not None:
            self._maybe_answer_task.cancel()

        t = asyncio.create_task(_answer_if_validated(ctx, self._answer_speech))
        self._maybe_answer_task = t
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _start_speech(
        self, data: _SpeechData, *, interrupt_current_if_possible: bool
    ) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            await self._start_future

        if data.source is None:
            raise ValueError("source must be provided")

        # interrupt the current speech if possible, otherwise wait before playing the new speech
        if self._playing_speech is not None:
            assert self._play_task is not None

            if (
                interrupt_current_if_possible
                and self._playing_speech
                and self._playing_speech.allow_interruptions
            ):
                logger.debug("assistant - interrupting current speech")
                self._playing_speech.interrupted = True

            logger.debug("assistant - waiting for current speech to finish")
            await self._play_task
            logger.debug("assistant - current speech finished")
        elif self._play_task is not None:
            self._play_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._play_task

        self._play_task = asyncio.create_task(self._play_speech_if_validated(data))

    async def _play_speech_if_validated(self, data: _SpeechData) -> None:
        """
        Start synthesis and playout the speech only if validated
        """
        self._log_debug(f"assistant - play_speech_if_validated {data}")
        assert data.source is not None

        # reset volume before starting a new speech
        self._vol_filter.reset()
        po_tx, po_rx = aio.channel()  # playout channel

        tts_forwarder = utils._noop.Nop()
        if self._opts.transcription:
            tts_forwarder = transcription.TTSSegmentsForwarder(
                room=self._start_args.room,
                participant=self._start_args.room.local_participant,
                track=self._pub.sid,
                auto_playout=False,
                sentence_tokenizer=self._opts.sentence_tokenizer,
                word_tokenizer=self._opts.word_tokenizer,
                hyphenate_word=self._opts.hyphenate_word,
                speed=self._opts.transcription_speed,
            )

        tts_co = self._synthesize_task(data, po_tx, tts_forwarder)
        _synthesize_task = asyncio.create_task(tts_co)

        try:
            with contextlib.suppress(aio.ChanClosed):
                await data.val_ch.recv()  # wait for speech validation before playout

            # validated!
            self._log_debug("assistant - speech validated")
            self._playing_speech = data
            self._playout_start_time = time.time()

            if data.answering_user_speech is not None:
                msg = allm.ChatMessage(
                    text=data.answering_user_speech, role=allm.ChatRole.USER
                )
                self._chat_ctx.messages.append(msg)
                self.emit("user_speech_committed", self._chat_ctx, msg)

            await self._playout_task(po_rx, tts_forwarder)

            msg = allm.ChatMessage(
                text=data.collected_text,
                role=allm.ChatRole.ASSISTANT,
            )

            if data.add_to_ctx:
                self._chat_ctx.messages.append(msg)
                if data.interrupted:
                    self.emit("agent_speech_interrupted", self._chat_ctx, msg)
                    await tts_forwarder.aclose(wait=False)
                else:
                    self.emit("agent_speech_committed", self._chat_ctx, msg)
                    await tts_forwarder.aclose()

            self._log_debug(
                "assistant - playout finished", extra={"interrupted": data.interrupted}
            )

        except Exception:
            logger.exception("error while playing speech")
        finally:
            self._playing_speech = None
            with contextlib.suppress(asyncio.CancelledError):
                _synthesize_task.cancel()
                await _synthesize_task

            await tts_forwarder.aclose(wait=False)
            self._log_debug("assistant - play_speech_if_validated finished")

    async def _synthesize_task(
        self,
        data: _SpeechData,
        po_tx: aio.ChanSender[rtc.AudioFrame],
        tts_forwarder: transcription.TTSSegmentsForwarder | utils._noop.Nop,
    ) -> None:
        """Synthesize speech from the source"""
        assert data.source is not None
        self._log_debug("tts inference started")

        if isinstance(data.source, str):
            # No streaming is needed, use the TTS directly
            # This should be faster when the whole text is known in advance
            # (no buffering on the provider side)
            data.collected_text = data.source
            tts_forwarder.push_text(data.source)
            tts_forwarder.mark_text_segment_end()
            _start_time = time.time()
            _first_frame = True
            async for audio in self._tts.synthesize(data.source):
                if _first_frame:
                    dt = time.time() - _start_time
                    _first_frame = False
                    self._log_debug(f"assistant - tts first frame in {dt:.2f}s")

                tts_forwarder.push_audio(audio.data)
                po_tx.send_nowait(audio.data)

            tts_forwarder.mark_audio_segment_end()
            po_tx.close()
            self._log_debug("tts inference finished")
            return

        # otherwise, stream the text to the TTS..
        tts_stream = self._tts.stream()

        async def _forward_stream():
            """forward tts_stream to playout"""
            _start_time = time.time()
            _first_frame = True
            async for event in tts_stream:
                if event.type == atts.SynthesisEventType.FINISHED:
                    break

                if event.type == atts.SynthesisEventType.AUDIO:
                    if _first_frame:
                        dt = time.time() - _start_time
                        _first_frame = False
                        self._log_debug(
                            f"assistant - tts first frame in {dt:.2f}s (streamed)"
                        )

                    assert event.audio is not None
                    tts_forwarder.push_audio(event.audio.data)
                    po_tx.send_nowait(event.audio.data)

            tts_forwarder.mark_audio_segment_end()

        _forward_task = asyncio.create_task(_forward_stream())
        try:
            if isinstance(data.source, allm.LLMStream):
                # stream the LLM output to the TTS
                assistant_ctx = AssistantContext(self, data.source)
                token = _ContextVar.set(assistant_ctx)
                async for chunk in data.source:
                    alt = chunk.choices[0].delta.content
                    if not alt:
                        continue
                    data.collected_text += alt
                    tts_forwarder.push_text(alt)
                    tts_stream.push_text(alt)

                tts_forwarder.mark_text_segment_end()
                tts_stream.mark_segment_end()

                if len(data.source.called_functions) > 0:
                    self.emit("function_calls_collected", assistant_ctx)

                await data.source.aclose()

                if len(data.source.called_functions) > 0:
                    self.emit("function_calls_finished", assistant_ctx)

                _ContextVar.reset(token)
            else:
                # user defined source, stream the text to the TTS
                async for seg in data.source:
                    data.collected_text += seg
                    tts_forwarder.push_text(seg)
                    tts_stream.push_text(seg)

                tts_forwarder.mark_text_segment_end()
                tts_stream.mark_segment_end()

            await tts_stream.aclose()
        except Exception:
            logger.exception("error while streaming text to TTS")
        finally:
            po_tx.close()
            with contextlib.suppress(asyncio.CancelledError):
                _forward_task.cancel()
                await _forward_task

            self._log_debug("tts inference finished")

    async def _playout_task(
        self,
        po_rx: aio.ChanReceiver[rtc.AudioFrame],
        tts_forwarder: transcription.TTSSegmentsForwarder | utils._noop.Nop,
    ) -> None:
        """Playout the synthesized speech with volume control"""
        assert self._audio_source is not None

        sample_idx = 0
        first_frame = True

        def _should_break():
            eps = 1e-6
            assert self._playing_speech is not None
            return (
                self._playing_speech.interrupted and self._vol_filter.filtered() <= eps
            )

        async for buf in po_rx:
            if first_frame:
                self._agent_started_speaking()
                tts_forwarder.segment_playout_started()  # we have only one segment
                first_frame = False

            if _should_break():
                break

            i = 0
            while i < len(buf.data):
                if _should_break():
                    break

                ms10 = buf.sample_rate // 100
                rem = min(ms10, len(buf.data) - i)
                data = buf.data[i : i + rem]
                i += rem

                dt = 1 / len(data)
                for si in range(0, len(data)):
                    vol = self._vol_filter.apply(dt, self._target_volume)
                    j = data[si] / 32768
                    data[si] = int(j * vol * 32768)
                    sample_idx += 1

                await self._audio_source.capture_frame(
                    rtc.AudioFrame(
                        data=data.tobytes(),
                        sample_rate=buf.sample_rate,
                        num_channels=buf.num_channels,
                        samples_per_channel=rem,
                    )
                )

        self._agent_stopped_speaking()

    def _log_debug(self, msg: str, **kwargs) -> None:
        if self._opts.debug:
            logger.debug(msg, **kwargs)
