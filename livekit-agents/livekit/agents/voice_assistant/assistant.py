import asyncio
import contextlib
import contextvars
import time
from typing import Any, AsyncIterable, Literal

from attrs import define
from livekit import rtc

from .. import aio, utils
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
    user_speech: str | None = None
    interrupted: bool = False


def _validate_speech(data: _SpeechData):
    data.validated = True
    data.val_ch.close()


@define(kw_only=True, frozen=True)
class _AssistantOptions:
    plotting: bool
    debug: bool
    allow_interruptions: bool
    int_speech_duration: float
    int_min_words: int
    base_volume: float


CallContextVar = contextvars.ContextVar("voice_assistant_call_context")


class CallContext:
    def __init__(self, assistant: "VoiceAssistant") -> None:
        self._assistant = assistant
        self._texts = []
        self._metadata = dict()

    @property
    def assistant(self) -> "VoiceAssistant":
        return self._assistant

    def store_text(self, text: str) -> None:
        self._texts.append(text)

    def store_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value


EventTypes = Literal[
    "user_started_speaking",
    "user_stopped_speaking",
    "agent_started_speaking",
    "agent_stopped_speaking",
]


class VoiceAssistant(utils.EventEmitter[EventTypes]):
    _SAY_CUSTOM_PRIORITY = 3
    _SAY_AGENT_ANSWER_PRIORITY = 2
    _SAY_CALL_CONTEXT_PRIORITY = 1

    def __init__(
        self,
        vad: avad.VAD,
        stt: astt.STT,
        llm: allm.LLM,
        tts: atts.TTS,
        *,
        fnc_ctx: allm.FunctionContext | None = None,
        allow_interruptions: bool = True,
        interrupt_volume: float = 0.05,
        interrupt_speech_duration: float = 0.7,
        interrupt_min_words: int = 3,
        base_volume: float = 1.0,
        debug: bool = False,
        plotting: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
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
        )
        self._vad, self._tts, self._llm, self._stt = vad, tts, llm, stt
        self._fnc_ctx, self._ctx = fnc_ctx, allm.ChatContext()
        self._speaking, self._user_speaking = False, False
        self._plotter = plotter.AssistantPlotter(self._loop)

        self._audio_source, self._audio_stream = None, None
        self._closed, self._started, self._ready = False, False, False

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
        self._speech_queue = asyncio.PriorityQueue[_SpeechData]()
        self._user_speaking, self._agent_speaking = False, False

        self._target_volume = self._opts.base_volume
        self._vol_filter = utils.ExpFilter(0.9, max_val=self._opts.base_volume)
        self._vol_filter.apply(1.0, self._opts.base_volume)
        self._speech_prob = 0.0
        self._speaking_avg = utils.MovingAverage(
            int(self._opts.int_speech_duration * 100)
        )
        self._transcripted_text, self._interim_text = "", ""
        self._start_future = asyncio.Future()

    @property
    def chat_context(self) -> allm.ChatContext:
        return self._ctx

    @property
    def started(self) -> bool:
        return self._started

    def start(self, room: rtc.Room, participant: rtc.RemoteParticipant | str) -> None:
        """Start the voice assistant in a room

        Args:
            room: the room currently in use
            participant: the participant to listen to, can either be a participant or a participant identity
        """
        if self.started:
            logger.warning("voice assistant already started")
            return

        self._started = True
        self._room = room
        if isinstance(participant, rtc.RemoteParticipant):
            self._participant_identity = participant.identity
        else:
            self._participant_identity = participant

        p = room.participants_by_identity.get(self._participant_identity)
        if p is not None:
            # check if a track has already been published
            for pub in p.tracks.values():
                if pub.subscribed:
                    self._on_track_subscribed(pub.track, pub, p)  # type: ignore
                else:
                    self._on_track_published(pub, p)

        self._room.on("track_published", self._on_track_published)
        self._room.on("track_subscribed", self._on_track_subscribed)
        self._room.on("track_unsubscribed", self._on_track_unsubscribed)

        self._launch_task = asyncio.create_task(self._launch())

    async def _launch(self):
        self._log_debug("assistant - launching")

        if self._opts.plotting:
            self._plotter.start()

        self._audio_source = rtc.AudioSource(
            self._tts.sample_rate, self._tts.num_channels
        )
        self._update_task = asyncio.create_task(self._update_loop())

        track = rtc.LocalAudioTrack.create_audio_track("voice", self._audio_source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        self._pub = await self._room.local_participant.publish_track(track, options)
        self._start_future.set_result(None)

    def _on_track_published(
        self, pub: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        if participant.identity != self._participant_identity:
            return

        if pub.source == rtc.TrackSource.SOURCE_MICROPHONE:
            pass
            # TODO: remove after Python SDK release
            # pub.set_subscribed(True)

    def _on_track_subscribed(
        self,
        track: rtc.RemoteTrack,
        pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if participant.identity != self._participant_identity:
            return

        if pub.source == rtc.TrackSource.SOURCE_MICROPHONE:
            audio_stream = rtc.AudioStream(track)
            self._mark_ready(audio_stream)

    def _on_track_unsubscribed(
        self,
        track: rtc.RemoteTrack,
        pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if participant.identity != self._participant_identity:
            return

        if pub.source == rtc.TrackSource.SOURCE_MICROPHONE:
            self._mark_dirty()

    def _mark_dirty(self):
        """Called when the AudioStream isn't valid anymore (microphone unsubscribed or participant
        disconnected)"""
        if not self._ready:
            logger.warning("assistant already marked as dirty")
            return

        self._log_debug("marking assistant as dirty")
        self._ready = False

        if self._recognize_task is not None:
            self._recognize_task.cancel()

    def _mark_ready(self, audio_stream: rtc.AudioStream):
        """We have everything we need to start the voice assistant (audio sources & audio
        streams)"""
        if self._ready:
            logger.warning("assistant already marked as ready")
            return

        self._log_debug("marking assistant as ready")
        self._audio_stream = audio_stream
        self._ready = True
        self._recognize_task = asyncio.create_task(self._recognize_loop())

    async def aclose(self, wait: bool = True) -> None:
        if not self.started:
            return

        self._closed = True
        self._start_future.cancel()
        self._room.off("track_published", self._on_track_published)
        self._room.off("track_subscribed", self._on_track_subscribed)
        self._room.off("track_unsubscribed", self._on_track_unsubscribed)

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

    async def say(
        self,
        source: str | allm.LLMStream | AsyncIterable[str],
        allow_interruptions: bool = True,
        add_to_ctx: bool = True,
        stream: bool = True,
        enqueue: bool = True,
    ) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            if not self._started:
                await self._start_future

        if isinstance(source, str) and stream:
            text = source

            async def _gen():
                yield text

            source = _gen()

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

    async def _recognize_loop(self):
        """Recognize speech from the audio stream and do voice activity detection"""
        assert self._audio_stream is not None

        vad_stream = self._vad.stream()
        stt_stream = self._stt.stream()

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
                    if stt_event.type == astt.SpeechEventType.FINAL_TRANSCRIPT:
                        self._recv_final_transcript(stt_event)
                    elif stt_event.type == astt.SpeechEventType.INTERIM_TRANSCRIPT:
                        self._recv_interim_transcript(stt_event)
                    elif stt_event.type == astt.SpeechEventType.END_OF_SPEECH:
                        self._transcript_finished(stt_event)
        except Exception:
            logger.exception("error in recognize loop")
        finally:
            await stt_stream.aclose(wait=False)
            await vad_stream.aclose(wait=False)
            await select.aclose()

    async def _update_loop(self):
        """Update the volume every 10ms based on the speech probability"""
        speech_prob_avg = utils.MovingAverage(100)  # avg over 1s

        vad_pw = 2.4  # should this be exposed
        while not self._closed:
            bvol = self._opts.base_volume

            self._speaking_avg.add_sample(int(self._user_speaking))
            speech_prob_avg.add_sample(
                self._speech_prob
            )  # not totally accurate due to timing between vad inference and this task
            self._target_volume = max(0, 1 - speech_prob_avg.get_avg() * vad_pw) * bvol

            if self._playing_speech:
                if not self._playing_speech.allow_interruptions:
                    # avoid volume to go to 0 even if speech probability is high
                    self._target_volume = max(self._target_volume, bvol * 0.5)

                if self._playing_speech.interrupted:
                    # the current speech is interrupted, target volume should be 0
                    self._target_volume = 0

            if self._user_speaking:
                self._interrupt_if_needed()

            if self._opts.plotting:
                self._plotter.plot_value("raw_t_vol", self._target_volume)
                self._plotter.plot_value("vol", self._vol_filter.filtered())

            await asyncio.sleep(0.01)

    def _recv_final_transcript(self, ev: astt.SpeechEvent):
        self._log_debug(f"assistant - received transcript {ev.alternatives[0].text}")
        self._transcripted_text += ev.alternatives[0].text
        self._interrupt_if_needed()
        self._maybe_answer(self._transcripted_text)

    def _recv_interim_transcript(self, ev: astt.SpeechEvent):
        self._interim_text = ev.alternatives[0].text
        self._interrupt_if_needed()

    def _transcript_finished(self, ev: astt.SpeechEvent):
        self._log_debug("assistant - transcript finished")
        self._transcripted_text = self._interim_text = ""
        self._interrupt_if_needed()
        self._validate_answer_if_needed()

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
        self._interrupt_if_needed()
        self._validate_answer_if_needed()
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

    def _interrupt_if_needed(self):
        """Check whether the current speech should be interrupted"""
        if (
            not self._playing_speech
            or not self._opts.allow_interruptions
            or self._playing_speech.interrupted
        ):
            return

        if (
            self._speaking_avg.get_avg() < 0.9
        ):  # allow 10% of "noise"/false positives in the VAD?
            return

        txt = self._transcripted_text.strip().split()
        if len(txt) <= self._opts.int_min_words:
            txt = self._interim_text.strip().split()
            if len(txt) <= self._opts.int_min_words:
                return

        if (
            self._playout_start_time is not None
            and (time.time() - self._playout_start_time) < 0.5
        ):  # don't interrupt new speech (if they're not older than 0.5s)
            return

        self._log_debug("assistant - interrupting speech")
        self._playing_speech.interrupted = True

    def _validate_answer_if_needed(self):
        if self._answer_speech is None:
            return

        if self._agent_speaking and (
            self._playing_speech and not self._playing_speech.interrupted
        ):
            return

        self._log_debug("assistant - validating answer")
        _validate_speech(self._answer_speech)

    def _maybe_answer(self, text: str) -> None:
        async def _answer_if_validated(
            ctx: allm.ChatContext, data: _SpeechData
        ) -> None:
            try:
                llm_stream = await self._llm.chat(ctx, fnc_ctx=self._fnc_ctx)
                data.source = llm_stream
                await self._start_speech(data, interrupt_current_if_possible=False)
            except Exception:
                logger.exception("error while answering")

        if self._maybe_answer_task is not None:
            self._maybe_answer_task.cancel()

        data = _SpeechData(
            allow_interruptions=self._opts.allow_interruptions,
            add_to_ctx=True,
            val_ch=aio.Chan[None](),
        )
        self._answer_speech = data

        msg = self._ctx.messages.copy()
        msg.append(allm.ChatMessage(text=text, role=allm.ChatRole.USER))
        ctx = allm.ChatContext(messages=msg)

        t = asyncio.create_task(_answer_if_validated(ctx, data))
        self._maybe_answer_task = t
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _start_speech(
        self, data: _SpeechData, *, interrupt_current_if_possible: bool
    ):
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

        self._play_task = asyncio.create_task(self._play_speech_if_validated(data))

    async def _play_speech_if_validated(self, data: _SpeechData) -> None:
        """
        Start synthesis and playout the speech only if validated
        """
        self._log_debug(f"assistant - maybe_play_speech {data}")
        assert data.source is not None

        # reset volume before starting a new speech
        self._vol_filter.reset()
        po_tx, po_rx = aio.channel()  # playout channel
        tts_co = self._tts_task(data.source, po_tx)
        _tts_task = asyncio.create_task(tts_co)

        try:
            with contextlib.suppress(aio.ChanClosed):
                _ = (
                    await data.val_ch.recv()
                )  # wait for speech validation before playout

            self._log_debug("assistant - speech validated")
            # validated!
            self._playing_speech = data
            self._playout_start_time = time.time()
            await self._playout_task(po_rx)
            if (
                not data.interrupted
                and data.add_to_ctx
                and _tts_task.done()
                and not _tts_task.cancelled()
            ):
                # add the played text to the chat context if it was not interrupted
                # and the synthesis was successful
                text = _tts_task.result()

                if data.user_speech is not None:
                    self._ctx.messages.append(
                        allm.ChatMessage(text=data.user_speech, role=allm.ChatRole.USER)
                    )

                self._ctx.messages.append(
                    allm.ChatMessage(
                        text=text,
                        role=allm.ChatRole.ASSISTANT,
                    )
                )

            self._log_debug(
                "assistant - playout finished", extra={"interrupted": data.interrupted}
            )
        except Exception:
            logger.exception("error while playing speech")
        finally:
            self._playing_speech = None
            with contextlib.suppress(asyncio.CancelledError):
                _tts_task.cancel()
                await _tts_task

            self._log_debug("assistant - maybe_play_speech finished")

    async def _tts_task(
        self,
        source: str | allm.LLMStream | AsyncIterable[str],
        po_tx: aio.ChanSender[rtc.AudioFrame],
    ) -> str:
        """Synthesize speech from the source

        Returns:
            the completed text from the source
        """
        self._log_debug("tts inference started")

        if isinstance(source, str):
            # No streaming is needed, use the TTS directly
            # This should be faster when the whole text is known in advance
            # (no buffering on the provider side)
            _start_time = time.time()
            _first_frame = True
            async for audio in self._tts.synthesize(source):
                if _first_frame:
                    dt = time.time() - _start_time
                    _first_frame = False
                    self._log_debug(f"assistant - tts first frame in {dt:.2f}s")

                po_tx.send_nowait(audio.data)

            po_tx.close()
            self._log_debug("tts inference finished")
            return source

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
                    po_tx.send_nowait(event.audio.data)

        _forward_task = asyncio.create_task(_forward_stream())
        text = ""
        try:
            if isinstance(source, allm.LLMStream):
                # stream the LLM output to the TTS
                async for chunk in source:
                    alt = chunk.choices[0].delta.content
                    if not alt:
                        continue
                    text += alt
                    tts_stream.push_text(alt)

                tts_stream.mark_segment_end()

                await source.aclose()
            else:
                # user defined source, stream the text to the TTS
                async for seg in source:
                    text += seg
                    tts_stream.push_text(seg)

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

        return text

    async def _playout_task(
        self,
        po_rx: aio.ChanReceiver[rtc.AudioFrame],
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
            if first_frame and self._opts.debug:
                self._agent_started_speaking()
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

                frame = rtc.AudioFrame(
                    data=data.tobytes(),
                    sample_rate=buf.sample_rate,
                    num_channels=buf.num_channels,
                    samples_per_channel=rem,
                )
                await self._audio_source.capture_frame(frame)

        self._agent_stopped_speaking()

    def _log_debug(self, msg: str, **kwargs):
        if self._opts.debug:
            logger.debug(msg, **kwargs)
