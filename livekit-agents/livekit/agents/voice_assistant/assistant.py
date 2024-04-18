import contextvars
import sys
import asyncio
import io
import struct
import contextlib
import functools
import enum
import logging
import time

from typing import AsyncIterable, AsyncIterator, Callable, ClassVar, Literal, Tuple, Any

from attrs import define
from livekit import rtc

from .. import aio
from .. import utils
from .. import llm as allm
from .. import stt as astt
from .. import tts as atts
from .. import vad as avad
from ..log import logger

from . import plotter


@define(kw_only=True)
class _SpeechData:
    source: str | allm.LLMStream | AsyncIterable[str]
    allow_interruptions: bool
    add_to_ctx: bool  # should this synthesis be added to the chat context
    val_ch: aio.Chan[None]  # validate the speech for playout
    interrupted: bool = False


@define(kw_only=True, frozen=True)
class _AssistantOptions:
    plotting: bool
    debug: bool
    allow_interruptions: bool
    int_speech_duration: float
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


class VoiceAssistant:
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
        interrupt_speech_duration: float = 1.5,
        base_volume: float = 1.0,
        debug: bool = False,
        plotting: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._opts = _AssistantOptions(
            plotting=plotting,
            debug=debug,
            allow_interruptions=allow_interruptions,
            int_speech_duration=interrupt_speech_duration,
            base_volume=base_volume,
        )
        self._vad, self._tts, self._llm, self._stt = vad, tts, llm, stt
        self._fnc_ctx, self._ctx = fnc_ctx, allm.ChatContext()
        self._speaking, self._user_speaking = False, False
        self._plotter = plotter.AssistantPlotter(self._loop)
        self._closed = False

        self._audio_source, self._audio_stream = None, None
        self._ready = False

        # synthesis states
        self._cur_speech: _SpeechData | None = None
        self._play_task: asyncio.Task | None = None
        self._user_speaking, self._agent_speaking = False, False

        self._target_volume = self._opts.base_volume
        self._vol_filter = utils.ExpFilter(0.9, max_val=self._opts.base_volume)
        self._vol_filter.apply(1.0, self._opts.base_volume)
        self._speech_prob = 0.0

    @property
    def chat_context(self) -> allm.ChatContext:
        return self._ctx

    @property
    def started(self) -> bool:
        return self._room is not None

    async def start(
        self, room: rtc.Room, participant: rtc.RemoteParticipant | str
    ) -> None:
        """Start the voice assistant in a room

        Args:
            room: the room currently in use
            participant: the participant to listen to, can either be a participant or a participant identity
        """
        if self.started:
            raise ValueError("voice assistant is already started")

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
                    pub.set_subscribed(True)

        self._room.on("track_published", self._on_track_published)
        self._room.on("track_subscribed", self._on_track_subscribed)
        self._room.on("track_unsubscribed", self._on_track_unsubscribed)

    def _on_track_published(
        self, pub: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        if participant.identity != self._participant_identity:
            return

        if pub.source == rtc.TrackSource.SOURCE_MICROPHONE:
            pub.set_subscribed(True)

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
            self._mark_ready()

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
        assert self._audio_stream is not None
        if not self._ready:
            return

        self._ready = False
        self._recognize_task.cancel()

    def _mark_ready(self):
        """We have everything we need to start the voice assistant (audio sources & audio
        streams)"""
        assert self._audio_stream is not None

        if self._ready:
            raise ValueError("voice assistant is already ready")

        self._ready = True
        if self._opts.plotting:
            self._plotter.start()

        self._recognize_task = asyncio.create_task(self._recognize_loop())
        self._update_task = asyncio.create_task(self._update_loop())

    async def aclose(self, wait: bool = True) -> None:
        if not self.started:
            return

        self._closed = True
        self._room.off("track_published", self._on_track_published)
        self._room.off("track_subscribed", self._on_track_subscribed)
        self._room.off("track_unsubscribed", self._on_track_unsubscribed)

        if self._opts.plotting:
            self._plotter.terminate()

        self._recognize_task.cancel()
        if not wait:
            self._update_task.cancel()
            if self._play_task is not None:
                self._play_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await self._recognize_task
            await self._update_task
            if self._play_task is not None:
                await self._play_task

    async def say(
        self,
        source: str | allm.LLMStream | AsyncIterable[str],
        allow_interruptions: bool = True,
        add_to_ctx: bool = True,
    ) -> None:
        val_ch = aio.Chan[None]()
        data = _SpeechData(
            source=source,
            allow_interruptions=allow_interruptions,
            add_to_ctx=add_to_ctx,
            val_ch=val_ch,
        )

        val_ch.send_nowait(None)
        await self._start_speech(data)

    async def _recognize_loop(self):
        """Recognize speech from the audio stream and do voice activity detection"""
        assert self._audio_stream is not None

        self._audio_source = rtc.AudioSource(
            self._tts.sample_rate, self._tts.num_channels
        )
        track = rtc.LocalAudioTrack.create_audio_track(
            "assistant_voice", self._audio_source
        )

        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        self._pub = await self._room.local_participant.publish_track(track, options)

        vad_stream = self._vad.stream()
        stt_stream = self._stt.stream()

        select = aio.select([self._audio_stream, vad_stream, stt_stream])
        try:
            async for s in select:
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
        finally:
            await stt_stream.aclose(wait=False)
            await vad_stream.aclose(wait=False)

    async def _update_loop(self):
        """Update the volume every 10ms based on the speech probability"""
        speech_avg = utils.MovingAverage(100)  # avg over 1s
        vad_pw = 2.4  # should this be exposed
        while not self._closed:
            bvol = self._opts.base_volume

            speech_avg.add_sample(self._speech_prob)
            self._target_volume = max(0, 1 - speech_avg.get_avg() * vad_pw) * bvol

            if self._cur_speech and self._cur_speech.interrupted:
                # the current speech is interrupted, target volume should be 0
                self._target_volume = 0

            if self._opts.plotting:
                self._plotter.plot_value("raw_t_vol", self._target_volume)
                self._plotter.plot_value("vol", self._vol_filter.filtered())

            await asyncio.sleep(0.01)

    def _recv_final_transcript(self, ev: astt.SpeechEvent):
        self._log_debug(f"assistant - received transcript {ev.alternatives[0].text}")

    def _transcript_finished(self, ev: astt.SpeechEvent):
        # validate playout
        pass

    def _did_vad_inference(self, ev: avad.VADEvent):
        self._plotter.plot_value("vad_raw", ev.raw_inference_prob)
        self._plotter.plot_value("vad_smoothed", ev.probability)
        self._plotter.plot_value("vad_dur", ev.inference_duration * 1000)
        self._speech_prob = ev.raw_inference_prob

    def _user_started_speaking(self):
        self._log_debug("assistant - user started speaking")
        self._plotter.plot_event("user_started_speaking")
        self._user_speaking = True

    def _user_stopped_speaking(self, speech_duration: float):
        self._log_debug(f"assistant - user stopped speaking {speech_duration:.2f}s")
        self._plotter.plot_event("user_started_speaking")
        self._user_speaking = False

    def _agent_started_speaking(self):
        self._log_debug("assistant - agent started speaking")
        self._plotter.plot_event("agent_started_speaking")
        self._agent_speaking = True

    def _agent_stopped_speaking(self):
        self._log_debug("assistant - agent stopped speaking")
        self._plotter.plot_event("agent_stopped_speaking")
        self._agent_speaking = False

    def _log_debug(self, msg: str, **kwargs):
        if self._opts.debug:
            logger.debug(msg, **kwargs)

    async def _start_speech(
        self,
        data: _SpeechData,
    ):
        # interrupt the current speech if possible and start the new one
        if self._cur_speech is not None:
            assert self._play_task is not None

            if self._cur_speech and self._cur_speech.allow_interruptions:
                await self._stop_speech()

            with contextlib.suppress(asyncio.CancelledError):
                await self._play_task

        if self._user_speaking and data.allow_interruptions:
            self._log_debug("assistant - user is speaking, ignoring new speech request")
            return

        # start the new synthesis
        self._cur_speech = data
        self._play_task = asyncio.create_task(self._maybe_play_speech(data))

    async def _stop_speech(self):
        if self._cur_speech is None:
            return

        with contextlib.suppress(asyncio.CancelledError):
            assert self._play_task is not None
            self._cur_speech.interrupted = True
            await self._play_task

    async def _maybe_play_speech(self, data: _SpeechData) -> None:
        """
        Start synthesis and playout the speech only if validated
        """
        self._log_debug(f"assistant - maybe_play_speech {data}")

        # reset volume before starting a new speech
        self._vol_filter.reset()
        po_tx, po_rx = aio.channel()  # playout channel
        tts_co = self._tts_task(data.source, po_tx)
        playout_co = self._playout_task(po_rx, data.val_ch)

        _tts_task = asyncio.create_task(tts_co)
        try:
            await playout_co

            if (
                not data.interrupted
                and data.add_to_ctx
                and _tts_task.done()
                and not _tts_task.cancelled()
            ):
                # add the played text to the chat context if it was not interrupted
                # and the synthesis was successful
                text = _tts_task.result()

                new_msg = allm.ChatMessage(
                    text=text,
                    role=allm.ChatRole.ASSISTANT,
                )
                self._ctx.messages.append(new_msg)

            self._log_debug(
                "assistant - playout finished", extra={"interrupted": data.interrupted}
            )
        except Exception:
            logger.exception("error while playing speech")
        finally:
            with contextlib.suppress(asyncio.CancelledError):
                _tts_task.cancel()
                await _tts_task

            self._cur_speech = None
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
        self._log_debug(f"tts inference started")

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

            po_tx.close()

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
                await source.aclose(wait=False)
            elif isinstance(source, AsyncIterable):
                # user defined source, stream the text to the TTS
                async for text in source:
                    text += text
                    tts_stream.push_text(text)

                tts_stream.mark_segment_end()
        finally:
            await tts_stream.aclose(wait=False)
            if isinstance(source, allm.LLMStream):
                await source.aclose(wait=False)

            with contextlib.suppress(asyncio.CancelledError):
                _forward_task.cancel()
                await _forward_task

            po_tx.close()

        return text

    async def _playout_task(
        self,
        po_rx: aio.ChanReceiver[rtc.AudioFrame],
        val_rx: aio.ChanReceiver[None],
    ) -> None:
        """Playout the synthesized speech with volume control"""
        assert self._audio_source is not None

        _ = await val_rx.recv()  # wait for speech validation before playout

        sample_idx = 0
        first_frame = True

        def _should_break():
            eps = 1e-6
            assert self._cur_speech is not None
            return self._cur_speech.interrupted and self._vol_filter.filtered() <= 1e-6

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
