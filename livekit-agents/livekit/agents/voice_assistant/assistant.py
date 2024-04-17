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

from math import copysign, exp2, pi
from typing import AsyncIterable, AsyncIterator, Callable, ClassVar, Literal, Tuple, Any

from attrs import define, evolve
from livekit import rtc

from .. import aio
from .. import utils
from .. import llm as allm
from .. import stt as astt
from .. import tts as atts
from .. import vad as avad
from ..log import logger

from . import plotter


class _InterruptState(enum.Enum):
    NONE = 0  # bring back volume to 1
    STARTED = 1  # volume to int_volume - speech_prob
    VALIDATED = 2  # volume to 0


class _PlayoutResult(enum.Enum):
    FINISHED = 0
    FAILED = 1
    INTERRUPTED = 2


@define(kw_only=True)
class _SpeechData:
    source: str | allm.LLMStream | AsyncIterable[str]
    allow_interruptions: bool
    add_to_ctx: bool  # should this synthesis be added to the chat context
    int_ch: aio.Chan[_InterruptState]
    val_ch: aio.Chan[None]  # valide the speech for playout


@define(kw_only=True, frozen=True)
class _AssistantOptions:
    plotting: bool
    debug: bool
    allow_interruptions: bool
    int_speech_duration: float
    int_volume: float
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
        interrupt_volume: float = 0.025,
        interrupt_speech_duration: float = 2.0,
        base_volume: float = 0.8,
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
            int_volume=interrupt_volume,
            base_volume=base_volume,
        )
        self._vad, self._tts, self._llm, self._stt = vad, tts, llm, stt
        self._fnc_ctx, self._ctx = fnc_ctx, allm.ChatContext()
        self._speaking, self._user_speaking = False, False
        self._plotter = plotter.AssistantPlotter(self._loop)

        # synthesis states
        self._cur_speech: _SpeechData | None = None
        self._play_task: asyncio.Task | None = None

        self._user_speaking, self._agent_speaking = False, False

    @property
    def chat_context(self) -> allm.ChatContext:
        return self._ctx

    async def start(
        self, audio_stream: rtc.AudioStream, audio_source: rtc.AudioSource
    ) -> None:
        if self._opts.plotting:
            self._plotter.start()

        self._audio_source = audio_source
        self._audio_stream = audio_stream
        self._recognize_task = asyncio.create_task(self._recognize(audio_stream))

    async def aclose(self, wait: bool = True) -> None:
        if self._opts.plotting:
            self._plotter.terminate()

        if not wait:
            self._recognize_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await self._recognize_task

    async def say(
        self,
        source: str | allm.LLMStream | AsyncIterable[str],
        allow_interruptions: bool = True,
        add_to_ctx: bool = True,
    ) -> None:
        int_ch = aio.Chan[_InterruptState]()
        val_ch = aio.Chan[None]()
        data = _SpeechData(
            source=source,
            allow_interruptions=allow_interruptions,
            add_to_ctx=add_to_ctx,
            int_ch=int_ch,
            val_ch=val_ch,
        )

        val_ch.send_nowait(None)
        await self._start_speech(data)

    async def _recognize(self, audio_stream: rtc.AudioStream):
        # do VAD + ASR
        vad_stream = self._vad.stream()
        stt_stream = self._stt.stream()

        select = aio.select([audio_stream, vad_stream])
        async for s in select:
            if s.selected is audio_stream:
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
                    self._received_final_transcript(stt_event)

    def _received_final_transcript(self, ev: astt.SpeechEvent):
        if self._opts.debug:
            logger.debug(f"assistant - received transcript {ev.alternatives[0].text}")
            
        if self._cur_speech:
            self._cur_speech.int_ch.send_nowait(_InterruptState.VALIDATED)
        pass

    def _transcript_finished(self, ev: astt.SpeechEvent):
        # validate playout
        pass

    def _did_vad_inference(self, ev: avad.VADEvent):
        self._plotter.plot_value("vad_raw", ev.raw_inference_prob)
        self._plotter.plot_value("vad_smoothed", ev.probability)
        self._plotter.plot_value("vad_dur", ev.inference_duration * 1000)

    def _user_started_speaking(self):
        if self._opts.debug:
            logger.debug("assistant - user started speaking")

        self._user_speaking = True

        self._plotter.plot_event("user_started_speaking")
        if self._cur_speech:
            self._cur_speech.int_ch.send_nowait(_InterruptState.STARTED)

    def _user_stopped_speaking(self, speech_duration: float):
        if self._opts.debug:
            logger.debug(f"assistant - user stopped speaking {speech_duration:.2f}s")

        self._user_speaking = False

        self._plotter.plot_event("user_started_speaking")
        if self._cur_speech and speech_duration < self._opts.int_speech_duration:
            self._cur_speech.int_ch.send_nowait(_InterruptState.NONE)

    def _agent_started_speaking(self):
        if self._opts.debug:
            logger.debug("assistant - agent started speaking")

        self._agent_speaking = True

    def _agent_stopped_speaking(self):
        if self._opts.debug:
            logger.debug("assistant - agent stopped speaking")

        self._agent_speaking = False

    async def _start_speech(
        self,
        data: _SpeechData,
    ):
        # interrupt the current speech if possible and start the new one
        if self._play_task is not None:
            assert self._play_task is not None

            if self._cur_speech and self._cur_speech.allow_interruptions:
                self._cur_speech.int_ch.send_nowait(_InterruptState.VALIDATED)

            with contextlib.suppress(asyncio.CancelledError):
                await self._play_task

        if self._user_speaking and data.allow_interruptions:
            logger.debug("assistant - user is speaking, ignoring new speech request")
            return

        # start the new synthesis
        self._play_task = asyncio.create_task(self._maybe_play_speech(data))

    async def _maybe_play_speech(self, data: _SpeechData) -> None:
        if self._cur_speech is not None:
            raise RuntimeError("speech is already playing")  # this shouldn't happen

        if self._opts.debug:
            logger.debug(f"assistant - maybe_play_speech {data}")

        self._cur_speech = data

        po_tx, po_rx = aio.channel()
        text_tx, text_rx = aio.channel()
        tts_stream = self._tts.stream()

        tts_inference = _tts_inference(tts_stream, text_rx, po_tx, self._opts)
        buffer_playout = _buffer_playout(
            po_rx,
            data.int_ch,
            data.val_ch,
            self._agent_started_speaking,
            self._agent_stopped_speaking,
            self._opts,
            self._audio_source,
            self._plotter,
        )

        collected_text = ""

        async def _send_text():
            nonlocal collected_text
            with contextlib.closing(text_tx):
                if isinstance(data.source, str):
                    text_tx.send_nowait(data.source)
                    collected_text = data.source
                elif isinstance(data.source, allm.LLMStream):
                    async for chunk in data.source:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            text_tx.send_nowait(delta)
                            collected_text += delta
                elif isinstance(data.source, AsyncIterable):
                    async for text in data.source:
                        text_tx.send_nowait(text)
                        collected_text += text

        select = aio.select([tts_inference, buffer_playout, _send_text()])
        try:
            async with contextlib.aclosing(select):
                async for s in select:
                    s.result()

                    # buffer playout may have finished before tts_inference if the speech was
                    # interrupted. so break now
                    if s.selected is buffer_playout:
                        if s.result() == _PlayoutResult.INTERRUPTED:
                            logger.debug("assistant - buffer playout interruption done")
                        elif s.result() == _PlayoutResult.FINISHED and data.add_to_ctx:
                            new_msg = allm.ChatMessage(
                                text=collected_text,
                                role=allm.ChatRole.ASSISTANT,
                            )
                            self._ctx.messages.append(new_msg)
                        break

        except:
            logger.exception("error while playing speech")
        finally:
            logger.debug("assistant - maybe_play_speech finished")
            self._cur_speech = None


async def _tts_inference(
    stream: atts.SynthesizeStream,
    text_rx: aio.ChanReceiver[str],  # closing marks the end of segment
    po_tx: aio.ChanSender[rtc.AudioFrame],
    opts: _AssistantOptions,
):
    start_time = time.time()
    if opts.debug:
        logger.debug("tts inference started")

    select = aio.select([stream, text_rx])
    try:
        first_frame = True
        async with contextlib.aclosing(select):
            async for s in select:
                if s.selected is stream:
                    event: atts.SynthesisEvent = s.result()
                    if event.type == atts.SynthesisEventType.FINISHED:
                        break

                    if event.type != atts.SynthesisEventType.AUDIO:
                        continue

                    if first_frame and opts.debug:
                        dt = time.time() - start_time
                        logger.debug(f"tts first frame in {dt:.2f}s")
                        first_frame = False

                    assert event.audio
                    po_tx.send_nowait(event.audio.data)

                if s.selected is text_rx:
                    if s.exc:
                        stream.mark_segment_end()
                        continue

                    text: str = s.result()
                    stream.push_text(text)

    finally:
        po_tx.close()
        await stream.aclose()
        if opts.debug:
            dt = time.time() - start_time
            logger.debug(f"tts inference finished in {dt:.2f}s")


async def _buffer_playout(
    po_rx: aio.ChanReceiver[rtc.AudioFrame],
    int_rx: aio.ChanReceiver[_InterruptState],
    val_rx: aio.ChanReceiver[None],
    agent_started_speaking: Callable,
    agent_stopped_speaking: Callable,
    opts: _AssistantOptions,
    source: rtc.AudioSource,
    plotter: plotter.AssistantPlotter,
) -> _PlayoutResult:
    select = aio.select([int_rx, val_rx])
    async with contextlib.aclosing(select):
        async for s in select:
            if s.selected is val_rx:
                if opts.debug:
                    logger.debug("buffer playout validated")
                break  # start playout

            if s.selected is int_rx:
                state: _InterruptState = s.result()
                if state == _InterruptState.VALIDATED:
                    return _PlayoutResult.INTERRUPTED  # cancel playout

    t_vol = opts.base_volume  # target volume
    int_validated = False

    async def _capture():
        nonlocal t_vol, int_validated

        agent_started_speaking()

        # TODO: too overkill, we just use critical damping
        F, Z, R = 3.0, 1, 0
        K1 = Z / (pi * F)
        K2 = 1 / ((2 * pi * F) ** 2)
        K3 = R * Z / (2 * pi * F)

        sample_idx = 0
        xp, vol, yd = t_vol, t_vol, 0

        plot_task = None
        if opts.plotting:

            async def _plot_10ms():
                nonlocal t_vol, vol
                while True:
                    plotter.plot_value("raw_t_vol", t_vol)
                    plotter.plot_value("vol", vol)
                    await asyncio.sleep(0.01)

            plot_task = asyncio.create_task(_plot_10ms())

        dirty = False
        while not dirty:
            try:
                buf = await po_rx.recv()
                i = 0
                while i < len(buf.data):
                    eps = 1e-6
                    if int_validated and vol <= eps:
                        dirty = True
                        break

                    ms10 = buf.sample_rate // 100
                    rem = min(ms10, len(buf.data) - i)
                    data = buf.data[i : i + rem]
                    i += rem

                    # TODO(theomonnom): optimize using numpy?
                    dt = 1 / buf.sample_rate
                    for j in range(0, len(data)):
                        xd = (t_vol - xp) / dt
                        xp = t_vol

                        k2_s = max(K2, 1.1 * (dt**2 / 4 + dt * K1 / 2))
                        vol = vol + dt * yd
                        yd = yd + dt * (t_vol + K3 * xd - vol - K1 * yd) / k2_s

                        e = data[j] / 32768
                        data[j] = int(e * vol * 32768)
                        sample_idx += 1

                    frame = rtc.AudioFrame(
                        data=data.tobytes(),
                        sample_rate=buf.sample_rate,
                        num_channels=buf.num_channels,
                        samples_per_channel=rem,
                    )
                    await source.capture_frame(frame)

            except aio.ChanClosed:
                break

        agent_stopped_speaking()

        with contextlib.suppress(asyncio.CancelledError):
            if plot_task:
                plot_task.cancel()
                await plot_task

    try:
        cap = _capture()
        select = aio.select([cap, int_rx])
        while True:
            s = await select()
            if s.selected is int_rx:
                state: _InterruptState = s.result()
                if state == _InterruptState.NONE:
                    t_vol = opts.base_volume
                elif state == _InterruptState.STARTED:
                    t_vol = opts.base_volume * opts.int_volume
                elif state == _InterruptState.VALIDATED:
                    t_vol = 0
                    int_validated = True

            if s.selected is cap:
                s.result()
                break

        await select.aclose()
        logger.info("voice assistant playout finished")

        if int_validated:
            return _PlayoutResult.INTERRUPTED

        return _PlayoutResult.FINISHED
    except Exception:
        logger.exception("error playout buffer")

    return _PlayoutResult.FAILED
