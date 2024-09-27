from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterable, Awaitable, Optional, Union

from livekit import rtc

from .. import llm, utils
from .. import stf as speech_to_face
from .. import tts as text_to_speech
from ..transcription import AssistantTranscriptionOptions, TTSSegmentsForwarder
from .agent_playout import AgentPlayout, PlayoutHandle
from .log import logger

SpeechSource = Union[AsyncIterable[str], str, Awaitable[str]]


class SynthesisHandle:
    def __init__(
        self,
        *,
        speech_id: str,
        tts_source: SpeechSource,
        agent_playout: AgentPlayout,
        transcription_fwd: TTSSegmentsForwarder,
        tts: text_to_speech.TTS,
        stf: Optional[speech_to_face.STF],
    ) -> None:
        (
            self._speech_id,
            self._tts_source,
            self._agent_playout,
            self._tr_fwd,
            self._tts,
            self._stf,
        ) = (
            speech_id,
            tts_source,
            agent_playout,
            transcription_fwd,
            tts,
            stf,
        )
        self._play_handle: PlayoutHandle | None = None
        self._interrupt_fut = asyncio.Future[None]()

        self._audio_buf_ch = utils.aio.Chan[rtc.AudioFrame]()
        self._video_buf_ch = utils.aio.Chan[rtc.VideoFrameEvent]() if stf else None

    @property
    def speech_id(self) -> str:
        return self._speech_id

    @property
    def tts_forwarder(self) -> TTSSegmentsForwarder:
        return self._tr_fwd

    @property
    def validated(self) -> bool:
        return self._play_handle is not None

    @property
    def interrupted(self) -> bool:
        return self._interrupt_fut.done()

    @property
    def play_handle(self) -> PlayoutHandle | None:
        return self._play_handle

    def play(self) -> PlayoutHandle:
        """Validate the speech for playout"""
        if self.interrupted:
            raise RuntimeError("synthesis was interrupted")

        self._play_handle = self._agent_playout.play(
            speech_id=self._speech_id,
            transcription_fwd=self._tr_fwd,
            audio_playout_source=self._audio_buf_ch,
            video_playout_source=self._video_buf_ch,
        )
        return self._play_handle

    def interrupt(self) -> None:
        """Interrupt the speech"""
        if self.interrupted:
            return

        logger.debug(
            "interrupting synthesis/playout",
            extra={"speech_id": self.speech_id},
        )

        if self._play_handle is not None:
            self._play_handle.interrupt()

        self._interrupt_fut.set_result(None)


class AgentOutput:
    def __init__(
        self,
        *,
        room: rtc.Room,
        agent_playout: AgentPlayout,
        llm: llm.LLM,
        tts: text_to_speech.TTS,
        stf: Optional[speech_to_face.STF],
        transcription: AssistantTranscriptionOptions,
    ) -> None:
        (
            self._room,
            self._agent_playout,
            self._llm,
            self._tts,
            self._stf,
            self._transcription,
        ) = (
            room,
            agent_playout,
            llm,
            tts,
            stf,
            transcription,
        )
        self._tasks = set[asyncio.Task[Any]]()

    @property
    def playout(self) -> AgentPlayout:
        return self._agent_playout

    async def aclose(self) -> None:
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)

    def synthesize(self, *, speech_id: str, tts_source: SpeechSource) -> SynthesisHandle:
        """
        Fills _audio_buf_ch with audio frames and _tr_fwd with transcription segments.
        """
        def _before_forward(
            fwd: TTSSegmentsForwarder,
            transcription: rtc.Transcription,
        ):
            if not transcription:
                transcription.segments = []

            return transcription

        # TODO: add alignment option
        transcription_fwd = TTSSegmentsForwarder(
            room=self._room,
            participant=self._room.local_participant,
            speed=self._transcription.agent_transcription_speed,
            sentence_tokenizer=self._transcription.sentence_tokenizer,
            word_tokenizer=self._transcription.word_tokenizer,
            hyphenate_word=self._transcription.hyphenate_word,
            before_forward_cb=_before_forward,
        )

        handle = SynthesisHandle(
            speech_id=speech_id,
            tts_source=tts_source,
            agent_playout=self._agent_playout,
            transcription_fwd=transcription_fwd,
            tts=self._tts,
            stf=self._stf,
        )

        task = asyncio.create_task(self._synthesize_task(handle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.remove)
        return handle

    @utils.log_exceptions(logger=logger)
    async def _synthesize_task(self, handle: SynthesisHandle) -> None:
        """Synthesize speech from the source"""
        tts_source = handle._tts_source

        if isinstance(tts_source, Awaitable):
            tts_source = await tts_source
            co = _str_synthesis_task(tts_source, handle)
        elif isinstance(tts_source, str):
            co = _str_synthesis_task(tts_source, handle)
        else:
            co = _stream_synthesis_task(tts_source, handle)

        synth = asyncio.create_task(co)
        synth.add_done_callback(lambda _: handle._audio_buf_ch.close())
        try:
            _ = await asyncio.wait(
                [synth, handle._interrupt_fut], return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            await utils.aio.gracefully_cancel(synth)


@utils.log_exceptions(logger=logger)
async def _str_synthesis_task(tts_text: str, handle: SynthesisHandle) -> None:
    """synthesize speech from a string"""
    if not handle.tts_forwarder.closed:
        handle.tts_forwarder.push_text(tts_text)
        handle.tts_forwarder.mark_text_segment_end()

    start_time = time.time()
    first_frame = True

    # TODO: add face synthesis
    try:
        async for audio in handle._tts.synthesize(tts_text):
            if first_frame:
                first_frame = False
                logger.debug(
                    "received first TTS frame",
                    extra={
                        "speech_id": handle.speech_id,
                        "elapsed": round(time.time() - start_time, 3),
                        "streamed": False,
                    },
                )

            frame = audio.frame

            handle._audio_buf_ch.send_nowait(frame)
            if not handle.tts_forwarder.closed:
                handle.tts_forwarder.push_audio(frame)

    finally:
        if not handle.tts_forwarder.closed:
            handle.tts_forwarder.mark_audio_segment_end()


@utils.log_exceptions(logger=logger)
async def _stream_synthesis_task(tts_source: AsyncIterable[str], handle: SynthesisHandle) -> None:
    """synthesize speech and face from streamed text"""
    tts_source, transcript_source = utils.aio.itertools.tee(tts_source, 2)
    tts_stream = handle._tts.stream()
    stf_stream = handle._stf.speech_stream() if handle._stf else None

    @utils.log_exceptions(logger=logger)
    async def _read_synthesized_audio_task():
        start_time = time.time()
        first_frame = True
        async for audio in tts_stream:
            if first_frame:
                first_frame = False
                logger.debug(
                    "first TTS frame",
                    extra={
                        "speech_id": handle.speech_id,
                        "elapsed": round(time.time() - start_time, 3),
                        "streamed": True,
                    },
                )

            if stf_stream:
                stf_stream.push_audio(audio)

            if not handle._tr_fwd.closed:
                handle._tr_fwd.push_audio(audio.frame) # TODO: add alignment

            handle._audio_buf_ch.send_nowait(audio.frame)

        if handle._tr_fwd and not handle._tr_fwd.closed:
            handle._tr_fwd.mark_audio_segment_end()

    @utils.log_exceptions(logger=logger)
    async def _read_transcript_task():
        # TODO: this is weird to syncronize the tts and transcript streams here
        # I'd sugest moving this syncronisation to the agent_playout, but this is a design decision
        # We either want transcription to apear in the chat ASAP or in sync with the audio
        async for seg in transcript_source:
            if not handle._tr_fwd.closed:
                handle._tr_fwd.push_text(seg)

        if not handle.tts_forwarder.closed:
            handle.tts_forwarder.mark_text_segment_end()

    @utils.log_exceptions(logger=logger)
    async def _read_synthesized_video_task():
        start_time = time.time()
        first_frame = True
        async for ev in stf_stream:
            if first_frame:
                first_frame = False
                logger.debug(
                    "first STF frame",
                    extra={
                        "speech_id": handle.speech_id,
                        "elapsed": round(time.time() - start_time, 3),
                        "streamed": True,
                    },
                )
            handle._video_buf_ch.send_nowait(ev.frame)

    tasks: list[asyncio.Task] = []
    try:
        async for seg in tts_source:
            if not tasks:
                # start the task when we receive the first text segment (so start_time is more accurate)
                tasks = [
                    asyncio.create_task(_read_synthesized_audio_task()),
                    asyncio.create_task(_read_transcript_task()),
                ]
                if stf_stream:
                    tasks.append(asyncio.create_task(_read_synthesized_video_task()))

            tts_stream.push_text(seg)
            # TODO: support stf from text

        tts_stream.end_input()
        if stf_stream:
            stf_stream.end_input()

        if tasks:
            await asyncio.gather(*tasks)

    finally:
        for task in tasks:
            await utils.aio.gracefully_cancel(task)

        await tts_stream.aclose()
        if stf_stream:
            await stf_stream.aclose()
