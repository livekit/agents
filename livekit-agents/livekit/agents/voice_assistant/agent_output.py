from __future__ import annotations

import asyncio
import contextlib
from typing import AsyncIterable, Union

from livekit import rtc

from .. import aio, transcription, utils
from .. import llm as llm
from .. import tts as text_to_speech
from .cancellable_source import CancellableAudioSource, PlayoutHandle
from .log import logger

SpeechSource = Union[AsyncIterable[str], str]


class SynthesisHandle:
    def __init__(
        self,
        *,
        speech_source: SpeechSource,
        audio_source: CancellableAudioSource,
        tts: text_to_speech.TTS,
        transcription_fwd: transcription.TTSSegmentsForwarder | None = None,
    ) -> None:
        self._speech_source, self._audio_source, self._tts, self._tr_fwd = (
            speech_source,
            audio_source,
            tts,
            transcription_fwd,
        )
        self._buf_ch = aio.Chan[rtc.AudioFrame]()
        self._play_handle: PlayoutHandle | None = None
        self._interrupt_fut = asyncio.Future()
        self._collected_text = ""  # collected text from the async stream

    @property
    def validated(self) -> bool:
        return self._play_handle is not None

    @property
    def interrupted(self) -> bool:
        return self._interrupt_fut.done()

    @property
    def collected_text(self) -> str:
        return self._collected_text

    @property
    def play_handle(self) -> PlayoutHandle | None:
        return self._play_handle

    def play(self) -> PlayoutHandle:
        """Validate the speech for playout"""
        if self.interrupted:
            raise RuntimeError("synthesis was interrupted")

        self._play_handle = self._audio_source.play(
            self._buf_ch,
        )
        return self._play_handle

    def interrupt(self) -> None:
        """Interrupt the speech"""
        if self._play_handle is not None:
            self._play_handle.interrupt()

        self._interrupt_fut.set_result(None)


class AgentOutput:
    def __init__(
        self,
        *,
        room: rtc.Room,
        source: CancellableAudioSource,
        llm: llm.LLM,
        tts: text_to_speech.TTS,
    ) -> None:
        self._room, self._source, self._llm, self._tts = room, source, llm, tts
        self._tasks = set()

    async def aclose(self) -> None:
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)

    def synthesize(self, *, transcript: SpeechSource) -> SynthesisHandle:
        transcription_fwd = transcription.TTSSegmentsForwarder(
            room=self._room, participant=self._room.local_participant
        )

        handle = SynthesisHandle(
            speech_source=transcript,
            audio_source=self._source,
            tts=self._tts,
            transcription_fwd=transcription_fwd,
        )

        task = asyncio.create_task(
            self._synthesize_task(
                handle,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.remove)
        return handle

    @utils.log_exceptions(logger=logger)
    async def _synthesize_task(
        self,
        handle: SynthesisHandle,
    ) -> None:
        """Synthesize speech from the source"""
        if isinstance(handle._speech_source, str):
            co = _str_synthesis_co(handle._speech_source, handle)
        else:
            co = _stream_synthesis_co(handle._speech_source, handle)

        synth = asyncio.create_task(co)
        try:
            _ = await asyncio.wait(
                [synth, handle._interrupt_fut], return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            synth.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await synth


async def _str_synthesis_co(
    text: str,
    handle: SynthesisHandle,
) -> None:
    """synthesize speech from a string"""
    if handle._tr_fwd is not None:
        handle._tr_fwd.push_text(text)
        handle._tr_fwd.mark_text_segment_end()

    # start_time = time.time()
    # first_frame = True
    # audio_duration = 0.0
    handle._collected_text = text

    try:
        async for audio in handle._tts.synthesize(text):
            # if first_frame:
            # first_frame = False
            # dt = time.time() - start_time
            # self._log_debug(f"tts first frame in {dt:.2f}s")

            frame = audio.data
            # audio_duration += frame.samples_per_channel / frame.sample_rate

            handle._buf_ch.send_nowait(frame)
            if handle._tr_fwd is not None:
                handle._tr_fwd.push_audio(frame)

    finally:
        if handle._tr_fwd is not None:
            handle._tr_fwd.mark_audio_segment_end()
        handle._buf_ch.close()
        # self._log_debug(f"tts finished synthesising {audio_duration:.2f}s of audio")


async def _stream_synthesis_co(
    streamed_text: AsyncIterable[str],
    handle: SynthesisHandle,
) -> None:
    """synthesize speech from streamed text"""

    async def _read_generated_audio_task():
        # start_time = time.time()
        # first_frame = True
        # audio_duration = 0.0
        async for event in tts_stream:
            if event.type == text_to_speech.SynthesisEventType.AUDIO:
                # if first_frame:
                #    first_frame = False
                #    dt = time.time() - start_time
                #    self._log_debug(f"tts first frame in {dt:.2f}s (streamed)")

                assert event.audio is not None
                frame = event.audio.data
                # audio_duration += frame.samples_per_channel / frame.sample_rate
                if handle._tr_fwd is not None:
                    handle._tr_fwd.push_audio(frame)
                handle._buf_ch.send_nowait(frame)

        # self._log_debug(
        #    f"tts finished synthesising {audio_duration:.2f}s audio (streamed)"
        # )

    # otherwise, stream the text to the TTS
    tts_stream = handle._tts.stream()
    read_atask = asyncio.create_task(_read_generated_audio_task())

    try:
        async for seg in streamed_text:
            handle._collected_text = seg
            if handle._tr_fwd is not None:
                handle._tr_fwd.push_text(seg)

            tts_stream.push_text(seg)

    finally:
        if handle._tr_fwd is not None:
            handle._tr_fwd.mark_text_segment_end()

        tts_stream.mark_segment_end()
        await tts_stream.aclose()
        await read_atask

        if handle._tr_fwd is not None:
            handle._tr_fwd.mark_audio_segment_end()

        handle._buf_ch.close()
