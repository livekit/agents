from __future__ import annotations

import asyncio
import inspect
from typing import Any, AsyncIterable, Awaitable, Callable, Union

from livekit import rtc

from .. import llm, tokenize, utils
from .. import transcription as agent_transcription
from .. import tts as text_to_speech
from .agent_playout import AgentPlayout, PlayoutHandle
from .log import logger

SpeechSource = Union[AsyncIterable[str], str, Awaitable[str]]


class SynthesisHandle:
    def __init__(
        self,
        *,
        speech_id: str,
        tts_source: SpeechSource,
        transcript_source: SpeechSource,
        agent_playout: AgentPlayout,
        tts: text_to_speech.TTS,
        transcription_fwd: agent_transcription.TTSSegmentsForwarder,
    ) -> None:
        (
            self._tts_source,
            self._transcript_source,
            self._agent_playout,
            self._tts,
            self._tr_fwd,
        ) = (
            tts_source,
            transcript_source,
            agent_playout,
            tts,
            transcription_fwd,
        )
        self._buf_ch = utils.aio.Chan[rtc.AudioFrame]()
        self._play_handle: PlayoutHandle | None = None
        self._interrupt_fut = asyncio.Future[None]()
        self._speech_id = speech_id

    @property
    def speech_id(self) -> str:
        return self._speech_id

    @property
    def tts_forwarder(self) -> agent_transcription.TTSSegmentsForwarder:
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
            self._speech_id, self._buf_ch, transcription_fwd=self._tr_fwd
        )
        return self._play_handle

    def interrupt(self) -> None:
        """Interrupt the speech"""
        if self.interrupted:
            return

        logger.debug(
            "agent interrupted",
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
    ) -> None:
        self._room, self._agent_playout, self._llm, self._tts = (
            room,
            agent_playout,
            llm,
            tts,
        )
        self._tasks = set[asyncio.Task[Any]]()

    @property
    def playout(self) -> AgentPlayout:
        return self._agent_playout

    async def aclose(self) -> None:
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)

    def synthesize(
        self,
        *,
        speech_id: str,
        tts_source: SpeechSource,
        transcript_source: SpeechSource,
        transcription: bool,
        transcription_speed: float,
        sentence_tokenizer: tokenize.SentenceTokenizer,
        word_tokenizer: tokenize.WordTokenizer,
        hyphenate_word: Callable[[str], list[str]],
    ) -> SynthesisHandle:
        def _before_forward(
            fwd: agent_transcription.TTSSegmentsForwarder,
            transcription: rtc.Transcription,
        ):
            if not transcription:
                transcription.segments = []

            return transcription

        transcription_fwd = agent_transcription.TTSSegmentsForwarder(
            room=self._room,
            participant=self._room.local_participant,
            speed=transcription_speed,
            sentence_tokenizer=sentence_tokenizer,
            word_tokenizer=word_tokenizer,
            hyphenate_word=hyphenate_word,
            before_forward_cb=_before_forward,
        )

        handle = SynthesisHandle(
            tts_source=tts_source,
            transcript_source=transcript_source,
            agent_playout=self._agent_playout,
            tts=self._tts,
            transcription_fwd=transcription_fwd,
            speech_id=speech_id,
        )

        task = asyncio.create_task(self._synthesize_task(handle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.remove)
        return handle

    @utils.log_exceptions(logger=logger)
    async def _synthesize_task(self, handle: SynthesisHandle) -> None:
        """Synthesize speech from the source"""
        tts_source = handle._tts_source
        transcript_source = handle._transcript_source

        if isinstance(tts_source, Awaitable):
            tts_source = await tts_source
        if isinstance(transcript_source, Awaitable):
            transcript_source = await transcript_source

        if isinstance(tts_source, str):
            co = self._str_synthesis_task(tts_source, transcript_source, handle)
        else:
            co = self._stream_synthesis_task(tts_source, transcript_source, handle)

        synth = asyncio.create_task(co)
        synth.add_done_callback(lambda _: handle._buf_ch.close())
        try:
            _ = await asyncio.wait(
                [synth, handle._interrupt_fut], return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            await utils.aio.gracefully_cancel(synth)

    @utils.log_exceptions(logger=logger)
    async def _read_transcript_task(
        self, transcript_source: AsyncIterable[str] | str, handle: SynthesisHandle
    ) -> None:
        try:
            if isinstance(transcript_source, str):
                handle._tr_fwd.push_text(transcript_source)
            else:
                async for seg in transcript_source:
                    if not handle._tr_fwd.closed:
                        handle._tr_fwd.push_text(seg)

            if not handle.tts_forwarder.closed:
                handle.tts_forwarder.mark_text_segment_end()
        finally:
            if inspect.isasyncgen(transcript_source):
                await transcript_source.aclose()

    @utils.log_exceptions(logger=logger)
    async def _str_synthesis_task(
        self,
        tts_text: str,
        transcript_source: AsyncIterable[str] | str,
        handle: SynthesisHandle,
    ) -> None:
        """synthesize speech from a string"""
        read_transcript_atask: asyncio.Task | None = None

        first_frame = True
        tts_stream = handle._tts.synthesize(tts_text)
        try:
            async for audio in tts_stream:
                if first_frame:
                    first_frame = False
                    read_transcript_atask = asyncio.create_task(
                        self._read_transcript_task(transcript_source, handle)
                    )

                handle._buf_ch.send_nowait(audio.frame)
                if not handle.tts_forwarder.closed:
                    handle.tts_forwarder.push_audio(audio.frame)

            if not handle.tts_forwarder.closed:
                handle.tts_forwarder.mark_audio_segment_end()

            if read_transcript_atask is not None:
                await read_transcript_atask
        finally:
            await tts_stream.aclose()

            if read_transcript_atask is not None:
                await utils.aio.gracefully_cancel(read_transcript_atask)

    @utils.log_exceptions(logger=logger)
    async def _stream_synthesis_task(
        self,
        tts_source: AsyncIterable[str],
        transcript_source: AsyncIterable[str] | str,
        handle: SynthesisHandle,
    ) -> None:
        """synthesize speech from streamed text"""

        @utils.log_exceptions(logger=logger)
        async def _read_generated_audio_task(
            tts_stream: text_to_speech.SynthesizeStream,
        ) -> None:
            try:
                async for audio in tts_stream:
                    if not handle._tr_fwd.closed:
                        handle._tr_fwd.push_audio(audio.frame)

                    handle._buf_ch.send_nowait(audio.frame)
            finally:
                if handle._tr_fwd and not handle._tr_fwd.closed:
                    handle._tr_fwd.mark_audio_segment_end()

                await tts_stream.aclose()

        tts_stream: text_to_speech.SynthesizeStream | None = None
        read_tts_atask: asyncio.Task | None = None
        read_transcript_atask: asyncio.Task | None = None

        try:
            async for seg in tts_source:
                if tts_stream is None:
                    tts_stream = handle._tts.stream()
                    read_tts_atask = asyncio.create_task(
                        _read_generated_audio_task(tts_stream)
                    )
                    read_transcript_atask = asyncio.create_task(
                        self._read_transcript_task(transcript_source, handle)
                    )

                tts_stream.push_text(seg)

            if tts_stream is not None:
                tts_stream.end_input()
                assert read_transcript_atask and read_tts_atask
                await read_tts_atask
                await read_transcript_atask

        finally:
            if read_tts_atask is not None:
                assert read_transcript_atask is not None
                await utils.aio.gracefully_cancel(read_tts_atask, read_transcript_atask)

            if inspect.isasyncgen(tts_source):
                await tts_source.aclose()
