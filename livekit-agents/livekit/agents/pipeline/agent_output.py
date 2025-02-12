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

    def create_synthesis_handle(
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
            rtc_transcription: rtc.Transcription,
        ):
            if not transcription:
                rtc_transcription.segments = []

            return rtc_transcription

        transcription_fwd = agent_transcription.TTSSegmentsForwarder(
            room=self._room,
            participant=self._room.local_participant,
            speed=transcription_speed,
            sentence_tokenizer=sentence_tokenizer,
            word_tokenizer=word_tokenizer,
            hyphenate_word=hyphenate_word,
            before_forward_cb=_before_forward,
        )

        return SynthesisHandle(
            tts_source=tts_source,
            transcript_source=transcript_source,
            agent_playout=self._agent_playout,
            tts=self._tts,
            transcription_fwd=transcription_fwd,
            speech_id=speech_id,
        )

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
