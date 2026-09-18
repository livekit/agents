"""Attribute racing STT transcripts to committed turns without access to AgentSession."""

from __future__ import annotations

import asyncio

from livekit import rtc

from ... import stt
from ...log import logger
from ...types import APIConnectOptions
from ...utils import aio
from ._chat_context import AMDTranscript, AMDTranscriptSource


class AMDTurnTranscriptAccumulator:
    """Buffer a turn's session and AMD transcripts so either source can be selected later."""

    def __init__(self) -> None:
        self._transcripts: dict[AMDTranscriptSource | None, str] = {"session": "", "amd": ""}

    def snapshot(self, source: AMDTranscriptSource | None) -> AMDTranscript:
        return AMDTranscript(self._transcripts.get(source, ""), source)

    def add(self, source: AMDTranscriptSource, transcript: str, *, replace: bool = False) -> None:
        transcript = transcript.strip()
        if not transcript:
            return
        self._transcripts[source] = (
            transcript if replace else f"{self._transcripts[source]} {transcript}".strip()
        )


class AMDRacingSTT:
    """Select the fastest STT or fall back to session STT when AMD STT fails."""

    def __init__(
        self,
        model: stt.STT | None,
        conn_options: APIConnectOptions,
        *,
        race_session: bool = True,
    ) -> None:
        self._model = model
        self._conn_options = conn_options
        self._source: AMDTranscriptSource | None = "session" if model is None else None
        if model is not None and not race_session:
            self._source = "amd"
        self._current = AMDTurnTranscriptAccumulator()
        self._stream: stt.RecognizeStream | None = None

        self._reading = False
        self._stream_read_task: asyncio.Task[None] | None = None
        self._stream_close_task: asyncio.Task[None] | None = None

    @property
    def amd_stt_active(self) -> bool:
        """Whether the optional AMD stream can still receive audio."""
        return self._source != "session"  # already implies self._model is not None

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if not self.amd_stt_active:
            return
        try:
            if self._stream is None:
                assert self._model is not None
                self._stream = self._model.stream(conn_options=self._conn_options)
                self._stream_read_task = asyncio.create_task(self._read(self._stream))
            self._stream.push_frame(frame)
        except Exception as exc:
            self.fail(exc)

    async def _read(self, stream: stt.RecognizeStream) -> None:
        self._reading = True
        try:
            async with stream:
                async for event in stream:
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
                        transcript = event.alternatives[0].text.strip()
                        self.push_transcript(transcript=transcript, source="amd")
        except Exception as exc:
            self.fail(exc)

    def push_transcript(
        self, transcript: str, *, replace: bool = False, source: AMDTranscriptSource = "session"
    ) -> None:
        if not transcript.strip():
            return

        if source == "session":
            self._current.add("session", transcript, replace=replace)
            if self._source is None:
                self.close_stream()
            return

        if self._source == "session":
            return

        self._current.add("amd", transcript, replace=replace)
        self._source = "amd"

    def end_turn(self, session_transcript: str) -> AMDTranscript:
        self.push_transcript(session_transcript, replace=True, source="session")
        turn, self._current = self._current, AMDTurnTranscriptAccumulator()
        return turn.snapshot(self._source)

    def close_stream(self) -> None:
        self._source = "session"
        if self._stream is None:
            return
        stream, self._stream = self._stream, None

        async def close_stream() -> None:
            if self._stream_read_task is not None:
                await aio.cancel_and_wait(self._stream_read_task)
            # A reader cancelled before starting never enters the stream context.
            if not self._reading:
                await stream.aclose()

        self._stream_close_task = asyncio.create_task(close_stream())

    async def aclose(self) -> None:
        self.close_stream()
        if self._stream_close_task is not None:
            await asyncio.shield(self._stream_close_task)

    def fail(self, exc: Exception) -> None:
        """Use the buffered session transcript for the open turn and subsequent turns."""
        if not self.amd_stt_active:
            return
        logger.warning(
            "amd stt failed, falling back to session stt", extra={"error_type": type(exc).__name__}
        )
        self.close_stream()
