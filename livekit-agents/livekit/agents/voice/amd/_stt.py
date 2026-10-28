"""Attribute session and AMD STT transcripts to committed turns without access to AgentSession."""

from __future__ import annotations

import asyncio

from livekit import rtc

from ... import stt
from ...log import logger
from ...types import APIConnectOptions
from ...utils import aio
from ._chat_context import AMDTranscript, AMDTranscriptSource


class AMDTurnTranscriptAccumulator:
    """Buffer a turn's session and AMD transcripts."""

    def __init__(self) -> None:
        self._transcripts: dict[AMDTranscriptSource, str] = {"session": "", "amd": ""}

    def get(self, source: AMDTranscriptSource) -> str:
        return self._transcripts[source]

    def add(self, source: AMDTranscriptSource, transcript: str, *, replace: bool = False) -> None:
        transcript = transcript.strip()
        if not transcript:
            return
        self._transcripts[source] = (
            transcript if replace else f"{self._transcripts[source]} {transcript}".strip()
        )


class AMDTranscriber:
    """Collect each turn's session transcript and, with an AMD STT, its own transcript too.

    Neither source wins: the classifier sees both, so a late or missing transcript from
    one STT does not empty the turn. Without session STT, AMD uses its own transcript
    alone. If the AMD stream fails, turns fall back to the session transcript alone.
    """

    def __init__(
        self,
        model: stt.STT | None,
        conn_options: APIConnectOptions,
        *,
        session_stt: bool = True,
    ) -> None:
        self._model = model
        self._conn_options = conn_options
        self._session_enabled = session_stt or model is None
        self._amd_enabled = model is not None
        self._current = AMDTurnTranscriptAccumulator()
        # the turn just ended before its AMD final arrived: the AMD STT finalizes a sentence
        # a little after the session commits it, so a final within the wait belongs to it
        self._late: AMDTurnTranscriptAccumulator | None = None
        self._late_until = 0.0
        self._late_final = asyncio.Event()
        self._stream: stt.RecognizeStream | None = None

        self._reading = False
        self._stream_read_task: asyncio.Task[None] | None = None
        self._stream_close_task: asyncio.Task[None] | None = None

    @property
    def amd_stt_active(self) -> bool:
        """Whether the optional AMD stream can still receive audio."""
        return self._amd_enabled

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
                        late = self._late is not None
                        if late and asyncio.get_running_loop().time() > self._late_until:
                            self._late = None
                        if self._late is not None:
                            self._late.add("amd", transcript)
                            self._late = None
                            self._late_final.set()
                        else:
                            self.push_transcript(transcript=transcript, source="amd")
        except Exception as exc:
            self.fail(exc)

    def push_transcript(
        self, transcript: str, *, replace: bool = False, source: AMDTranscriptSource = "session"
    ) -> None:
        self._current.add(source, transcript, replace=replace)

    def end_turn(self, session_transcript: str, *, late_wait: float = 0.0) -> AMDTranscript:
        """End the turn. Within `late_wait` seconds, the next AMD final still belongs to it
        when it has no AMD text yet."""
        self.push_transcript(session_transcript, replace=True, source="session")
        turn, self._current = self._current, AMDTurnTranscriptAccumulator()
        # without session STT the turn has no text until its AMD final arrives
        late = (
            late_wait > 0
            and self.amd_stt_active
            and not turn.get("amd")
            and (bool(turn.get("session")) or not self._session_enabled)
        )
        self._late = turn if late else None
        self._late_final = asyncio.Event()
        if late:
            self._late_until = asyncio.get_running_loop().time() + late_wait
        return AMDTranscript(
            session=turn.get("session") if self._session_enabled else None,
            amd=turn.get("amd") if self._amd_enabled else None,
        )

    @property
    def awaits_late_final(self) -> bool:
        """Whether the turn just ended still waits for its AMD final."""
        return self._late is not None

    async def late_amd_transcript(self) -> str | None:
        """The AMD transcript of the turn just ended, once its final arrives or the wait
        passes. None when that turn needs no wait."""
        late, arrived = self._late, self._late_final
        if late is None:
            return None
        remaining = self._late_until - asyncio.get_running_loop().time()
        try:
            await asyncio.wait_for(arrived.wait(), max(0.0, remaining))
        except asyncio.TimeoutError:
            pass
        if self._late is late:
            self._late = None
        return late.get("amd") or None

    def close_stream(self) -> None:
        self._amd_enabled = False
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
        """Use the session transcript alone for the open turn and subsequent turns."""
        if not self.amd_stt_active:
            return
        logger.warning(
            "amd stt failed, falling back to session stt", extra={"error_type": type(exc).__name__}
        )
        self._session_enabled = True
        self.close_stream()
