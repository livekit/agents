from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass
from typing import Any, Literal

from livekit import rtc

from ... import stt
from ...log import logger
from ...types import APIConnectOptions
from ...utils import aio
from ._fsm import _Transcript

_Source = Literal["session", "amd"]
_MAX_TEXT = 16000


@dataclass(frozen=True)
class TurnTranscript:
    """A committed turn's transcript, ready for classification."""

    turn_id: int
    transcript: _Transcript


class _TurnText:
    """Race the session transcript against the AMD transcript for one turn."""

    def __init__(self) -> None:
        self.ready = asyncio.Event()
        self.winner: _Source | None = None
        self._texts: dict[_Source, str] = {"session": "", "amd": ""}

    @property
    def text(self) -> str:
        return self._texts[self.winner] if self.winner else ""

    def snapshot(self) -> _Transcript:
        other = self._texts["amd" if self.winner == "session" else "session"]
        return _Transcript(self.text, self.winner, other)

    def add(self, source: _Source, text: str, *, replace: bool = False) -> None:
        text = text.strip()
        if not text:
            return
        joined = text if replace else f"{self._texts[source]} {text}".strip()
        self._texts[source] = joined[-_MAX_TEXT:]
        if self.winner is None:
            self.winner = source
            self.ready.set()


class RacingSTT:
    """One STT for AMD. The session transcript races an optional AMD model per turn.

    Audio and session finals go in. One ``TurnTranscript`` per committed turn
    comes out, when the first text lands or the grace period expires. The model
    keeps one stream open for the run and each EOT flushes it. AMD finals belong
    to the newest committed turn that is still waiting for text; anything later
    belongs to the next turn, as with the session's own STT.
    """

    def __init__(
        self,
        model: stt.STT | None,
        conn_options: APIConnectOptions,
        *,
        grace_period: float,
    ) -> None:
        self._model = model
        self._conn_options = conn_options
        self._grace_period = grace_period
        self._current = _TurnText()
        self._waiting: dict[int, _TurnText] = {}
        self._pending: tuple[int, _TurnText] | None = None
        self._events = aio.Chan[TurnTranscript]()
        self._stream: stt.RecognizeStream | None = None
        self._reading = False
        self._tasks: set[asyncio.Task[None]] = set()
        self._failed = False

    @property
    def model_active(self) -> bool:
        """Whether the AMD model can still contribute text. Session text always can."""
        return self._model is not None and not self._failed

    @property
    def current_text(self) -> str:
        return self._current.text

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        model = self._model
        if model is None or self._failed:
            return
        try:
            if self._stream is None:
                self._stream = model.stream(conn_options=self._conn_options)
                self._spawn(self._read(self._stream))
            self._stream.push_frame(frame)
        except Exception as exc:
            self._fail(exc)

    def push_session_text(self, text: str) -> None:
        self._current.add("session", text)

    def end_turn(self, turn_id: int, session_text: str) -> _Transcript:
        """Commit the open turn at client-side EOT and return its transcript so far."""
        turn = self._current
        turn.add("session", session_text, replace=True)
        self._current = _TurnText()
        try:
            if self._stream is not None and self.model_active:
                self._stream.flush()
        except Exception as exc:
            self._fail(exc)
        if turn.ready.is_set() or not self.model_active:
            self._pending = None
            self._events.send_nowait(TurnTranscript(turn_id, turn.snapshot()))
        else:
            self._waiting[turn_id] = turn
            self._pending = (turn_id, turn)
            self._spawn(self._await_ready(turn_id, turn))
        return turn.snapshot()

    def __aiter__(self) -> AsyncIterator[TurnTranscript]:
        return self._events

    async def aclose(self) -> None:
        await aio.cancel_and_wait(*self._tasks)
        if self._stream is not None and not self._reading:
            await self._stream.aclose()
        self._events.close()

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _await_ready(self, turn_id: int, turn: _TurnText) -> None:
        try:
            await asyncio.wait_for(turn.ready.wait(), self._grace_period)
        except asyncio.TimeoutError:
            pass
        self._emit(turn_id)

    def _emit(self, turn_id: int) -> None:
        turn = self._waiting.pop(turn_id, None)
        if turn is None:
            return
        if self._pending is not None and self._pending[0] == turn_id:
            self._pending = None
        self._events.send_nowait(TurnTranscript(turn_id, turn.snapshot()))

    async def _read(self, stream: stt.RecognizeStream) -> None:
        self._reading = True
        try:
            async with stream:
                async for event in stream:
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
                        target = self._pending[1] if self._pending else self._current
                        target.add("amd", event.alternatives[0].text)
        except Exception as exc:
            self._fail(exc)

    def _fail(self, exc: Exception) -> None:
        """Continue with the session transcript only. Turns waiting for AMD text stop waiting."""
        self._failed = True
        logger.warning("amd stt failed", extra={"error_type": type(exc).__name__})
        for turn_id in list(self._waiting):
            self._emit(turn_id)
