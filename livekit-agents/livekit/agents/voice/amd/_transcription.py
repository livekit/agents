from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Literal

from livekit import rtc

from ... import stt
from ...log import logger
from ...types import APIConnectOptions
from ...utils import aio


class TurnTranscript:
    """Race final transcripts without changing the Agent's STT input.

    Each optional STT stream ends at client-side EOT. Its reader can drain
    independently, so late finals cannot be assigned to another turn.
    """

    def __init__(
        self,
        model: stt.STT | None,
        conn_options: APIConnectOptions,
        on_update: Callable[[TurnTranscript], None],
    ) -> None:
        self.turn_id: int | None = None
        self.ready = asyncio.Event()
        self.winner: Literal["session", "amd"] | None = None
        self._texts = {"session": "", "amd": ""}
        self._model = model
        self._conn_options = conn_options
        self._on_update = on_update
        self._stream: stt.RecognizeStream | None = None
        self._frames: list[rtc.AudioFrame] = []
        self._task: asyncio.Task[None] | None = None
        self._reading = False
        self._drain_timer: asyncio.TimerHandle | None = None
        self._failed = False

    @property
    def text(self) -> str:
        return self._texts[self.winner] if self.winner else ""

    @property
    def pending(self) -> bool:
        return self._task is not None and not self._task.done()

    def add_session_text(self, text: str, *, replace: bool = False) -> None:
        if self._failed and text.strip():
            self.winner = "session"
        self._add_text("session", text, replace=replace)

    def _add_text(
        self, source: Literal["session", "amd"], text: str, *, replace: bool = False
    ) -> None:
        text = text.strip()
        if not text:
            return
        self._texts[source] = (text if replace else f"{self._texts[source]} {text}".strip())[
            -16000:
        ]
        if self.winner is None:
            self.winner = source
            self.ready.set()
        if self.turn_id is not None:
            self._on_update(self)

    def history(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "turn_id": self.turn_id,
            "transcript": self.text,
            "transcript_source": self.winner,
        }
        other = self._texts["amd" if self.winner == "session" else "session"]
        if other and other != self.text:
            entry["alternative_transcript"] = other
        return entry

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._model is None or self._failed:
            return
        if not self._model.capabilities.streaming:
            self._frames.append(frame)
            return
        try:
            if self._stream is None:
                self._stream = self._model.stream(conn_options=self._conn_options)
                self._task = asyncio.create_task(self._recognize())
            self._stream.push_frame(frame)
        except Exception as exc:
            self._fail(exc)

    def commit(self, turn_id: int, session_text: str) -> None:
        self.add_session_text(session_text, replace=True)
        self.turn_id = turn_id
        try:
            if self._stream is not None and not self._failed:
                self._stream.end_input()
            elif self._frames:
                self._task = asyncio.create_task(self._recognize())
        except Exception as exc:
            self._fail(exc)
        if self.pending:
            # Bound the losing request without discarding text at the reply deadline.
            assert self._task is not None
            self._drain_timer = asyncio.get_running_loop().call_later(30, self._task.cancel)

    async def _recognize(self) -> None:
        self._reading = True
        try:
            if self._stream is not None:
                async with self._stream:
                    async for event in self._stream:
                        if (
                            event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
                            and event.alternatives
                        ):
                            self._add_text("amd", event.alternatives[0].text)
            else:
                assert self._model is not None
                event = await self._model.recognize(self._frames, conn_options=self._conn_options)
                if event.alternatives:
                    self._add_text("amd", event.alternatives[0].text)
        except Exception as exc:
            self._fail(exc)
        finally:
            self._frames.clear()
            if self._drain_timer:
                self._drain_timer.cancel()

    def _fail(self, exc: Exception) -> None:
        self._failed = True
        logger.warning("AMD STT failed", extra={"error_type": type(exc).__name__})

    async def aclose(self) -> None:
        if self._drain_timer:
            self._drain_timer.cancel()
        if self._task:
            await aio.cancel_and_wait(self._task)
        if self._stream and not self._reading:
            await self._stream.aclose()
        self._frames.clear()
