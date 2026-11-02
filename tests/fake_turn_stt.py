"""STT fake that replays the event order of a turn-detecting provider.

Cartesia Ink-2's Turns API maps to SpeechEvents as: `turn.start` -> START_OF_SPEECH,
`turn.eager_end` -> PREFLIGHT_TRANSCRIPT, `turn.end` -> FINAL_TRANSCRIPT immediately
followed by END_OF_SPEECH. Deepgram Flux behaves the same way. Timings are relative to
the first pushed audio frame, so these line up with FakeVAD/FakeAudioInput schedules.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Literal

from pydantic import BaseModel

from livekit.agents import LanguageCode
from livekit.agents.stt import (
    STT,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions


class ScriptedTurn(BaseModel):
    speech_start: float
    """When `turn.start` fires."""
    final_at: float
    """When `turn.end` fires, emitting the committed transcript and END_OF_SPEECH."""
    final_text: str
    eager_at: float | None = None
    """When `turn.eager_end` fires. None skips the eager signal entirely."""
    eager_text: str = ""
    """The eager guess. Differs from ``final_text`` when the provider revises it."""


class TurnScriptedSTT(STT):
    def __init__(
        self,
        *,
        turns: list[ScriptedTurn],
        aligned_transcript: Literal["word", "segment"] | bool = False,
    ) -> None:
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=aligned_transcript,
            )
        )
        self._turns = turns
        self._done_fut = asyncio.Future[None]()

    @property
    def turns_done(self) -> asyncio.Future[None]:
        return self._done_fut

    async def _recognize_impl(
        self, buffer, *, language: str | None, conn_options: APIConnectOptions
    ) -> SpeechEvent:
        raise NotImplementedError("streaming only")

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> TurnScriptedStream:
        return TurnScriptedStream(stt=self, conn_options=conn_options)


class TurnScriptedStream(RecognizeStream):
    def __init__(self, *, stt: TurnScriptedSTT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._stt: TurnScriptedSTT = stt

    def _emit(self, event_type: SpeechEventType, text: str = "") -> None:
        # the Cartesia plugin builds SpeechData without start_time/end_time; keep that
        # so timestamp-dependent code paths behave the same here
        alternatives = (
            [SpeechData(text=text, language=LanguageCode("en"), confidence=0.9)] if text else []
        )
        self._event_ch.send_nowait(SpeechEvent(type=event_type, alternatives=alternatives))

    async def _run(self) -> None:
        await self._input_ch.recv()  # align the script with the first pushed frame
        origin = time.perf_counter()

        async def sleep_until(t: float) -> None:
            remaining = t - (time.perf_counter() - origin)
            if remaining > 0:
                await asyncio.sleep(remaining)

        for turn in self._stt._turns:
            await sleep_until(turn.speech_start)
            self._emit(SpeechEventType.START_OF_SPEECH)

            if turn.eager_at is not None:
                await sleep_until(turn.eager_at)
                self._emit(SpeechEventType.PREFLIGHT_TRANSCRIPT, turn.eager_text)

            await sleep_until(turn.final_at)
            self._emit(SpeechEventType.FINAL_TRANSCRIPT, turn.final_text)
            self._emit(SpeechEventType.END_OF_SPEECH)

        with contextlib.suppress(asyncio.InvalidStateError):
            self._stt._done_fut.set_result(None)

        async for _ in self._input_ch:
            pass
