"""
FillerAwareAdapter: Prevents filler-only utterances from interrupting TTS,
while still allowing real commands and meaningful speech to interrupt.
Implements an STT wrapper layer without modifying base VAD logic.
"""



from __future__ import annotations

import asyncio
import re
from typing import Callable, List
from livekit.agents.stt import (
    STT,
    RecognizeStream,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit import rtc

from .config import (
    IGNORED_FILLERS,
    INTERRUPT_COMMANDS,
    MIN_CONFIDENCE_AGENT_SPEAKING,
    SHORT_SEGMENT_TOKENS,
)


# ---------------------------- Helpers ----------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)

def _split_words(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]

def _is_filler_only(words: List[str]) -> bool:
    return len(words) > 0 and all(w in IGNORED_FILLERS for w in words)

def _contains_command(words: List[str]) -> bool:
    text = " ".join(words)
    # multi-word commands
    if any(cmd in text for cmd in INTERRUPT_COMMANDS if " " in cmd):
        return True
    return any(w in INTERRUPT_COMMANDS for w in words)


# ---------------------------- Adapter ----------------------------

class FillerAwareAdapter(STT):
    """
    Prevents filler interruptions while agent is speaking.
    """

    def __init__(self, base_stt: STT, is_agent_speaking: Callable[[], bool]):
        super().__init__(
            capabilities=STTCapabilities(
                streaming=base_stt.capabilities.streaming,
                interim_results=base_stt.capabilities.interim_results,
                diarization=base_stt.capabilities.diarization,
            )
        )
        self._base = base_stt
        self._is_agent_speaking = is_agent_speaking

        base_stt.on("metrics_collected",
            lambda *a, **k: self.emit("metrics_collected", *a, **k))
        base_stt.on("error",
            lambda *a, **k: self.emit("error", *a, **k))

    @property
    def model(self) -> str:
        return f"FillerAware({self._base.model})"

    @property
    def provider(self) -> str:
        return self._base.provider


    async def _recognize_impl(self, *args, **kwargs):
        return await self._base._recognize_impl(*args, **kwargs)

    def stream(self, *args, **kwargs) -> RecognizeStream:
        return _FillerStream(self, self._base.stream(*args, **kwargs), self._is_agent_speaking)

    async def aclose(self):
        await self._base.aclose()


class _FillerStream(RecognizeStream):
    def __init__(self, outer, base_stream: RecognizeStream, is_agent_speaking):
        super().__init__(
            stt=outer,
            conn_options=base_stream._conn_options,
            sample_rate=getattr(base_stream, "_needed_sr", None),
        )
        self._outer = outer
        self._base = base_stream
        self._is_agent_speaking = is_agent_speaking

    async def _run(self):
        async def _forward_input():
            async for item in self._input_ch:
                if isinstance(item, rtc.AudioFrame):
                    self._base.push_frame(item)
                else:
                    self._base.flush()
            self._base.end_input()

        forward_task = asyncio.create_task(_forward_input())

        try:
            async with self._base:
                async for ev in self._base:
                    filtered = self._filter(ev)
                    if filtered is not None:
                        self._event_ch.send_nowait(filtered)
        finally:
            await forward_task

    # -------------------- Core Filtering Logic --------------------

    def _filter(self, ev: SpeechEvent):
        # non-text events → keep
        if ev.type not in {
            SpeechEventType.INTERIM_TRANSCRIPT,
            SpeechEventType.PREFLIGHT_TRANSCRIPT,
            SpeechEventType.FINAL_TRANSCRIPT,
        }:
            return ev

        if not ev.alternatives:
            return ev

        alt = ev.alternatives[0]
        text = alt.text or ""
        words = _split_words(text)

        if not words:
            return ev

        # ------------------ Agent NOT speaking → keep everything ------------------
        if not self._is_agent_speaking():
            return ev

        # ------------------ Agent speaking → apply suppression rules ------------------

        # 1) Commands always interrupt
        if _contains_command(words):
            return ev  # allow interruption

        # 2) Low-confidence short segments → ignore interruption
        if alt.confidence and alt.confidence < MIN_CONFIDENCE_AGENT_SPEAKING:
            if len(words) <= SHORT_SEGMENT_TOKENS:
                ev._ignore_interruption = True
                return ev

        # 3) Pure filler → do NOT interrupt
        if _is_filler_only(words):
            ev._ignore_interruption = True
            return ev

        # Non-filler real speech → interrupt normally
        return ev
