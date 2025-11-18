"""
Interrupt Handler for LiveKit-style agents.
Corrected version — passes all tests.
"""

import asyncio
import re
import os


def _default_ignored_words():
    env = os.getenv("IGNORED_WORDS", "uh,umm,hmm")
    return [w.strip().lower() for w in env.split(",") if w.strip()]


class InterruptHandler:
    def __init__(self, ignored_words=None, confidence_threshold=0.35):
        self.ignored_words = ignored_words or _default_ignored_words()
        self.confidence_threshold = float(confidence_threshold)
        self.is_agent_speaking = False

        # compile pattern for full-word match
        escaped = [re.escape(w) for w in self.ignored_words]
        pattern = r"^\s*(?:" + "|".join(escaped) + r")\s*$"
        self._ignored_pattern = re.compile(pattern, re.IGNORECASE)

        # callbacks
        self.on_valid_interrupt = lambda text: None
        self.on_ignored_filler = lambda text: None

    async def on_agent_state_change(self, is_speaking: bool):
        self.is_agent_speaking = bool(is_speaking)

    async def on_transcription_event(self, text: str, confidence: float = 1.0):
        if not text:
            return None

        trimmed = text.strip()
        if not trimmed:
            return None

        # ignore low confidence
        if confidence < self.confidence_threshold:
            return None

        lowered = trimmed.lower()

        # 1) If agent is speaking and text is a filler → ignore
        if self.is_agent_speaking and self._ignored_pattern.match(lowered):
            self.on_ignored_filler(trimmed)
            return "ignored_filler"

        # 2) If text contains a stop command → interrupt
        stop_cmds = ["stop", "wait", "pause", "hold on", "no", "never mind", "not that"]
        if any(cmd in lowered for cmd in stop_cmds):
            self.on_valid_interrupt(trimmed)
            return "interrupt"

        # 3) If agent is speaking → ANY speech interrupts
        if self.is_agent_speaking:
            self.on_valid_interrupt(trimmed)
            return "interrupt"

        # 4) If agent is quiet → normal user speech
        self.on_valid_interrupt(trimmed)
        return "user_speech"
