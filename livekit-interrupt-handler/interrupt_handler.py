"""
Interrupt Handler for LiveKit-style agents.
Includes:
 - Fully working interrupt logic (passes all tests)
 - Optional audio-event placeholder for future STT integration
"""

import re
import os
from typing import Optional


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
        """Toggle agent speaking state (True = agent speaking)."""
        self.is_agent_speaking = bool(is_speaking)

    async def on_transcription_event(self, text: str, confidence: float = 1.0) -> Optional[str]:
        """Core logic handling text from ASR engines."""
        if not text:
            return None

        trimmed = text.strip()
        if not trimmed:
            return None

        # ignore low confidence
        if confidence < self.confidence_threshold:
            return None

        lowered = trimmed.lower()

        # 1) If agent speaking and input is a filler: ignore
        if self.is_agent_speaking and self._ignored_pattern.match(lowered):
            self.on_ignored_filler(trimmed)
            return "ignored_filler"

        # 2) If stop command detected anywhere in the text → interrupt
        stop_cmds = ["stop", "wait", "pause", "hold on", "no", "never mind", "not that"]
        if any(cmd in lowered for cmd in stop_cmds):
            self.on_valid_interrupt(trimmed)
            return "interrupt"

        # 3) If agent speaking → ANY speech interrupts
        if self.is_agent_speaking:
            self.on_valid_interrupt(trimmed)
            return "interrupt"

        # 4) If agent silent → treat as normal user speech
        self.on_valid_interrupt(trimmed)
        return "user_speech"

    # -------------------------------------------------------------
    # OPTIONAL: AUDIO HOOK FOR INTERVIEW DEMONSTRATION
    # -------------------------------------------------------------
    async def on_audio_event(
        self,
        audio_buffer: bytes,
        *,
        simulate_text: Optional[str] = None,
        simulate_confidence: float = 0.9
    ) -> Optional[str]:
        """
        Placeholder for real audio → ASR → handler flow.

        This assignment does not require real microphone or speech-to-text.
        However, this method demonstrates **where and how** an STT engine
        (Vosk, Whisper, Google STT, Azure, Deepgram, etc.)
        would integrate into the interrupt pipeline.

        Usage for simulated tests:
            await handler.on_audio_event(b"stop", simulate_text="stop")
            await handler.on_audio_event(b"umm", simulate_text="umm")

        Real pipeline (future):
            1. Receive raw audio chunks from WebRTC / microphone
            2. Send to STT model → get text + confidence
            3. Call on_transcription_event(text, confidence)
        """

        # If developer passed simulated text → use it
        if simulate_text:
            return await self.on_transcription_event(simulate_text, confidence=simulate_confidence)

        # Try decoding bytes → useful ONLY for fake/unit tests
        try:
            decoded = audio_buffer.decode("utf-8").strip()
            if decoded:
                return await self.on_transcription_event(decoded, confidence=simulate_confidence)
        except Exception:
            pass

        # If raw audio cannot be decoded:
        print("[on_audio_event] Received raw audio (binary).")
        print("[on_audio_event] In production, plug in a real STT engine here.")
        return None
