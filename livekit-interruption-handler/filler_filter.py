import logging
from typing import Tuple

from livekit.agents import AutoSubscribe, Worker
from livekit.agents.stt import STT
from livekit.agents.tts import TTS
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.rtc import Transcription

from .config import config

logger = logging.getLogger("filler-filter")


class FillerWordFilter:
    """
    Intelligent filter that distinguishes meaningful interruptions from filler words
    """

    def __init__(self, stt_engine: STT, tts_engine: TTS):
        self.stt = stt_engine
        self.tts = tts_engine
        self.is_agent_speaking = False
        self.partial_transcript = ""
        self.ignored_count = 0
        self.interruption_count = 0

    def set_agent_speaking(self, speaking: bool):
        """Update agent speaking state"""
        self.is_agent_speaking = speaking
        if not speaking:
            self.partial_transcript = ""

    def _contains_interruption_trigger(self, text: str) -> bool:
        """Check if text contains real interruption triggers"""
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in config.interruption_triggers)

    def _is_only_filler(self, text: str, confidence: float) -> bool:
        """Determine if text contains only filler words"""
        if confidence < config.confidence_threshold:
            return True

        words = text.lower().split()
        if not words:
            return True

        # Check if all significant words are fillers
        significant_words = [w for w in words if len(w) > 1]  # ignore single letters
        if not significant_words:
            return True

        return all(word in config.ignored_words for word in significant_words)

    def should_interrupt(self, transcription: Transcription) -> Tuple[bool, str]:
        """
        Determine if transcription should cause interruption
        Returns: (should_interrupt, reason)
        """
        text = transcription.text.strip()
        confidence = transcription.confidence

        # Always process when agent is not speaking
        if not self.is_agent_speaking:
            return True, "agent_not_speaking"

        # Low confidence background noise
        if confidence < config.confidence_threshold:
            logger.debug(f"Ignoring low confidence speech: '{text}' (confidence: {confidence:.2f})")
            self.ignored_count += 1
            return False, "low_confidence"

        # Check for interruption triggers
        if self._contains_interruption_trigger(text):
            logger.info(f"Real interruption detected: '{text}'")
            self.interruption_count += 1
            return True, "interruption_trigger"

        # Check if only filler words
        if self._is_only_filler(text, confidence):
            logger.debug(f"Ignoring filler speech: '{text}'")
            self.ignored_count += 1
            return False, "filler_words"

        # Mixed content with non-filler words
        logger.info(f"Meaningful speech during agent talk: '{text}'")
        self.interruption_count += 1
        return True, "meaningful_speech"

    def get_stats(self) -> dict:
        """Get filtering statistics"""
        return {
            "ignored_count": self.ignored_count,
            "interruption_count": self.interruption_count,
            "ignored_words": list(config.ignored_words),
            "interruption_triggers": list(config.interruption_triggers)
        }


class FilteredVoiceAssistant(VoiceAssistant):
    """
    Enhanced VoiceAssistant with intelligent interruption handling
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filler_filter = FillerWordFilter(self._stt, self._tts)

        # Override transcription handler
        self._user_speech_stream = None

    async def start(self):
        """Start the assistant with filler filtering"""
        await super().start()

    async def _on_tts_start(self):
        """Called when TTS starts"""
        self.filler_filter.set_agent_speaking(True)
        await super()._on_tts_start()

    async def _on_tts_end(self):
        """Called when TTS ends"""
        self.filler_filter.set_agent_speaking(False)
        await super()._on_tts_end()

    async def _on_transcription(self, transcription: Transcription):
        """
        Override transcription handling with intelligent filtering
        """
        should_interrupt, reason = self.filler_filter.should_interrupt(transcription)

        if should_interrupt:
            logger.info(f"Processing interruption: {transcription.text} (reason: {reason})")
            await super()._on_transcription(transcription)
        else:
            logger.debug(f"Filtered out: {transcription.text} (reason: {reason})")

    def update_ignored_words(self, words: list[str]):
        """Dynamically update ignored words list"""
        config.update_ignored_words(words)
        logger.info(f"Updated ignored words: {config.ignored_words}")

    def get_filter_stats(self) -> dict:
        """Get filtering statistics"""
        return self.filler_filter.get_stats()