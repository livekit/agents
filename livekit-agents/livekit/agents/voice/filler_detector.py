from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ..log import logger


@dataclass
class FillerDetectionResult:
    """Result of filler word detection analysis"""
    is_filler_only: bool
    """True if the transcript contains only filler words"""
    contains_meaningful_content: bool
    """True if the transcript contains any non-filler content"""
    original_transcript: str
    """The original transcript text"""
    filtered_transcript: str
    """Transcript with filler words removed"""
    confidence: float
    """Confidence score from STT (0.0 to 1.0)"""
    should_interrupt: bool
    """True if this transcript should interrupt the agent"""
    detection_reason: str
    """Reason for the detection decision (for debugging)"""


class FillerDetector:
    """Detects and filters filler words from user speech transcripts
    
    This detector intelligently distinguishes between filler words (uh, umm, hmm)
    and meaningful interruptions (wait, stop, no). It only ignores fillers when
    the agent is actively speaking, treating them as valid speech otherwise.
    
    Key behaviors:
    - Agent speaking + filler-only → Ignore (don't interrupt)
    - Agent speaking + meaningful content → Interrupt immediately
    - Agent quiet + any speech → Always register as valid
    - Low confidence + agent speaking → Treat as filler
    """
    
    DEFAULT_FILLER_WORDS = [
        "uh", "um", "umm", "hmm", "hm", "haan", "ah", "er", "mm",
        "mhm", "uh-huh", "mm-hmm", "uhh", "err", "ehh", "huh"
    ]
    
    # Language-specific filler words for multi-language support
    LANGUAGE_FILLER_WORDS = {
        "en": ["uh", "um", "umm", "er", "err", "ah", "ahh", "hm", "hmm", "mm", "mhm", "mm-hmm", "uh-huh", "uhh", "aah"],
        "hi": ["achha", "accha", "haan", "hmm", "uh", "um", "acha"],
        "es": ["eh", "este", "pues", "bueno", "mmm"],
        "fr": ["euh", "ben", "voilà", "quoi", "hein", "hmm"],
    }
    
    def __init__(
        self,
        *,
        filler_words: Sequence[str] | None = None,
        min_confidence_threshold: float = 0.3,
        enable_logging: bool = True,
        languages: Sequence[str] | None = None,
    ) -> None:
        """Initialize the filler detector.
        
        Args:
            filler_words: List of words/phrases to treat as fillers. 
                If None, uses DEFAULT_FILLER_WORDS
            min_confidence_threshold: Minimum STT confidence (0.0-1.0) to consider 
                transcript valid. Lower confidence transcripts are treated as fillers.
            enable_logging: Whether to log detection events for debugging
            languages: List of language codes to load filler words from (e.g., ['en', 'hi'])
                If None and filler_words is None, uses DEFAULT_FILLER_WORDS only
        """
        # Build filler word set from multiple sources
        if filler_words is not None:
            self._filler_words = set(word.lower().strip() for word in filler_words)
        elif languages:
            # Combine filler words from specified languages
            combined_fillers = set()
            for lang in languages:
                if lang in self.LANGUAGE_FILLER_WORDS:
                    combined_fillers.update(self.LANGUAGE_FILLER_WORDS[lang])
                else:
                    logger.warning(
                        f"Language '{lang}' not found in LANGUAGE_FILLER_WORDS, skipping"
                    )
            self._filler_words = combined_fillers if combined_fillers else set(self.DEFAULT_FILLER_WORDS)
        else:
            self._filler_words = set(self.DEFAULT_FILLER_WORDS)
        
        self._min_confidence = max(0.0, min(1.0, min_confidence_threshold))
        self._enable_logging = enable_logging
        self._languages = list(languages) if languages else None
        
        # Pre-compile regex pattern for word boundary matching
        self._rebuild_pattern()
        
        # Enhanced statistics for debugging
        self._stats = {
            'total_transcripts': 0,
            'filler_only_ignored': 0,
            'meaningful_interruptions': 0,
            'low_confidence_ignored': 0,
            'agent_quiet_valid': 0,  # All speech when agent is quiet
            'empty_transcripts': 0,
        }
    
    def _rebuild_pattern(self) -> None:
        """Rebuild the regex pattern after filler words change"""
        if not self._filler_words:
            # Empty pattern that matches nothing
            self._filler_pattern = re.compile(r'(?!.*)', re.IGNORECASE)
            return
            
        escaped_fillers = [re.escape(word) for word in self._filler_words]
        self._filler_pattern = re.compile(
            r'\b(' + '|'.join(escaped_fillers) + r')\b',
            re.IGNORECASE
        )
    
    def update_filler_words(self, filler_words: Sequence[str]) -> None:
        """Dynamically update the filler word list at runtime.
        
        Args:
            filler_words: New list of filler words to use
        """
        self._filler_words = set(word.lower().strip() for word in filler_words)
        self._rebuild_pattern()
        
        if self._enable_logging:
            logger.info(
                "Updated filler words list",
                extra={"filler_words": sorted(list(self._filler_words))}
            )
    
    def add_language_fillers(self, language_code: str) -> None:
        """Add filler words for a specific language to the existing set.
        
        Args:
            language_code: Language code (e.g., 'hi', 'es', 'fr')
        """
        if language_code in self.LANGUAGE_FILLER_WORDS:
            new_fillers = self.LANGUAGE_FILLER_WORDS[language_code]
            self._filler_words.update(new_fillers)
            self._rebuild_pattern()
            
            if self._enable_logging:
                logger.info(
                    f"Added {len(new_fillers)} filler words for language '{language_code}'",
                    extra={"added_words": new_fillers}
                )
        else:
            logger.warning(f"Language '{language_code}' not found in LANGUAGE_FILLER_WORDS")
    
    def detect(
        self, transcript: str, confidence: float, *, agent_speaking: bool = False
    ) -> FillerDetectionResult:
        """Detect if transcript contains only filler words."""
        
        # Debug logging
        if self._enable_logging:
            logger.info(
                f"[FillerDetector] Input: transcript='{transcript}', "
                f"confidence={confidence:.3f}, agent_speaking={agent_speaking}"
            )

        self._stats["total_transcripts"] += 1

        # Empty transcript check
        if not transcript or not transcript.strip():
            self._stats["empty_transcripts"] += 1
            if self._enable_logging:
                logger.debug("[FillerDetector] Empty transcript detected")
            return FillerDetectionResult(
                is_filler_only=True,
                contains_meaningful_content=False,
                original_transcript=transcript,
                filtered_transcript="",
                confidence=confidence,
                should_interrupt=False,
                detection_reason="empty_transcript",
            )

        # Low confidence during agent speech -> treat as filler
        # FIX: Changed _min_confidence_threshold to _min_confidence
        if agent_speaking and confidence < self._min_confidence:
            self._stats["low_confidence_ignored"] += 1
            if self._enable_logging:
                logger.info(
                    f"[FillerDetector] Low confidence during agent speech: "
                    f"confidence={confidence:.3f} < {self._min_confidence}"
                )
            return FillerDetectionResult(
                is_filler_only=True,
                contains_meaningful_content=False,
                original_transcript=transcript,
                filtered_transcript="",
                confidence=confidence,
                should_interrupt=False,
                detection_reason="low_confidence_during_agent_speech",
            )

        # Check if transcript contains only filler words
        is_filler_only, filtered = self._is_filler_only(transcript)
        
        # Debug: Show matched filler words
        if self._enable_logging:
            matched_fillers = [word for word in transcript.lower().split() if word in self._filler_words]
            logger.info(
                f"[FillerDetector] Matched fillers: {matched_fillers}, "
                f"is_filler_only={is_filler_only}, filtered='{filtered}'"
            )

        if is_filler_only and agent_speaking:
            # Filler-only speech during agent speaking -> ignore
            self._stats["filler_only_ignored"] += 1
            if self._enable_logging:
                logger.info(
                    f"[FillerDetector] ✓ Ignoring filler-only transcript: '{transcript}'"
                )
            return FillerDetectionResult(
                is_filler_only=True,
                contains_meaningful_content=False,
                original_transcript=transcript,
                filtered_transcript="",
                confidence=confidence,
                should_interrupt=False,
                detection_reason="filler_only_during_agent_speech",
            )

        elif not is_filler_only and agent_speaking:
            # Meaningful content during agent speaking -> interrupt
            self._stats["meaningful_interruptions"] += 1
            if self._enable_logging:
                logger.info(
                    f"[FillerDetector] ✓ Meaningful interruption: '{transcript}' -> '{filtered}'"
                )
            return FillerDetectionResult(
                is_filler_only=False,
                contains_meaningful_content=True,
                original_transcript=transcript,
                filtered_transcript=filtered,
                confidence=confidence,
                should_interrupt=True,
                detection_reason="meaningful_interruption",
            )

        else:
            # Agent not speaking -> all speech is valid
            self._stats["agent_quiet_valid"] += 1
            if self._enable_logging:
                logger.debug(
                    f"[FillerDetector] Agent quiet - treating as valid: '{transcript}'"
                )
            return FillerDetectionResult(
                is_filler_only=is_filler_only,
                contains_meaningful_content=not is_filler_only,
                original_transcript=transcript,
                filtered_transcript=filtered if not is_filler_only else "",
                confidence=confidence,
                should_interrupt=True,
                detection_reason="agent_quiet_all_speech_valid",
            )
    
    def get_stats(self) -> dict[str, int]:
        """Get detection statistics for debugging and monitoring"""
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset detection statistics"""
        self._stats = {
            'total_transcripts': 0,
            'filler_only_ignored': 0,
            'meaningful_interruptions': 0,
            'low_confidence_ignored': 0,
            'agent_quiet_valid': 0,
            'empty_transcripts': 0,
        }
    
    @property
    def filler_words(self) -> set[str]:
        """Get the current list of filler words"""
        return self._filler_words.copy()
    
    @property
    def min_confidence_threshold(self) -> float:
        """Get the minimum confidence threshold"""
        return self._min_confidence
    
    @min_confidence_threshold.setter
    def min_confidence_threshold(self, value: float) -> None:
        """Set the minimum confidence threshold (0.0 to 1.0)"""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        self._min_confidence = value
        
        if self._enable_logging:
            logger.info(
                f"Updated min_confidence_threshold to {value}",
                extra={"new_threshold": value}
            )
