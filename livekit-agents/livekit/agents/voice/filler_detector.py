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
    LANGUAGE_FILLERS = {
        'en': ["uh", "um", "umm", "hmm", "hm", "ah", "er", "mm", "mhm", "uh-huh", "mm-hmm"],
        'hi': ["haan", "hmm", "Accha", "Acha", "uhh", "aah", "achha"],  # Hindi fillers
        'es': ["eh", "este", "pues", "mmm"],  # Spanish fillers
        'fr': ["euh", "heu", "ben", "hmm"],  # French fillers
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
                if lang in self.LANGUAGE_FILLERS:
                    combined_fillers.update(self.LANGUAGE_FILLERS[lang])
                else:
                    logger.warning(
                        f"Language '{lang}' not found in LANGUAGE_FILLERS, skipping"
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
        if language_code in self.LANGUAGE_FILLERS:
            new_fillers = self.LANGUAGE_FILLERS[language_code]
            self._filler_words.update(new_fillers)
            self._rebuild_pattern()
            
            if self._enable_logging:
                logger.info(
                    f"Added {len(new_fillers)} filler words for language '{language_code}'",
                    extra={"added_words": new_fillers}
                )
        else:
            logger.warning(f"Language '{language_code}' not found in LANGUAGE_FILLERS")
    
    def detect(
        self,
        transcript: str,
        confidence: float = 1.0,
        *,
        agent_speaking: bool = False,
    ) -> FillerDetectionResult:
        """Analyze a transcript to determine if it's filler-only or contains meaningful content.
        
        Args:
            transcript: The transcript text to analyze
            confidence: STT confidence score (0.0 to 1.0)
            agent_speaking: Whether the agent is currently speaking
            
        Returns:
            FillerDetectionResult with analysis details
        """
        self._stats['total_transcripts'] += 1
        
        # Handle empty or whitespace-only transcripts
        if not transcript or not transcript.strip():
            self._stats['empty_transcripts'] += 1
            return FillerDetectionResult(
                is_filler_only=True,
                contains_meaningful_content=False,
                original_transcript=transcript,
                filtered_transcript="",
                confidence=confidence,
                should_interrupt=False,
                detection_reason="empty_transcript",
            )
        
        # Normalize confidence to 0.0-1.0 range
        confidence = max(0.0, min(1.0, confidence))
        
        # Low confidence transcripts are treated as potential fillers ONLY when agent is speaking
        if confidence < self._min_confidence and agent_speaking:
            self._stats['low_confidence_ignored'] += 1
            if self._enable_logging:
                logger.debug(
                    "Low confidence transcript treated as filler during agent speech",
                    extra={
                        "transcript": transcript,
                        "confidence": confidence,
                        "threshold": self._min_confidence,
                        "agent_speaking": agent_speaking,
                    }
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
        
        # Remove filler words and check what remains
        filtered = self._filler_pattern.sub('', transcript).strip()
        # Clean up multiple spaces and punctuation-only results
        filtered = re.sub(r'\s+', ' ', filtered)
        # Remove standalone punctuation
        filtered = re.sub(r'^[^\w\s]+$', '', filtered).strip()
        
        is_filler_only = not bool(filtered)
        contains_meaningful = bool(filtered)
        
        # Decision logic:
        # 1. Agent NOT speaking: All speech is valid (treat as potential interruption)
        # 2. Agent IS speaking:
        #    a. Filler-only: Don't interrupt (ignore)
        #    b. Contains meaningful content: Interrupt immediately
        
        if not agent_speaking:
            # Agent is quiet - all speech is valid, even fillers
            self._stats['agent_quiet_valid'] += 1
            should_interrupt = True
            reason = "agent_quiet_all_speech_valid"
            
            if self._enable_logging:
                logger.debug(
                    "Speech detected while agent quiet - treating as valid",
                    extra={
                        "original": transcript,
                        "confidence": confidence,
                        "is_filler_only": is_filler_only,
                    }
                )
        elif is_filler_only:
            # Agent speaking + only fillers = ignore
            should_interrupt = False
            self._stats['filler_only_ignored'] += 1
            reason = "filler_only_during_agent_speech"
            
            if self._enable_logging:
                logger.info(
                    "Ignoring filler-only transcript during agent speech",
                    extra={
                        "original": transcript,
                        "confidence": confidence,
                        "agent_speaking": agent_speaking,
                    }
                )
        else:
            # Agent speaking + meaningful content = interrupt
            should_interrupt = True
            self._stats['meaningful_interruptions'] += 1
            reason = "meaningful_interruption"
            
            if self._enable_logging:
                logger.info(
                    "Meaningful interruption detected",
                    extra={
                        "original": transcript,
                        "filtered": filtered,
                        "confidence": confidence,
                        "agent_speaking": agent_speaking,
                    }
                )
        
        return FillerDetectionResult(
            is_filler_only=is_filler_only,
            contains_meaningful_content=contains_meaningful,
            original_transcript=transcript,
            filtered_transcript=filtered,
            confidence=confidence,
            should_interrupt=should_interrupt,
            detection_reason=reason,
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