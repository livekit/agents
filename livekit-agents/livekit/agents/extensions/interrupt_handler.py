"""Interrupt handling extension for LiveKit Agents.

This module defines configuration, state tracking, and handler classes
for voice interruption handling built on top of LiveKit Agents.

Production-grade features:
- Debouncing for interim transcripts
- Confidence smoothing via EMA
- Robust text normalization (unicode, repeated chars, punctuation)
- Fuzzy matching with graceful fallback
- Rate limiting for interrupts
- Structured JSON-friendly logging
"""

import asyncio
import json
import logging
import os
import re
import time
import unicodedata
from typing import Any, Dict, Iterable, Literal, Optional


logger = logging.getLogger("interrupt_handler")

# Attempt to import rapidfuzz for fuzzy matching, gracefully degrade if not available
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.info("rapidfuzz not available; fuzzy matching will be disabled")


# Hindi transliteration mapping (Romanized Hindi → Devanagari concepts)
HINDI_TRANSLITERATION_MAP = {
    # Affirmative fillers
    "haan": "हाँ",
    "han": "हाँ",
    "haa": "हाँ",
    "ha": "हाँ",
    # Negative
    "nahi": "नहीं",
    "nai": "नहीं",
    "nahiin": "नहीं",
    "nhi": "नहीं",
    # Stop commands
    "ruk": "रुको",
    "ruko": "रुको",
    "rukko": "रुको",
    "ruk jao": "रुक जाओ",
    "ruk ja": "रुक जाओ",
    # Pause/wait
    "thaher": "ठहर",
    "thahro": "ठहर",
    "ruko": "रुको",
    # Stop/enough
    "bas": "बस",
    "band karo": "बंद करो",
    "band kar": "बंद करो",
    # Okay/acknowledgement
    "thik": "ठीक",
    "theek": "ठीक",
    "tik": "ठीक",
    "thik hai": "ठीक है",
    "theek hai": "ठीक है",
    # Other fillers
    "arey": "अरे",
    "arre": "अरे",
    "achha": "अच्छा",
    "accha": "अच्छा",
    "haina": "है ना",
    "hai na": "है ना",
}


class InterruptConfig:
    """Configuration for the interruption handler.

    Fields:
        ignored_words: List of filler words or phrases to ignore while the agent is speaking.
        command_words: List of high-priority words or phrases that should always trigger an
            interruption.
        min_confidence: Minimum ASR confidence required for non-command interruptions.
        interim_interrupt_policy: Policy for interim transcripts ("conservative" or "aggressive").
        debounce_ms: Milliseconds to wait before processing interim transcripts (debounce window).
        confidence_smoothing_alpha: Exponential moving average alpha for confidence smoothing.
        use_fuzzy_matching: Enable fuzzy matching for short words (requires rapidfuzz).
        log_verbosity: Logging level ("DEBUG" or "INFO").
        interrupt_rate_limit_ms: Minimum milliseconds between interrupt calls (rate limiting).
    """

    ignored_words: list
    command_words: list
    min_confidence: float
    interim_interrupt_policy: Literal["conservative", "aggressive"]
    debounce_ms: int
    confidence_smoothing_alpha: float
    use_fuzzy_matching: bool
    log_verbosity: str
    interrupt_rate_limit_ms: int
    language_mode: Literal["en", "hi", "auto", "multi"]
    ignored_words_en: list
    ignored_words_hi: list
    command_words_en: list
    command_words_hi: list

    def __init__(
        self,
        *,
        ignored_words: Optional[Iterable[str]] = None,
        command_words: Optional[Iterable[str]] = None,
        min_confidence: Optional[float] = None,
        interim_interrupt_policy: Literal["conservative", "aggressive"] = "conservative",
        debounce_ms: int = 200,
        confidence_smoothing_alpha: float = 0.3,
        use_fuzzy_matching: bool = True,
        log_verbosity: str = "INFO",
        interrupt_rate_limit_ms: int = 500,
        language_mode: Literal["en", "hi", "auto", "multi"] = "auto",
    ) -> None:
        env_ignored = os.getenv("LIVEKIT_IGNORED_WORDS")
        env_commands = os.getenv("LIVEKIT_COMMAND_WORDS")
        env_min_conf = os.getenv("LIVEKIT_MIN_CONFIDENCE")

        # English filler words
        self.ignored_words_en = ["uh", "umm", "um", "hmm", "er", "ah", "uhh", "mmm", "err"]

        # Hindi filler words (Romanized + Devanagari)
        self.ignored_words_hi = [
            "haan",
            "han",
            "haa",
            "hmm",
            "arey",
            "arre",
            "achha",
            "accha",
            "haina",
            "hai na",
            "हाँ",
            "अरे",
            "अच्छा",
            "है ना",
        ]

        # English command words
        self.command_words_en = ["stop", "wait", "no", "hold on", "pause", "hang on", "wait a second"]

        # Hindi command words (Romanized + Devanagari)
        self.command_words_hi = [
            "ruk",
            "ruko",
            "rukko",
            "ruk jao",
            "ruk ja",
            "nahi",
            "nai",
            "thaher",
            "thahro",
            "bas",
            "band karo",
            "band kar",
            "abey ruk",
            "nahi yaar",
            "रुको",
            "रुक जाओ",
            "नहीं",
            "ठहर",
            "बस",
            "बंद करो",
        ]

        if ignored_words is None:
            if env_ignored:
                ignored_words = [w.strip().lower() for w in env_ignored.split(",") if w.strip()]
            else:
                # Combine English + Hindi by default
                ignored_words = list(set(self.ignored_words_en + self.ignored_words_hi))

        if command_words is None:
            if env_commands:
                command_words = [w.strip().lower() for w in env_commands.split(",") if w.strip()]
            else:
                # Combine English + Hindi by default
                command_words = list(set(self.command_words_en + self.command_words_hi))

        if min_confidence is None:
            if env_min_conf:
                try:
                    min_confidence = float(env_min_conf)
                except ValueError:
                    min_confidence = 0.5
            else:
                min_confidence = 0.5

        self.ignored_words = [w.lower() for w in ignored_words]
        self.command_words = [w.lower() for w in command_words]
        self.min_confidence = float(min_confidence)
        self.interim_interrupt_policy = interim_interrupt_policy
        self.debounce_ms = max(0, int(debounce_ms))
        self.confidence_smoothing_alpha = max(0.0, min(1.0, float(confidence_smoothing_alpha)))
        self.use_fuzzy_matching = bool(use_fuzzy_matching) and RAPIDFUZZ_AVAILABLE
        self.log_verbosity = log_verbosity.upper() if log_verbosity else "INFO"
        self.interrupt_rate_limit_ms = max(0, int(interrupt_rate_limit_ms))
        self.language_mode = language_mode if language_mode in ("en", "hi", "auto", "multi") else "auto"


class InterruptState:
    """Runtime state for the interruption handler.

    Fields:
        agent_speaking: Whether the agent is currently speaking.
        last_interruption_time: Timestamp of the last interrupt call (for rate limiting).
        confidence_ema: Exponential moving average of ASR confidence values.
        last_interim_transcript: Last interim transcript received (for debouncing).
        last_transcript_time: Timestamp of the last transcript (for debouncing).
        debounce_task: Asyncio task handle for debounce timer.
    """

    def __init__(self) -> None:
        self.agent_speaking: bool = False
        self.last_interruption_time: float = 0.0
        self.confidence_ema: Optional[float] = None
        self.last_interim_transcript: str = ""
        self.last_transcript_time: float = 0.0
        self.debounce_task: Optional[asyncio.Task] = None


def _detect_language(text: str) -> Literal["en", "hi", "multi"]:
    """Detect language from text using heuristics.

    Args:
        text: Input text.

    Returns:
        "hi" if Hindi/Devanagari detected, "en" if English, "multi" if mixed.
    """
    if not text:
        return "en"

    # Check for Devanagari script
    has_devanagari = any("\u0900" <= char <= "\u097F" for char in text)

    # Check for Hindi transliteration keywords
    text_lower = text.lower()
    hindi_keywords = ["haan", "nahi", "ruk", "bas", "arey", "achha", "thik"]
    has_hindi_keywords = any(keyword in text_lower for keyword in hindi_keywords)

    if has_devanagari and has_hindi_keywords:
        return "multi"
    elif has_devanagari:
        return "hi"
    elif has_hindi_keywords:
        return "hi"
    else:
        return "en"


def _normalize_text_multilingual(text: str) -> tuple[str, str]:
    """Normalize text for multilingual (English + Hindi) matching.

    Handles:
    - Unicode normalization (NFKC for Hindi)
    - Devanagari nukta removal
    - Repeated character collapsing
    - Punctuation removal
    - Transliteration mapping

    Args:
        text: Raw transcript text.

    Returns:
        Tuple of (normalized_text, detected_language).
    """
    if not text:
        return "", "en"

    # Detect language first
    detected_lang = _detect_language(text)

    # Unicode normalization (NFKC for Hindi/Devanagari)
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Remove Devanagari nukta variants (क़ → क)
    text = re.sub(r"[\u093C]", "", text)  # Remove nukta combining character

    # Collapse repeated characters (e.g., "haaaaan" → "haan", "noooo" → "noo")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Remove punctuation except apostrophes and hyphens within words
    text = re.sub(r"[^\w\s'-]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Apply transliteration mapping for common Hindi words
    tokens = text.split()
    normalized_tokens = []
    for token in tokens:
        if token in HINDI_TRANSLITERATION_MAP:
            # Keep original for matching purposes
            normalized_tokens.append(token)
        else:
            normalized_tokens.append(token)

    normalized = " ".join(normalized_tokens)

    return normalized, detected_lang


def _normalize_text(text: str) -> str:
    """Normalize text for robust token matching.

    Applies unicode normalization, removes repeated characters, trims punctuation,
    and collapses whitespace.

    Args:
        text: Raw transcript text.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    # Unicode normalization (NFKD: compatibility decomposition)
    text = unicodedata.normalize("NFKD", text)

    # Lowercase
    text = text.lower()

    # Collapse repeated characters (e.g., "ummmmm" -> "umm", "noooo" -> "noo")
    # Keep up to 2 repetitions to preserve some expressiveness
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Remove punctuation except apostrophes and hyphens within words
    text = re.sub(r"[^\w\s'-]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _tokenize(text: str) -> list:
    """Split normalized text into tokens.

    Args:
        text: Normalized text.

    Returns:
        List of lowercase tokens.
    """
    return [t for t in text.split() if t]


def _contains_phrase_with_boundaries(text: str, phrase: str) -> bool:
    """Check if text contains phrase with word boundaries.

    Args:
        text: Normalized text to search in.
        phrase: Phrase to search for.

    Returns:
        True if phrase is found with word boundaries.
    """
    if not phrase:
        return False

    # Use regex word boundaries to avoid substring false positives
    # For Devanagari, word boundaries work differently, so also do simple substring check
    pattern = r"\b" + re.escape(phrase) + r"\b"
    if re.search(pattern, text, flags=re.IGNORECASE):
        return True

    # Fallback for Devanagari or languages without clear word boundaries
    # Check if phrase appears as complete tokens
    text_tokens = set(text.split())
    phrase_tokens = phrase.split()

    if len(phrase_tokens) == 1:
        return phrase.lower() in text_tokens
    else:
        # Multi-word phrase: check if it appears in sequence
        return phrase.lower() in text.lower()


def _match_command_in_list(
    normalized_text: str, command_list: list, use_fuzzy: bool = False
) -> Optional[str]:
    """Match command words/phrases in text, prioritizing multi-word phrases.

    Args:
        normalized_text: Normalized text.
        command_list: List of command words/phrases.
        use_fuzzy: Whether to use fuzzy matching for single tokens.

    Returns:
        Matched command string if found, None otherwise.
    """
    if not normalized_text or not command_list:
        return None

    # Sort commands by length (longest first) to match multi-word phrases before single words
    sorted_commands = sorted(command_list, key=lambda x: len(x.split()), reverse=True)

    for cmd in sorted_commands:
        if _contains_phrase_with_boundaries(normalized_text, cmd):
            return cmd

    # Fuzzy matching for single-word commands if enabled
    if use_fuzzy and RAPIDFUZZ_AVAILABLE:
        tokens = _tokenize(normalized_text)
        for token in tokens:
            for cmd in command_list:
                if " " not in cmd and fuzz.ratio(token, cmd) >= 85:
                    return cmd

    return None


def _fuzzy_match_token(token: str, candidates: list, threshold: int = 85) -> bool:
    """Check if token fuzzy-matches any candidate using rapidfuzz.

    Args:
        token: Token to match.
        candidates: List of candidate words.
        threshold: Minimum similarity score (0-100).

    Returns:
        True if a fuzzy match is found above threshold.
    """
    if not RAPIDFUZZ_AVAILABLE or not token or not candidates:
        return False

    for candidate in candidates:
        if fuzz.ratio(token, candidate) >= threshold:
            return True

    return False


class InterruptHandler:
    """Handle transcription events and decide when to interrupt the agent.

    This class subscribes to AgentSession events and applies interruption logic
    without modifying the core LiveKit Agents SDK.
    """

    def __init__(self, session, config=None) -> None:
        """Initialize the interruption handler.

        Args:
            session: The AgentSession instance to attach to.
            config: Optional InterruptConfig instance or configuration data.
        """

        if isinstance(config, InterruptConfig):
            self._config = config
        else:
            self._config = InterruptConfig(**(config or {})) if isinstance(config, dict) else InterruptConfig()

        self._session = session
        self._state = InterruptState()
        self._lock = asyncio.Lock()

        # Configure logger verbosity
        if self._config.log_verbosity == "DEBUG":
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    def on_agent_state_changed(self, event) -> None:
        """Handle agent_state_changed events from AgentSession.

        Updates the internal flag tracking whether the agent is currently speaking.
        """

        new_state = getattr(event, "new_state", None)
        self._state.agent_speaking = new_state == "speaking"

    def on_user_input_transcribed(self, event) -> None:
        """Handle user_input_transcribed events from AgentSession.

        Extracts transcript information and schedules asynchronous handling.
        Non-blocking to maintain low-latency event processing.
        """

        transcript = getattr(event, "transcript", "") or ""
        is_final = bool(getattr(event, "is_final", False))
        confidence = None  # Confidence would come from custom STT wrapper if available
        language = getattr(event, "language", None)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("No running event loop; skipping asynchronous handling of transcript.")
            return

        # Non-blocking: schedule handling without awaiting
        asyncio.create_task(
            self.handle_transcription(
                transcript=transcript,
                confidence=confidence,
                is_final=is_final,
                language=language,
            )
        )

    async def handle_transcription(
        self, transcript, confidence=None, is_final: bool = False, language: Optional[str] = None
    ) -> None:
        """Process a transcript segment and decide whether to interrupt.

        Implements debouncing for interim transcripts, rate limiting for interrupts,
        confidence smoothing via EMA, and structured logging.

        Args:
            transcript: The recognized text from the user.
            confidence: Optional ASR confidence value for the segment.
            is_final: Whether this transcript is final for the current segment.
            language: Optional language code from the transcript event.
        """

        current_time = time.time()

        # Update confidence EMA if provided
        if confidence is not None:
            if self._state.confidence_ema is None:
                self._state.confidence_ema = confidence
            else:
                alpha = self._config.confidence_smoothing_alpha
                self._state.confidence_ema = alpha * confidence + (1 - alpha) * self._state.confidence_ema

        # Use smoothed confidence for decision making
        smoothed_confidence = self._state.confidence_ema if self._state.confidence_ema is not None else confidence

        # Debounce interim transcripts
        if not is_final and self._config.debounce_ms > 0:
            # Cancel previous debounce task
            if self._state.debounce_task and not self._state.debounce_task.done():
                self._state.debounce_task.cancel()

            # Store interim transcript and schedule debounced processing
            self._state.last_interim_transcript = transcript
            self._state.last_transcript_time = current_time

            async def _debounced_process():
                await asyncio.sleep(self._config.debounce_ms / 1000.0)
                await self._process_transcript(
                    transcript, smoothed_confidence, is_final, language, current_time
                )

            self._state.debounce_task = asyncio.create_task(_debounced_process())
            return

        # Process final transcripts immediately (or interim if debounce disabled)
        await self._process_transcript(transcript, smoothed_confidence, is_final, language, current_time)

    async def _process_transcript(
        self,
        transcript: str,
        confidence: Optional[float],
        is_final: bool,
        language: Optional[str],
        timestamp: float,
    ) -> None:
        """Internal method to process transcript and trigger interruption if needed.

        Args:
            transcript: The recognized text.
            confidence: Smoothed confidence value.
            is_final: Whether this is a final transcript.
            language: Optional language code.
            timestamp: Timestamp when transcript was received.
        """

        async with self._lock:
            should_interrupt_decision = self.should_interrupt(
                transcript=transcript, confidence=confidence, is_final=is_final
            )

            if not should_interrupt_decision:
                return

            # Rate limiting: check if we're within rate limit window
            time_since_last_interrupt = (timestamp - self._state.last_interruption_time) * 1000
            if time_since_last_interrupt < self._config.interrupt_rate_limit_ms:
                self._log_structured(
                    "rate_limited",
                    {
                        "reason": "rate_limited",
                        "transcript": transcript,
                        "confidence": confidence,
                        "is_final": is_final,
                        "language": language,
                        "time_since_last_ms": time_since_last_interrupt,
                        "rate_limit_ms": self._config.interrupt_rate_limit_ms,
                    },
                    level=logging.DEBUG,
                )
                return

            # Log accepted interruption
            self._log_structured(
                "accepted_interruption",
                {
                    "decision": "interrupt",
                    "reason": "meaningful_speech",
                    "transcript": transcript,
                    "confidence": confidence,
                    "is_final": is_final,
                    "language": language,
                    "agent_speaking": self._state.agent_speaking,
                    "timestamp": timestamp,
                },
                level=logging.INFO,
            )

            # Update last interruption time
            self._state.last_interruption_time = timestamp

            # Call session.interrupt() non-blocking
            try:
                fut = self._session.interrupt()
                if asyncio.iscoroutine(fut) or asyncio.isfuture(fut):
                    await fut
            except Exception as e:
                logger.error(f"Error calling session.interrupt(): {e}")

    def should_interrupt(self, transcript, confidence=None, is_final: bool = False) -> bool:
        """Decide whether the given transcript should interrupt the agent.

        Applies production-grade multilingual logic:
        - Language detection (English/Hindi/Mixed)
        - Robust multilingual text normalization
        - Multi-word phrase matching
        - Command word priority (overrides short text, low confidence, fillers)
        - Language-aware filler detection
        - Optional fuzzy matching
        - Interim transcript policy
        - Structured logging with language metadata

        Args:
            transcript: Raw transcript text.
            confidence: Optional confidence value (smoothed).
            is_final: Whether this is a final transcript.

        Returns:
            True if agent should be interrupted, False otherwise.
        """

        # Multilingual normalization and language detection
        normalized, detected_lang = _normalize_text_multilingual(transcript)

        if not normalized:
            self._log_structured(
                "ignored_empty",
                {
                    "reason": "empty_transcript",
                    "transcript": transcript,
                    "normalized": normalized,
                    "detected_language": detected_lang,
                },
                level=logging.DEBUG,
            )
            return False

        # Agent not speaking: no interruption needed
        if not self._state.agent_speaking:
            self._log_structured(
                "ignored_agent_silent",
                {
                    "reason": "agent_not_speaking",
                    "transcript": normalized,
                    "detected_language": detected_lang,
                },
                level=logging.DEBUG,
            )
            return False

        # Tokenize
        tokens = _tokenize(normalized)

        # CRITICAL: Check for command words FIRST (before short-text filter)
        # This fixes the "no" command issue and ensures all commands work regardless of length
        matched_command = _match_command_in_list(
            normalized, self._config.command_words, use_fuzzy=self._config.use_fuzzy_matching
        )

        if matched_command:
            self._log_structured(
                "command_detected",
                {
                    "decision": "interrupt",
                    "reason": "command_word",
                    "transcript": normalized,
                    "command": matched_command,
                    "detected_language": detected_lang,
                    "tokens": tokens,
                },
                level=logging.INFO,
            )
            return True

        # Very short transcripts (1-2 chars): treat as filler (commands already checked above)
        if len(normalized) <= 2:
            self._log_structured(
                "ignored_short",
                {
                    "reason": "very_short_transcript",
                    "transcript": normalized,
                    "length": len(normalized),
                    "detected_language": detected_lang,
                },
                level=logging.DEBUG,
            )
            return False

        # Interim transcript policy: conservative by default
        # Commands already handled above, so only non-commands reach here
        if not is_final and self._config.interim_interrupt_policy == "conservative":
            self._log_structured(
                "ignored_interim_conservative",
                {
                    "reason": "interim_conservative_policy",
                    "transcript": normalized,
                    "is_final": is_final,
                    "detected_language": detected_lang,
                },
                level=logging.DEBUG,
            )
            return False

        # Low confidence filtering (commands already checked above)
        if confidence is not None and confidence < self._config.min_confidence:
            self._log_structured(
                "ignored_low_confidence",
                {
                    "reason": "low_confidence",
                    "transcript": normalized,
                    "confidence": confidence,
                    "min_confidence": self._config.min_confidence,
                    "detected_language": detected_lang,
                },
                level=logging.DEBUG,
            )
            return False

        # Language-aware filler detection
        # Select appropriate filler list based on language mode and detection
        if self._config.language_mode == "en":
            active_fillers = self._config.ignored_words_en
        elif self._config.language_mode == "hi":
            active_fillers = self._config.ignored_words_hi
        else:  # auto or multi
            # Use detected language to select appropriate filler list
            if detected_lang == "hi":
                active_fillers = self._config.ignored_words_hi + self._config.ignored_words_en
            elif detected_lang == "multi":
                active_fillers = self._config.ignored_words_en + self._config.ignored_words_hi
            else:
                active_fillers = self._config.ignored_words

        ignored_set = set(w.lower() for w in active_fillers)
        is_filler_only = True
        meaningful_tokens = []

        for token in tokens:
            # Check exact match
            if token in ignored_set:
                continue

            # Check transliteration mapping for Hindi
            if token in HINDI_TRANSLITERATION_MAP:
                # Treat transliterated Hindi fillers as fillers
                mapped = HINDI_TRANSLITERATION_MAP[token]
                if mapped in ignored_set or token in self._config.ignored_words_hi:
                    continue

            # Check fuzzy match if enabled
            if self._config.use_fuzzy_matching and _fuzzy_match_token(
                token, active_fillers, threshold=75
            ):
                continue

            # Not a filler word
            is_filler_only = False
            meaningful_tokens.append(token)

        if is_filler_only and tokens:
            self._log_structured(
                "ignored_filler",
                {
                    "reason": "filler_only",
                    "transcript": normalized,
                    "tokens": tokens,
                    "detected_language": detected_lang,
                    "match_type": "filler",
                },
                level=logging.INFO,
            )
            return False

        # Meaningful speech detected
        self._log_structured(
            "meaningful_speech_detected",
            {
                "decision": "interrupt",
                "reason": "meaningful_speech",
                "transcript": normalized,
                "tokens": tokens,
                "meaningful_tokens": meaningful_tokens,
                "detected_language": detected_lang,
                "match_type": "meaningful",
            },
            level=logging.INFO,
        )
        return True

    def _log_structured(self, event_name: str, data: Dict[str, Any], level: int = logging.INFO) -> None:
        """Log structured data in JSON-friendly format.

        Args:
            event_name: Name of the event being logged.
            data: Dictionary of structured log data.
            level: Logging level (e.g., logging.INFO, logging.DEBUG).
        """

        try:
            log_entry = {"event": event_name, **data}
            logger.log(level, json.dumps(log_entry))
        except Exception as e:
            logger.error(f"Error logging structured data: {e}")

    def update_config(self, **kwargs) -> None:
        """Update the handler configuration at runtime.

        Supports atomic updates to all configuration options including:
        ignored_words, command_words, min_confidence, interim_interrupt_policy,
        debounce_ms, confidence_smoothing_alpha, use_fuzzy_matching, log_verbosity,
        interrupt_rate_limit_ms.

        Args:
            **kwargs: Configuration parameters to update.
        """

        if "ignored_words" in kwargs and kwargs["ignored_words"] is not None:
            self._config.ignored_words = [
                str(w).lower() for w in kwargs["ignored_words"]  # type: ignore[union-attr]
            ]

        if "command_words" in kwargs and kwargs["command_words"] is not None:
            self._config.command_words = [
                str(w).lower() for w in kwargs["command_words"]  # type: ignore[union-attr]
            ]

        if "min_confidence" in kwargs and kwargs["min_confidence"] is not None:
            try:
                self._config.min_confidence = float(kwargs["min_confidence"])
            except (TypeError, ValueError):
                logger.debug("invalid min_confidence value provided; keeping existing setting")

        if "interim_interrupt_policy" in kwargs and kwargs["interim_interrupt_policy"] is not None:
            if kwargs["interim_interrupt_policy"] in ("conservative", "aggressive"):
                self._config.interim_interrupt_policy = kwargs["interim_interrupt_policy"]

        if "debounce_ms" in kwargs and kwargs["debounce_ms"] is not None:
            try:
                self._config.debounce_ms = max(0, int(kwargs["debounce_ms"]))
            except (TypeError, ValueError):
                logger.debug("invalid debounce_ms value; keeping existing setting")

        if "confidence_smoothing_alpha" in kwargs and kwargs["confidence_smoothing_alpha"] is not None:
            try:
                self._config.confidence_smoothing_alpha = max(
                    0.0, min(1.0, float(kwargs["confidence_smoothing_alpha"]))
                )
            except (TypeError, ValueError):
                logger.debug("invalid confidence_smoothing_alpha value; keeping existing setting")

        if "use_fuzzy_matching" in kwargs and kwargs["use_fuzzy_matching"] is not None:
            self._config.use_fuzzy_matching = bool(kwargs["use_fuzzy_matching"]) and RAPIDFUZZ_AVAILABLE

        if "log_verbosity" in kwargs and kwargs["log_verbosity"] is not None:
            verbosity = str(kwargs["log_verbosity"]).upper()
            if verbosity in ("DEBUG", "INFO"):
                self._config.log_verbosity = verbosity
                logger.setLevel(logging.DEBUG if verbosity == "DEBUG" else logging.INFO)

        if "interrupt_rate_limit_ms" in kwargs and kwargs["interrupt_rate_limit_ms"] is not None:
            try:
                self._config.interrupt_rate_limit_ms = max(0, int(kwargs["interrupt_rate_limit_ms"]))
            except (TypeError, ValueError):
                logger.debug("invalid interrupt_rate_limit_ms value; keeping existing setting")

        if "language_mode" in kwargs and kwargs["language_mode"] is not None:
            if kwargs["language_mode"] in ("en", "hi", "auto", "multi"):
                self._config.language_mode = kwargs["language_mode"]
                logger.info(f"Language mode updated to: {self._config.language_mode}")
