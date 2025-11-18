# filler_agent/intent_detector.py
from __future__ import annotations

from typing import Set, Literal, Optional

Decision = Literal["ignore_filler", "interrupt_agent", "user_speech"]

# Words like "yeah" / "ok" we treat as light confirmations.
_CONFIRMATION_WORDS = {"yeah", "ya", "ok", "okay", "haan"}


def _preprocess(text: str) -> list[str]:
    """
    Lowercase and split text into simple word tokens.
    Example: "Umm, okay stop!" -> ["umm", "okay", "stop"]
    """
    text = text.strip().lower()
    if not text:
        return []
    tokens: list[str] = []
    for raw in text.split():
        token = raw.strip(".,!?")
        if token:
            tokens.append(token)
    return tokens


def classify_transcript(
    transcript: str,
    *,
    agent_speaking: bool,
    is_final: bool,
    ignored_filler_words: Set[str],
    interrupt_command_words: Set[str],
    confidence: Optional[float] = None,
) -> Decision:
    """
    Decide what to do with a piece of transcribed user speech.

    Returns one of:
      - "ignore_filler"
      - "interrupt_agent"
      - "user_speech"
    """
    tokens = _preprocess(transcript)

    # Nothing meaningful heard
    if not tokens:
        return "ignore_filler" if agent_speaking else "user_speech"

    # If the agent is not speaking, we always treat it as user speech.
    if not agent_speaking:
        # Requirement: even "umm" should count as speech when agent is quiet.
        return "user_speech"

    # Agent IS speaking from here down.
    has_command = any(t in interrupt_command_words for t in tokens)
    non_filler_tokens = [t for t in tokens if t not in ignored_filler_words]

    # 1) If we see any obvious command words (stop, wait, etc.) we interrupt.
    if has_command:
        return "interrupt_agent"

    # 2) If everything is in the filler list, ignore.
    if not non_filler_tokens:
        # Example: "uh", "umm", "hmm", "haan"
        return "ignore_filler"

    # 3) Handle short "hmm yeah" type acknowledgements with low confidence.
    if (
        len(non_filler_tokens) <= 2
        and all(t in _CONFIRMATION_WORDS for t in non_filler_tokens)
    ):
        # In a real system you'd use the ASR confidence,
        # here we treat short confirmations as ignorable.
        if confidence is None or confidence < 0.5:
            # Treat as background murmur: "hmm yeah"
            return "ignore_filler"

    # 4) Anything else while the agent is speaking is treated as a real interruption.
    # Example: "no not that one", "umm okay stop" (if "stop" isn't in interrupt list)
    return "interrupt_agent"
