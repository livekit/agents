from dataclasses import dataclass, field
from typing import Set, List


@dataclass
class InterruptConfig:
    """
    Config for filler-aware interruption handling.
    """

    # Pure fillers when agent is speaking
    filler_words: Set[str] = field(default_factory=lambda: {
        "uh", "umm", "um", "hmm", "haan", "hmmm", "huh",
        "ah", "oh", "hmmmmm", "mmm",
    })

    # Max tokens for something to still count as "just filler"
    max_filler_tokens: int = 3

    # Explicit commands that should *always* interrupt when detected
    hard_interrupt_phrases: List[str] = field(default_factory=lambda: [
        "stop",
        "wait",
        "hold on",
        "one second",
        "listen",
        "bas",
        "ruk",
        "ruko",
        "thoda ruk",
        "no not that one",
        "not this",
    ])

    # Min non-filler tokens in speech (while agent is speaking) to treat it as real interruption
    min_real_words_to_interrupt: int = 2

    # If ASR confidence is provided: below this → ignore as noise
    min_confidence: float = 0.55

    # Enable logs
    debug_logging: bool = True
