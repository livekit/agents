# from dataclasses import dataclass, field
# from typing import Set, List


# @dataclass
# class InterruptConfig:
#     """
#     Config for filler-aware interruption handling.
#     """

#     # Pure fillers when agent is speaking
#     filler_words: Set[str] = field(default_factory=lambda: {
#         "uh", "umm", "um", "hmm", "haan", "hmmm", "huh",
#         "ah", "oh", "hmmmmm", "mmm",
#     })

#     # Max tokens for something to still count as "just filler"
#     max_filler_tokens: int = 3

#     # Explicit commands that should *always* interrupt when detected
#     hard_interrupt_phrases: List[str] = field(default_factory=lambda: [
#         "stop",
#         "wait",
#         "hold on",
#         "one second",
#         "listen",
#         "bas",
#         "ruk",
#         "ruko",
#         "thoda ruk",
#         "no not that one",
#         "not this",
#     ])

#     # Min non-filler tokens in speech (while agent is speaking) to treat it as real interruption
#     min_real_words_to_interrupt: int = 2

#     # If ASR confidence is provided: below this → ignore as noise
#     min_confidence: float = 0.55

#     # Enable logs
#     debug_logging: bool = True

from dataclasses import dataclass, field
from typing import Set, List, Iterable


@dataclass
class InterruptConfig:
    """
    Config for filler-aware interruption handling.
    Supports:
      - Dynamic updates at runtime (add/remove filler or interrupt phrases)
      - Multi-language filler detection (English + Hindi)
    """

    filler_words: Set[str] = field(default_factory=lambda: {
        # English fillers
        "uh", "umm", "um", "hmm", "hmmm", "huh", "ah", "oh", "mmm",
        # Hindi / Hinglish fillers
        "haan", "arey", "acha"
    })

    max_filler_tokens: int = 3

    hard_interrupt_phrases: List[str] = field(default_factory=lambda: [
        # English
        "stop",
        "wait",
        "hold on",
        "one second",
        "listen",
        "no not that one",
        "not this",
        # Hindi / Hinglish
        "bas",
        "ruk",
        "ruko",
        "thoda ruk",
        "thodi der ruk",
        "ek second"
    ])

    min_real_words_to_interrupt: int = 2
    min_confidence: float = 0.55
    debug_logging: bool = True

    # ---------- BONUS FEATURE: Dynamic runtime update support ----------

    def add_filler_words(self, words: Iterable[str]) -> None:
        """Add new filler words at runtime."""
        for w in words:
            w = w.strip().lower()
            if w:
                self.filler_words.add(w)

    def remove_filler_words(self, words: Iterable[str]) -> None:
        """Remove filler words at runtime."""
        for w in words:
            w = w.strip().lower()
            self.filler_words.discard(w)

    def add_hard_phrases(self, phrases: Iterable[str]) -> None:
        """Add new hard-interrupt phrases at runtime."""
        for p in phrases:
            p = p.strip().lower()
            if p and p not in self.hard_interrupt_phrases:
                self.hard_interrupt_phrases.append(p)

    def remove_hard_phrases(self, phrases: Iterable[str]) -> None:
        """Remove hard-interrupt phrases at runtime."""
        to_remove = {p.strip().lower() for p in phrases}
        self.hard_interrupt_phrases = [
            p for p in self.hard_interrupt_phrases if p not in to_remove
        ]
