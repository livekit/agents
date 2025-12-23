# interrupt_gate.py

from config import IGNORE_WORDS, HARD_WORDS


class InterruptionGate:
    """
    Decides whether user speech should:
    - IGNORE
    - INTERRUPT
    - RESPOND
    """

    def classify(self, text: str, is_agent_speaking: bool) -> str:
        clean = "".join(
            c for c in text.lower()
            if c.isalnum() or c in {" ", "-"}
        ).strip()

        if not clean:
            return "IGNORE"

        words = clean.split()

        # Agent silent → normal conversation
        if not is_agent_speaking:
            return "RESPOND"

        # Hard command anywhere → interrupt
        if any(w in HARD_WORDS for w in words):
            return "INTERRUPT"

        # Only backchannel words → ignore
        if all(w in IGNORE_WORDS for w in words):
            return "IGNORE"

        # Mixed input → interrupt
        return "INTERRUPT"
