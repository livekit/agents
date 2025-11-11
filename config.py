import os

# Default interruption filtering configuration.
# Import this module (or call apply()) before starting your agent to set env vars.

LIVEKIT_IGNORED_WORDS = "uh,umm,hmm,haan"
LIVEKIT_INTERRUPTION_KEYWORDS = "stop,wait,hold on,one second,no,not that"
LIVEKIT_MIN_ASR_CONFIDENCE = "0.5"


def apply() -> None:
    """Apply defaults for interruption filtering if not already set in the environment."""
    os.environ.setdefault("LIVEKIT_IGNORED_WORDS", LIVEKIT_IGNORED_WORDS)
    os.environ.setdefault("LIVEKIT_INTERRUPTION_KEYWORDS", LIVEKIT_INTERRUPTION_KEYWORDS)
    os.environ.setdefault("LIVEKIT_MIN_ASR_CONFIDENCE", LIVEKIT_MIN_ASR_CONFIDENCE)


# Auto-apply on import for convenience.
apply()
