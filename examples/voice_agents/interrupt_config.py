# interrupt_config.py

"""
Configuration for interrupt / backchannel word lists.

Change SOFT_WORDS and HARD_WORDS here without touching the agent logic.
All words must be lowercase because transcripts are lowercased before matching.
"""

SOFT_WORDS: set[str] = {
    "yeah",
    "yea",
    "yah",
    "ya",
    "ok",
    "okay",
    "k",
    "kk",
    "hmm",
    "mm",
    "mmm",
    "uh",
    "uhh",
    "uhm",
    "um",
    "huh",
    "right",
    "yep",
    "yup",
    "uh-huh",
    "uhhuh",
}

HARD_WORDS: set[str] = {
    "stop",
    "wait",
    "no",
    "nope",
    "cancel",
    "pause",
    "hold",
}
