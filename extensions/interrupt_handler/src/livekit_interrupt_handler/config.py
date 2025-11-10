import os
from typing import List

# Config read from environment (or .env)
IGNORED_WORDS_DEFAULT = ["uh", "umm", "hmm", "haan"]

def get_env_list(key: str, default: List[str]):
    val = os.getenv(key)
    if not val:
        return default
    # comma separated
    return [x.strip().lower() for x in val.split(",") if x.strip()]

class Config:
    ignored_words = get_env_list("IGNORED_WORDS", IGNORED_WORDS_DEFAULT)
    # commands that should always interrupt (lowercased)
    always_interrupt_commands = get_env_list(
        "ALWAYS_INTERRUPT_COMMANDS",
        ["stop", "wait", "pause", "hold", "no"]
    )
    # treat as interruption only if ASR confidence >= this value; else ignore (when agent speaking)
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
    # if True, use strict token matching for fillers; otherwise substring matches allowed
    strict_filler_match = os.getenv("STRICT_FILLER_MATCH", "true").lower() in ("1","true","yes")
