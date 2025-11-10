import os
import json
from typing import List, Set


# ----------------------------------------
# Helpers
# ----------------------------------------

def _env_list(name: str, default: List[str]) -> List[str]:
    """
    Reads environment variables that may contain:
    - JSON list:   ["uh","umm"]
    - comma list:  uh,umm,hmm
    """
    v = os.getenv(name)
    if not v:
        return default

    try:
        return json.loads(v) if v.strip().startswith("[") \
            else [x.strip() for x in v.split(",")]
    except Exception:
        return default


def _load_word_file(path: str) -> Set[str]:
    """
    Loads filler/interruption words from a text file.
    Each line is a separate word/phrase.
    """
    if not os.path.exists(path):
        return set()

    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if line:
                out.add(line)
    return out


def _load_all_word_files(folder: str) -> Set[str]:
    """
    Loads every .txt file inside the filler_words/ folder.
    """
    words = set()
    if not os.path.isdir(folder):
        return words

    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            words |= _load_word_file(os.path.join(folder, fname))

    return words


# ----------------------------------------
# File-based filler loading
# ----------------------------------------

BASE_DIR = os.path.join(os.path.dirname(__file__), "filler_words")

FILE_FILLERS = _load_all_word_files(BASE_DIR)


# ----------------------------------------
# Final IGNORED_FILLERS
# ----------------------------------------

DEFAULT_ENV_FILLERS = [
    "uh", "umm", "hmm", "haan", "okay", "hmm okay"
]

IGNORED_FILLERS: Set[str] = (
    {w.lower() for w in _env_list("IGNORED_FILLERS", DEFAULT_ENV_FILLERS)}
    | FILE_FILLERS
)


# ----------------------------------------
# Interrupt Commands
# ----------------------------------------

DEFAULT_COMMANDS = ["stop", "wait", "hold on", "pause"]

INTERRUPT_COMMANDS: Set[str] = {
    w.lower() for w in _env_list("INTERRUPT_COMMANDS", DEFAULT_COMMANDS)
}


# ----------------------------------------
# Thresholds
# ----------------------------------------

MIN_CONFIDENCE_AGENT_SPEAKING = float(
    os.getenv("MIN_CONFIDENCE_AGENT_SPEAKING", "0.35")
)

SHORT_SEGMENT_TOKENS = int(
    os.getenv("SHORT_SEGMENT_TOKENS", "5")
)
