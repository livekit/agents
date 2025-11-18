# extensions/interrupt_handler/config.py
import os


def get_env_list(name: str, default: str = "uh,umm,hmm,haan") -> list[str]:
    raw = os.getenv(name, default)
    return [w.strip().lower() for w in raw.split(",") if w.strip()]


IGNORED_WORDS: set[str] = set(get_env_list("IGNORED_WORDS", "uh,umm,hmm,haan"))
ASR_CONFIDENCE_THRESHOLD: float = float(os.getenv("ASR_CONFIDENCE_THRESHOLD", "0.6"))
COMMAND_KEYWORDS: set[str] = set(get_env_list("COMMAND_KEYWORDS", "stop,wait,pause,hold,no"))
