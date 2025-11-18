# extensions/interrupt_handler/config.py
import os

def get_env_list(name, default="uh,umm,hmm,haan"):
    return [w.strip().lower() for w in os.getenv(name, default).split(",") if w.strip()]

IGNORED_WORDS = set(get_env_list("IGNORED_WORDS", "uh,umm,hmm,haan"))
ASR_CONFIDENCE_THRESHOLD = float(os.getenv("ASR_CONFIDENCE_THRESHOLD", "0.6"))
COMMAND_KEYWORDS = set(get_env_list("COMMAND_KEYWORDS", "stop,wait,pause,hold,no"))
