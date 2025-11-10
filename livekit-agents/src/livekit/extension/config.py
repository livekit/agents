import os
from dataclasses import dataclass, field

def _split_env_list(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [w.strip().lower() for w in raw.split(",") if w.strip()]

@dataclass
class InterruptConfig:
    # Words treated as fillers WHILE the agent is speaking
    ignored_words: set[str] = field(default_factory=lambda: set(_split_env_list(
        "IGNORED_WORDS", "uh,umm,um,hmm,haan,mm,uhh,erm,eh, mmhmm, uhuh"
    )))

    # Words/phrases that must ALWAYS interrupt (subset matched within text)
    priority_words: set[str] = field(default_factory=lambda: set(_split_env_list(
        "PRIORITY_WORDS", "stop,wait,hold on,pause,no,not that,one second"
    )))

    # Minimum average confidence to treat a chunk as real speech while agent is speaking
    asr_conf_min: float = float(os.getenv("ASR_CONF_MIN", "0.55"))

    # When agent is speaking, ignore chunk if ≥ this fraction are fillers and no priority words
    filler_ratio_min: float = float(os.getenv("FILLER_RATIO_MIN", "0.7"))

    # If True, log detailed decisions
    debug_logging: bool = os.getenv("INTERRUPT_DEBUG", "1") == "1"
