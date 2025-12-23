# interruption/config.py

IGNORE_WORDS = {
    "yeah", "ok", "okay", "hmm", "uh-huh", "right",
    "aha", "mhm", "yep", "yes"
}

HARD_WORDS = {
    "stop", "wait", "no", "cancel", "hold"
}

# Small audio drain buffer to avoid late VAD races
TTS_DRAIN_SECONDS = 0.15
