import os

def get_config():
    ignored = os.getenv("IGNORED_WORDS", "uh,umm,hmm,haan").split(",")
    commands = os.getenv("COMMAND_WORDS", "stop,wait,no,hold").split(",")
    return {
        "ignored_words": [w.strip().lower() for w in ignored if w.strip()],
        "command_words": [w.strip().lower() for w in commands if w.strip()],
        "confidence_threshold": float(os.getenv("ASR_CONF_THRESHOLD", "0.6")),
    }
