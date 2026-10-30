import os

HOSTED_LOGS_ENV = "LIVEKIT_HOSTED_LOGS"
HOSTED_LOGS_ENABLED = "true"
HOSTED_LOGS_HANDOFF_FILE_ENV = "LIVEKIT_HOSTED_LOGS_HANDOFF_FILE"


def _handoff_file() -> str | None:
    if os.environ.get(HOSTED_LOGS_ENV) != HOSTED_LOGS_ENABLED:
        return None
    return os.environ.get(HOSTED_LOGS_HANDOFF_FILE_ENV) or None


def hand_off() -> None:
    path = _handoff_file()
    if path is None:
        return
    try:
        with open(path, "a"):
            pass
    except OSError:
        pass


def logs_handed_off() -> bool:
    path = _handoff_file()
    return path is not None and os.path.exists(path)


hand_off()
