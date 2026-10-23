from __future__ import annotations

import os

from ..log import logger

HOSTED_LOGS_ENV = "LIVEKIT_HOSTED_LOGS"
HOSTED_LOGS_SIDECAR = "sidecar"
HOSTED_LOGS_HANDOFF_FILE_ENV = "LIVEKIT_HOSTED_LOGS_HANDOFF_FILE"


def _handoff_file() -> str | None:
    if os.environ.get(HOSTED_LOGS_ENV) != HOSTED_LOGS_SIDECAR:
        return None
    return os.environ.get(HOSTED_LOGS_HANDOFF_FILE_ENV) or None


def logs_handled_by_sidecar() -> bool:
    path = _handoff_file()
    return path is not None and os.path.exists(path)


def hand_off_logs_to_sidecar() -> None:
    path = _handoff_file()
    if path is None:
        return
    try:
        with open(path, "a"):
            pass
    except OSError:
        logger.warning(
            "could not create the hosted log hand-off file, the SDK keeps exporting logs",
            extra={"path": path},
            exc_info=True,
        )
