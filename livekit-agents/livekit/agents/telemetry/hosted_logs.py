from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from ..version import __version__

HOSTED_LOGS_ENV = "LIVEKIT_HOSTED_LOGS"
HOSTED_LOGS_SIDECAR = "sidecar"
HOSTED_LOGS_ATTR = "lk.hosted_logs"


def logs_handled_by_sidecar() -> bool:
    return os.environ.get(HOSTED_LOGS_ENV) == HOSTED_LOGS_SIDECAR


def announce_sidecar_log_handoff() -> None:
    if not logs_handled_by_sidecar():
        return
    line = {
        "message": "log export handed off to the hosted log sidecar",
        "level": "INFO",
        "name": "livekit.agents",
        HOSTED_LOGS_ATTR: HOSTED_LOGS_SIDECAR,
        "sdk": "python",
        "sdk_version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    sys.stdout.write(json.dumps(line) + "\n")
    sys.stdout.flush()
