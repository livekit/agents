from __future__ import annotations

import time
import uuid


def time_ms() -> int:
    return int(time.time() * 1000)


def shortuuid(prefix: str = "") -> str:
    return prefix + str(uuid.uuid4().hex)[:12]
