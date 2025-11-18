# extensions/interrupt_handler/logger.py
import datetime
import json
import sys
from typing import Any


def log(kind: str, **data: Any) -> None:
    entry = {"time": datetime.datetime.utcnow().isoformat() + "Z", "kind": kind}
    entry.update(data)
    print(json.dumps(entry), file=sys.stdout, flush=True)
