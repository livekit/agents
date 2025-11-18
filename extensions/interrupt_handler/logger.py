# extensions/interrupt_handler/logger.py
import json, sys, datetime

def log(kind, **data):
    entry = {"time": datetime.datetime.utcnow().isoformat() + "Z", "kind": kind}
    entry.update(data)
    print(json.dumps(entry), file=sys.stdout, flush=True)
