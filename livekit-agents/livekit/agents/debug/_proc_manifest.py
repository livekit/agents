"""Per-process manifests so the debug.memory CLI can label memray captures.

memray names its captures by pid only. To tell a `.bin` apart as worker / job /
inference, every process writes a tiny JSON file when it starts (and again when
a job is assigned, to record the job_id). The CLI reads those files instead of
trying to parse the framework's stdout log stream.

Files are named ``livekit-proc-<parent_pid>-<pid>.json`` so they collide cleanly
across runs (a re-used pid in a different worker invocation won't overwrite an
old manifest from a different parent).

The default output directory matches where ``memray run --follow-fork`` writes:
the parent directory of the entrypoint script (``sys.argv[0]``). Override with
the ``LK_DEBUG_DIR`` environment variable.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MANIFEST_GLOB = "livekit-proc-*.json"


@dataclass(frozen=True)
class ProcInfo:
    pid: int
    parent_pid: int
    kind: str  # "worker" | "job" | "inference"
    job_id: str | None = None
    room_id: str | None = None
    started_at: float | None = None


def manifest_dir() -> Path:
    """Where to write manifests; matches memray --follow-fork's output dir."""
    if env := os.environ.get("LK_DEBUG_DIR"):
        return Path(env)
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0:
        try:
            return Path(argv0).resolve().parent
        except OSError:
            pass
    return Path.cwd()


def write_manifest(
    *, pid: int, parent_pid: int, kind: str, directory: Path | None = None, **extra: Any
) -> None:
    """Write ``{dir}/livekit-proc-<parent>-<pid>.json``. Silent on error.

    Best-effort: a permission failure or full disk should never break the agent.
    """
    try:
        d = directory or manifest_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"livekit-proc-{parent_pid}-{pid}.json"
        payload: dict[str, Any] = {
            "pid": pid,
            "parent_pid": parent_pid,
            "kind": kind,
            "started_at": time.time(),
            **extra,
        }
        path.write_text(json.dumps(payload))
    except OSError:
        pass


def load_manifests(directory: Path) -> dict[int, ProcInfo]:
    """Read every ``livekit-proc-*.json`` in ``directory`` into a pid -> ProcInfo map."""
    out: dict[int, ProcInfo] = {}
    if not directory.is_dir():
        return out
    for p in directory.glob(_MANIFEST_GLOB):
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        try:
            pid = int(rec["pid"])
            parent_pid = int(rec["parent_pid"])
            kind = str(rec["kind"])
        except (KeyError, TypeError, ValueError):
            continue
        out[pid] = ProcInfo(
            pid=pid,
            parent_pid=parent_pid,
            kind=kind,
            job_id=rec.get("job_id"),
            room_id=rec.get("room_id"),
            started_at=rec.get("started_at"),
        )
    return out
