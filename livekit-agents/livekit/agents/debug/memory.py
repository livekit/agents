"""Helpers for working with memray captures from agent workers.

The framework can run with `multiprocessing_context="fork"`, which lets
`memray run --follow-fork` capture the worker plus every job/inference child
in one shot. memray writes files named like::

    memray-agent.py.<parent_pid>.bin             # the worker (parent)
    memray-agent.py.<parent_pid>.bin.<child_pid> # one per fork

memray names captures by pid only, so to label a ``.bin`` as job / inference /
worker this module joins them against the framework's JSON memory logs
(``--logs``), which record ``pid`` + ``process_kind`` for every process. It
renders all captures at once into an index grouped by parent worker. See
``debug/README.md`` for the end-to-end workflow.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# memray --follow-fork filenames look like:
#   memray-<script>.<parent_pid>.bin            (parent capture)
#   memray-<script>.<parent_pid>.bin.<child_pid> (child capture)
_FILENAME_RE = re.compile(r"^memray-(?P<script>.+?)\.(?P<parent>\d+)\.bin(?:\.(?P<child>\d+))?$")


@dataclass(frozen=True)
class ProcInfo:
    """What the framework logged about a pid (see worker / supervised_proc)."""

    kind: str  # "worker" | "job" | "inference"
    job_id: str | None = None
    room_id: str | None = None
    last_memory_mb: float | None = None


@dataclass
class Capture:
    path: Path
    script: str
    parent_pid: int
    child_pid: int | None  # None means this *is* the parent (worker)
    info: ProcInfo | None = field(default=None)  # filled in from logs, if available

    @property
    def is_parent(self) -> bool:
        return self.child_pid is None

    @property
    def pid(self) -> int:
        """The pid this capture belongs to (the worker's, or the forked child's)."""
        return self.parent_pid if self.is_parent else self.child_pid  # type: ignore[return-value]

    @property
    def kind(self) -> str:
        if self.info is not None:
            return self.info.kind
        return "worker" if self.is_parent else "unknown"

    @property
    def label(self) -> str:
        extra = ""
        if self.info and self.info.job_id:
            extra = f"  job={self.info.job_id}"
        return f"{self.kind} pid={self.pid}{extra}"


def load_proc_info(log_paths: Sequence[Path]) -> dict[int, ProcInfo]:
    """Build a ``pid -> ProcInfo`` map from the worker's JSON memory logs.

    The framework logs a line per process with ``memory_event``, ``pid`` and
    ``process_kind`` (and ``job_id`` / ``room_id`` for jobs). memray names its
    capture files by pid only, so this is what lets a ``.bin`` be labelled
    job / inference / worker. Lines that aren't JSON are ignored; if nothing
    parses, the caller falls back to pid-only grouping.
    """
    out: dict[int, ProcInfo] = {}
    for log_path in log_paths:
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "memory_event" not in rec or "pid" not in rec or "process_kind" not in rec:
                continue
            try:
                pid = int(rec["pid"])
            except (TypeError, ValueError):
                continue
            out[pid] = ProcInfo(
                kind=str(rec["process_kind"]),
                job_id=rec.get("job_id"),
                room_id=rec.get("room_id"),
                last_memory_mb=rec.get("memory_usage_mb"),
            )
    return out


def discover(directory: Path, proc_info: dict[int, ProcInfo] | None = None) -> list[Capture]:
    """Find every memray capture in ``directory``, parsed into Capture records.

    When ``proc_info`` (from :func:`load_proc_info`) is given, each capture is
    annotated with its process kind.
    """
    proc_info = proc_info or {}
    out: list[Capture] = []
    for p in directory.iterdir():
        m = _FILENAME_RE.match(p.name)
        if not m:
            continue
        cap = Capture(
            path=p,
            script=m["script"],
            parent_pid=int(m["parent"]),
            child_pid=int(m["child"]) if m["child"] else None,
        )
        cap.info = proc_info.get(cap.pid)
        out.append(cap)
    # group by parent pid; within a group, the worker (no child_pid) first then
    # children in ascending pid order — gives a stable, fork-order-ish listing
    out.sort(key=lambda c: (c.parent_pid, c.child_pid if c.child_pid is not None else -1))
    return out


def _human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.0f}{unit}" if unit == "B" else f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}GB"


def _load_memray() -> Any | None:
    try:
        import memray
    except Exception:
        return None
    return memray


def _flamegraph(bin_path: Path) -> Path | None:
    """Render one capture with memray's own reporter. Returns the HTML path."""
    html_path = bin_path.with_name(bin_path.name + ".flamegraph.html")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "memray",
            "flamegraph",
            "--force",
            "-o",
            str(html_path),
            str(bin_path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"  ! failed to render {bin_path.name}: {proc.stderr.strip()}", file=sys.stderr)
        return None
    return html_path


def _write_index(directory: Path, rendered: Sequence[tuple[Capture, Path | None]]) -> Path:
    template = (Path(__file__).parent / "index.html").read_text()
    rows: list[str] = []
    current_parent: int | None = None
    for cap, html in rendered:
        # emit a group header row the first time we see a parent pid, and every
        # time it changes thereafter (rendered is already sorted by parent_pid)
        if current_parent is None or cap.parent_pid != current_parent:
            current_parent = cap.parent_pid
            rows.append(
                f'<tr class="group"><td colspan="4">'
                f"worker pid={cap.parent_pid} &nbsp; script={cap.script}"
                f"</td></tr>"
            )
        size = _human_size(cap.path.stat().st_size)
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cap.path.stat().st_mtime))
        role = f"{cap.kind} pid={cap.pid}"
        if cap.info and cap.info.job_id:
            role += f" &nbsp; <code>job={cap.info.job_id}</code>"
        if html is not None:
            link = f'<a href="{html.name}">{role}</a>'
        else:
            link = f"{role} <em>(render failed)</em>"
        rows.append(
            f"<tr><td>{link}</td><td>{size}</td><td>{when}</td>"
            f"<td><code>{cap.path.name}</code></td></tr>"
        )
    body = "\n".join(rows) if rows else '<tr><td colspan="4"><em>no captures found</em></td></tr>'
    index = directory / "index.html"
    index.write_text(template.replace("<!--CAPTURES-->", body))
    return index


def _cmd_list(directory: Path, proc_info: dict[int, ProcInfo]) -> int:
    caps = discover(directory, proc_info)
    if not caps:
        print(f"no memray-*.bin captures found in {directory}")
        return 0
    print(f"{len(caps)} capture(s) in {directory}:")
    if not proc_info:
        print("  (no --logs given; kinds shown as 'unknown' — pass the worker log to label them)")
    for c in caps:
        size = _human_size(c.path.stat().st_size)
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(c.path.stat().st_mtime))
        print(f"  {c.label:<44} {size:>8}  {when}  {c.path.name}")
    return 0


def _cmd_report(directory: Path, proc_info: dict[int, ProcInfo]) -> int:
    if _load_memray() is None:
        print("memray is not installed. Install with: uv pip install memray", file=sys.stderr)
        return 1
    caps = discover(directory, proc_info)
    if not caps:
        print(f"no memray-*.bin captures found in {directory}")
        return 0
    print(f"rendering {len(caps)} flamegraph(s) from {directory} ...")
    rendered = [(c, _flamegraph(c.path)) for c in caps]
    index = _write_index(directory, rendered)
    ok = sum(1 for _, h in rendered if h is not None)
    print(f"done: {ok}/{len(caps)} rendered")
    print(f"open {index}, or run: python -m livekit.agents.debug.memory serve {directory}")
    return 0


def _cmd_serve(directory: Path, proc_info: dict[int, ProcInfo], port: int) -> int:
    import functools
    import http.server
    import socketserver

    if _load_memray() is not None:
        _cmd_report(directory, proc_info)
    else:
        # no memray, but maybe flamegraph HTMLs already exist
        rendered = [
            (c, c.path.with_name(c.path.name + ".flamegraph.html"))
            for c in discover(directory, proc_info)
        ]
        _write_index(directory, [(c, h if h.exists() else None) for c, h in rendered])

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"serving {directory} at http://localhost:{port}/  (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m livekit.agents.debug.memory",
        description="Render and browse memray --follow-fork captures.",
    )
    sub = parser.add_subparsers(dest="cmd")

    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("dir", nargs="?", help="directory of captures (default: cwd)")
        p.add_argument(
            "--logs",
            action="append",
            default=[],
            metavar="PATH",
            help="JSON worker log file(s) used to label captures job/inference/worker",
        )

    p_report = sub.add_parser("report", help="render flamegraphs + an index.html for all captures")
    _add_common(p_report)

    p_list = sub.add_parser("list", help="list captures grouped by worker")
    _add_common(p_list)

    p_serve = sub.add_parser("serve", help="render + serve the captures over HTTP")
    _add_common(p_serve)
    p_serve.add_argument("--port", type=int, default=8042)

    args = parser.parse_args(argv)
    directory = Path(args.dir) if getattr(args, "dir", None) else Path.cwd()
    proc_info = load_proc_info([Path(p) for p in getattr(args, "logs", [])])

    if args.cmd == "list":
        return _cmd_list(directory, proc_info)
    if args.cmd == "serve":
        return _cmd_serve(directory, proc_info, args.port)
    if args.cmd == "report":
        return _cmd_report(directory, proc_info)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
