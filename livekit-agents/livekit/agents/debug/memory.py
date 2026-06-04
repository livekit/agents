"""Helpers for working with memray captures from agent workers.

The framework can run with ``multiprocessing_context="fork"``, which lets
``memray run --follow-fork`` capture the worker plus every job/inference child
in one shot. memray writes files named like::

    memray-agent.py.<parent_pid>.bin             # the worker (parent)
    memray-agent.py.<parent_pid>.bin.<child_pid> # one per fork

memray names captures by pid only, so to label a ``.bin`` as job / inference /
worker this module reads the per-process manifests the framework writes
alongside the captures (``livekit-proc-<parent_pid>-<pid>.json``). See
``debug/README.md`` for the end-to-end workflow.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._proc_manifest import ProcInfo, load_manifests

# memray --follow-fork filenames look like:
#   memray-<script>.<parent_pid>.bin            (parent capture)
#   memray-<script>.<parent_pid>.bin.<child_pid> (child capture)
_FILENAME_RE = re.compile(r"^memray-(?P<script>.+?)\.(?P<parent>\d+)\.bin(?:\.(?P<child>\d+))?$")


@dataclass
class Capture:
    path: Path
    script: str
    parent_pid: int
    child_pid: int | None  # None means this *is* the parent (worker)
    info: ProcInfo | None = field(default=None)

    @property
    def is_parent(self) -> bool:
        return self.child_pid is None

    @property
    def pid(self) -> int:
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


def discover(directory: Path) -> list[Capture]:
    """Find every memray capture in ``directory``, annotated with manifest info."""
    proc_info = load_manifests(directory)
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
    # group by parent pid; within a group, worker (no child_pid) first then children ascending
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


def _cmd_list(directory: Path) -> int:
    caps = discover(directory)
    if not caps:
        print(f"no memray-*.bin captures found in {directory}")
        return 0
    unknowns = sum(1 for c in caps if c.kind == "unknown")
    print(f"{len(caps)} capture(s) in {directory}:")
    if unknowns:
        print(
            f"  ({unknowns} capture(s) without a livekit-proc-*.json manifest — "
            "the agent didn't run from this directory or wrote them elsewhere)"
        )
    for c in caps:
        size = _human_size(c.path.stat().st_size)
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(c.path.stat().st_mtime))
        print(f"  {c.label:<44} {size:>8}  {when}  {c.path.name}")
    return 0


def _cmd_report(directory: Path) -> int:
    if _load_memray() is None:
        print("memray is not installed. Install with: uv pip install memray", file=sys.stderr)
        return 1
    caps = discover(directory)
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


def _cmd_serve(directory: Path, port: int) -> int:
    import functools
    import http.server
    import socketserver

    if _load_memray() is not None:
        _cmd_report(directory)
    else:
        rendered = [
            (c, c.path.with_name(c.path.name + ".flamegraph.html")) for c in discover(directory)
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

    p_report = sub.add_parser("report", help="render flamegraphs + an index.html for all captures")
    _add_common(p_report)

    p_list = sub.add_parser("list", help="list captures grouped by worker")
    _add_common(p_list)

    p_serve = sub.add_parser("serve", help="render + serve the captures over HTTP")
    _add_common(p_serve)
    p_serve.add_argument("--port", type=int, default=8042)

    args = parser.parse_args(argv)
    directory = Path(args.dir) if getattr(args, "dir", None) else Path.cwd()

    if args.cmd == "list":
        return _cmd_list(directory)
    if args.cmd == "serve":
        return _cmd_serve(directory, args.port)
    if args.cmd == "report":
        return _cmd_report(directory)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
