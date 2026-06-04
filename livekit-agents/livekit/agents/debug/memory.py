"""Helpers for working with memray captures from agent workers.

The framework can run with `multiprocessing_context="fork"`, which lets
`memray run --follow-fork` capture the worker plus every job/inference child
in one shot. memray writes files named like::

    memray-agent.py.<parent_pid>.bin             # the worker (parent)
    memray-agent.py.<parent_pid>.bin.<child_pid> # one per fork

This module renders all of them at once and produces an index that groups
children under their parent so you can tell which capture belongs to which
process. See ``debug/README.md`` for the end-to-end workflow.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# memray --follow-fork filenames look like:
#   memray-<script>.<parent_pid>.bin            (parent capture)
#   memray-<script>.<parent_pid>.bin.<child_pid> (child capture)
_FILENAME_RE = re.compile(r"^memray-(?P<script>.+?)\.(?P<parent>\d+)\.bin(?:\.(?P<child>\d+))?$")


@dataclass(frozen=True)
class Capture:
    path: Path
    script: str
    parent_pid: int
    child_pid: int | None  # None means this *is* the parent (worker)

    @property
    def is_parent(self) -> bool:
        return self.child_pid is None

    @property
    def label(self) -> str:
        if self.is_parent:
            return f"worker pid={self.parent_pid}"
        return f"child pid={self.child_pid}  (parent={self.parent_pid})"


def discover(directory: Path) -> list[Capture]:
    """Find every memray capture in ``directory``, parsed into Capture records."""
    out: list[Capture] = []
    for p in directory.iterdir():
        m = _FILENAME_RE.match(p.name)
        if not m:
            continue
        out.append(
            Capture(
                path=p,
                script=m["script"],
                parent_pid=int(m["parent"]),
                child_pid=int(m["child"]) if m["child"] else None,
            )
        )
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
        role = "worker" if cap.is_parent else f"child pid={cap.child_pid}"
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
    print(f"{len(caps)} capture(s) in {directory}:")
    for c in caps:
        size = _human_size(c.path.stat().st_size)
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(c.path.stat().st_mtime))
        print(f"  {c.label:<40} {size:>8}  {when}  {c.path.name}")
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
        # no memray, but maybe flamegraph HTMLs already exist
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

    p_report = sub.add_parser("report", help="render flamegraphs + an index.html for all captures")
    p_report.add_argument("dir", nargs="?", help="directory of captures (default: cwd)")

    p_list = sub.add_parser("list", help="list captures grouped by worker")
    p_list.add_argument("dir", nargs="?")

    p_serve = sub.add_parser("serve", help="render + serve the captures over HTTP")
    p_serve.add_argument("dir", nargs="?")
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
