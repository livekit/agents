"""Record every observability request byte-for-byte, then forward it to LiveKit Cloud.

Usage: capture_proxy.py <port> <upstream base url> <capture dir>
"""

import http.server
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT, UPSTREAM, OUT = int(sys.argv[1]), sys.argv[2].rstrip("/"), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()
_seq = [0]

FORWARDED = ("authorization", "content-type", "content-encoding", "user-agent", "accept")


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with _lock:
            _seq[0] += 1
            n = _seq[0]
        name = self.path.strip("/").replace("/", "_").split("?")[0]
        stem = OUT / f"{n:04d}_{name}"
        stem.with_suffix(".bin").write_bytes(body)
        headers = {k: v for k, v in self.headers.items() if k.lower() in FORWARDED}
        meta = {
            "seq": n,
            "time": time.time(),
            "path": self.path,
            "headers": {k: v for k, v in headers.items() if k.lower() != "authorization"},
        }
        req = urllib.request.Request(
            UPSTREAM + self.path, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                status, rbody, rtype = (
                    resp.status,
                    resp.read(),
                    resp.headers.get("Content-Type", ""),
                )
        except urllib.error.HTTPError as e:
            status, rbody, rtype = e.code, e.read(), e.headers.get("Content-Type", "")
        except (
            Exception
        ) as e:  # upstream unreachable: still acknowledge, the capture is what matters
            status, rbody, rtype = 200, b"", "application/x-protobuf"
            meta["upstream_error"] = repr(e)
        meta["upstream_status"] = status
        stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        self.send_response(status)
        if rtype:
            self.send_header("Content-Type", rtype)
        self.send_header("Content-Length", str(len(rbody)))
        self.end_headers()
        self.wfile.write(rbody)

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("proxy: " + (fmt % args) + "\n")


http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
