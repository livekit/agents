"""Session-scoped pytest fixture that bootstraps a LiveKit server for integration tests.

On macOS: runs `livekit-server --dev` as a subprocess.
On Linux: runs `docker run --rm livekit/livekit-server --dev` as a subprocess.
If port 7880 is already in use, assumes a server is already running and yields immediately.
"""

from __future__ import annotations

import platform
import socket
import subprocess
import time
import urllib.error
import urllib.request

import pytest

LK_URL = "ws://localhost:7880"
LK_API_KEY = "devkey"
LK_API_SECRET = "secret"

_PORT = 7880


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_ready(port: int, timeout: float = 15.0) -> None:
    """Wait until the LiveKit server is ready to accept connections.

    Polls the HTTP endpoint on the same port as a readiness proxy. Any HTTP
    response (including 4xx/5xx) confirms that the server's handler stack is
    fully initialised — a stronger signal than TCP reachability alone, which
    can return true before the WebSocket acceptor is ready, causing transient
    ConnectError failures in tests that open multiple rapid connections.
    """
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    req = urllib.request.Request(f"http://127.0.0.1:{port}", method="HEAD")
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=0.5):
                return  # any HTTP response means the server is up
        except urllib.error.HTTPError as exc:
            exc.close()  # HTTPError is a response object; close to avoid FD leak
            return  # a 4xx/5xx reply still means the server is ready
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            time.sleep(0.3)
    raise TimeoutError(
        f"LiveKit server did not become ready within {timeout}s (port {port})"
        + (f": last error: {last_exc!r}" if last_exc else "")
    )


@pytest.fixture(scope="session")
def livekit_server():
    if _port_in_use(_PORT):
        yield
        return

    system = platform.system()
    if system == "Darwin":
        proc = subprocess.Popen(
            ["livekit-server", "--dev"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        proc = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "host",
                "livekit/livekit-server",
                "--dev",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    try:
        _wait_for_ready(_PORT)
        yield
    finally:
        proc.terminate()
        proc.wait(timeout=10)
