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
    """Wait until the LiveKit server is fully ready to accept WebSocket connections.

    Polls the HTTP endpoint on the same port rather than just checking TCP
    reachability. The TCP port binding precedes full WebSocket handler
    initialisation, so a raw port check can allow tests to connect before the
    server is truly ready, causing transient ConnectError failures.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}", timeout=0.5)
            return  # any HTTP response means the server is up
        except urllib.error.HTTPError:
            return  # a 4xx/5xx reply still means the server is ready
        except Exception:
            time.sleep(0.3)
    raise TimeoutError(f"LiveKit server did not become ready within {timeout}s (port {port})")


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
