"""An isolated local SFU (``livekit-server --dev`` on free ports, private keys) and the room
helpers the room regression modules share."""

from __future__ import annotations

import asyncio
import gc
import json
import shutil
import socket
import subprocess
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pytest

from livekit import api, rtc

KEY = "APIregression"
SECRET = "regression-secret-regression-secret"
DATA_KEY = b"regression-shared-data-key-32byte"


@pytest.fixture(autouse=True)
def collect_closed_rooms() -> Iterator[None]:
    yield
    # SDK event adapters form cycles; release their handles while the FFI library is live.
    gc.collect()


def _port(kind: int = socket.SOCK_STREAM) -> int:
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


def _serve(executable: str, log_path: Path) -> tuple[subprocess.Popen[bytes], int] | None:
    """Starts a dev server on free ports; None when it exits first, as when another process took
    a port between probing and binding."""
    port = _port()
    config = {
        "port": port,
        "bind_addresses": ["127.0.0.1"],
        "rtc": {"tcp_port": _port(), "udp_port": _port(socket.SOCK_DGRAM)},
        "keys": {KEY: SECRET},
        "logging": {"level": "error"},
    }
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [executable, "--dev", "--node-ip", "127.0.0.1", "--config-body", json.dumps(config)],
            stdout=log,
            stderr=log,
        )
    deadline = time.monotonic() + 15
    while process.poll() is None:
        with socket.socket() as sock:
            sock.settimeout(0.1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return process, port
        if time.monotonic() >= deadline:
            process.kill()
            process.wait(timeout=5)
            raise AssertionError(log_path.read_text())
        time.sleep(0.05)
    return None


@pytest.fixture(scope="module")
def sfu(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Own a dev server on free signaling, TCP and UDP ports, with private test keys."""
    executable = shutil.which("livekit-server")
    if executable is None:
        pytest.fail("room regressions require livekit-server on PATH")
    log_path = tmp_path_factory.mktemp("regression-sfu") / "server.log"
    for _ in range(3):
        if started := _serve(executable, log_path):
            break
    else:
        pytest.fail(log_path.read_text())
    process, port = started
    try:
        yield f"ws://127.0.0.1:{port}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def room_token(name: str, identity: str, *, hidden: bool = False) -> str:
    return (
        api.AccessToken(KEY, SECRET)
        .with_identity(identity)
        .with_grants(
            api.VideoGrants(room_join=True, room=name, hidden=hidden, can_update_own_metadata=True)
        )
        .to_jwt()
    )


@asynccontextmanager
async def connected(
    url: str,
    name: str,
    identity: str,
    *,
    hidden: bool = False,
    options: rtc.RoomOptions | None = None,
) -> AsyncIterator[rtc.Room]:
    room = rtc.Room()
    try:
        await asyncio.wait_for(
            room.connect(
                url, room_token(name, identity, hidden=hidden), options or rtc.RoomOptions()
            ),
            15,
        )
        yield room
    finally:
        await asyncio.wait_for(room.disconnect(), 5)


def write_marker(path: Path, value: object) -> None:
    """Writes ``value`` as JSON at ``path`` in one rename, so ``read_marker`` never sees a
    partial file."""
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(value))
    partial.rename(path)


async def read_marker(path: Path) -> object:
    async with asyncio.timeout(15):
        while not path.exists():
            await asyncio.sleep(0.01)
    return json.loads(path.read_text())


async def tone(frequency: int, count: int | None = None) -> AsyncIterator[rtc.AudioFrame]:
    data = (4000 * np.sin(2 * np.pi * frequency * np.arange(960) / 48000)).astype(np.int16)
    index = 0
    while count is None or index < count:
        yield rtc.AudioFrame(
            data=data.tobytes(), sample_rate=48000, num_channels=1, samples_per_channel=960
        )
        index += 1
