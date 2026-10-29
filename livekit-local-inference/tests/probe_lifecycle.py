"""Adverse close and parent-loss controls at the actual worker boundary."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import TypeGuard

import numpy as np


sys.path.insert(
    0, os.environ.get("LIVEKIT_BOUNDARY_PACKAGE_ROOT", str(Path(__file__).resolve().parent))
)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _environment import worker_environment
from livekit.local_inference import EOT, VAD
from livekit.local_inference._pool import Pool, StateLostError


HERE = Path(__file__).resolve().parent


def pool(*, timeout: float = 2.0) -> Pool:
    return Pool(
        command=(sys.executable, "-I", "-B", str(HERE / "_fake_worker.py")),
        timeout=timeout,
        worker_environment=worker_environment(),
    )


def _wait_file(path: Path) -> None:
    deadline = time.monotonic() + 5
    while not path.exists() or not path.read_bytes():
        if time.monotonic() >= deadline:
            message = "fake did not witness its native boundary"
            raise AssertionError(message)
        time.sleep(0.005)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _reaped(pids: tuple[int, ...]) -> None:
    deadline = time.monotonic() + 3
    while any(_alive(pid) for pid in pids):
        if time.monotonic() >= deadline:
            message = "owned native worker survived its lifetime"
            raise AssertionError(message)
        time.sleep(0.005)


def cold_close(directory: Path) -> dict[str, float | bool]:
    ready = directory / "cold-ready"
    os.environ["LIVEKIT_BOUNDARY_FAKE_INIT_READY"] = str(ready)
    owner = pool()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(VAD, _pool=owner)
            _wait_file(ready)
            pids = owner.pids()
            assert int(ready.read_text()) in pids
            started = time.monotonic()
            owner.close()
            elapsed = time.monotonic() - started
            try:
                future.result(timeout=1)
            except StateLostError:
                pass
            else:
                message = "pre-HELLO native startup survived close"
                raise AssertionError(message)
        _reaped(pids)
    finally:
        owner.close()
        os.environ.pop("LIVEKIT_BOUNDARY_FAKE_INIT_READY", None)
    return {"pre_hello_entry_witnessed": True, "close_reaped": True, "close_s": elapsed}


def public_close(directory: Path, *, vad: bool) -> dict[str, float | bool | int]:
    ready = directory / ("vad-ready" if vad else "eot-ready")
    os.environ["LIVEKIT_BOUNDARY_FAKE_READY"] = str(ready)
    os.environ["LIVEKIT_BOUNDARY_FAKE_DELAY"] = "30"
    owner = pool(timeout=0.2)
    model = VAD(_pool=owner) if vad else EOT(_pool=owner)
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            active = executor.submit(model.predict, np.zeros(512, dtype=np.int16))
            _wait_file(ready)
            queued = executor.submit(model.predict, np.zeros(512, dtype=np.int16))
            pids = owner.pids()
            started = time.monotonic()
            model.close()
            elapsed = time.monotonic() - started
            for future in (active, queued):
                try:
                    future.result(timeout=1)
                except StateLostError, ValueError:
                    pass
                else:
                    message = "closed model admitted or replayed uncertain work"
                    raise AssertionError(message)
            try:
                model.predict(np.zeros(512, dtype=np.int16))
            except ValueError:
                pass
            else:
                message = "model admitted work after close"
                raise AssertionError(message)
        _reaped(pids)
        model.close()
    finally:
        model.close()
        owner.close()
        os.environ.pop("LIVEKIT_BOUNDARY_FAKE_READY", None)
        os.environ.pop("LIVEKIT_BOUNDARY_FAKE_DELAY", None)
    return {
        "public_model_close_reaped": True,
        "queued_calls_rejected": True,
        "close_s": elapsed,
        "retired_workers": len(pids),
    }


def reopen() -> dict[str, bool | int]:
    owner = pool()
    previous: list[int] = []
    try:
        for _ in range(2):
            with EOT(_pool=owner) as eot:
                assert eot.predict(np.full(512, 8192, dtype=np.int16)) == 0.25
                previous.extend(owner.pids())
            _reaped(tuple(previous))
            with VAD(_pool=owner) as vad:
                assert vad.predict(np.full(512, 8192, dtype=np.int16)) == 0.125
                previous.extend(owner.pids())
            _reaped(tuple(previous))
    finally:
        owner.close()
    return {
        "same_parent_reopen": True,
        "context_manager_cleanup": True,
        "retired_workers": len(previous),
    }


def _mapping(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _items(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def owner_dies(phase: str, directory: Path) -> dict[str, bool | str | int]:
    marker = directory / (phase + "-native-hang")
    receipt = directory / (phase + "-owner.json")
    environment = dict(os.environ)
    environment.update(
        {"LIVEKIT_BOUNDARY_NATIVE_HANG_LOG": str(marker), "LIVEKIT_NAPI_FAKE_MODE": phase}
    )
    child = subprocess.Popen(  # noqa: S603 - fixed current interpreter and first-party lifecycle probe.
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "owner", str(receipt)],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = child.communicate(timeout=8)
        assert child.returncode == 0, (stdout, stderr)
        raw: object = json.loads(receipt.read_text())
        assert _mapping(raw)
        values = raw.get("pids")
        observed = raw.get("native_marker")
        assert _items(values)
        assert isinstance(observed, str)
        pids = tuple(value for value in values if type(value) is int)
        assert len(pids) == len(values)
        assert pids
        assert all(pid > 0 for pid in pids)
        assert "NATIVE_HANG" in observed
        _reaped(pids)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=2)
    return {
        "phase": phase,
        "actual_owner_os_exit": True,
        "native_hang_witnessed": True,
        "orphan_workers_reaped": len(pids),
    }


def _owner(receipt: Path) -> None:
    marker = Path(os.environ["LIVEKIT_BOUNDARY_NATIVE_HANG_LOG"])
    phase = os.environ["LIVEKIT_NAPI_FAKE_MODE"]
    owner = pool()
    executor = ThreadPoolExecutor(max_workers=1)
    if phase == "dlopen-hang":
        executor.submit(VAD, _pool=owner)
    elif phase == "hang":
        model = EOT(_pool=owner)
        executor.submit(model.predict, np.zeros(512, dtype=np.int16))
    else:
        recurrent = VAD(_pool=owner)
        executor.submit(recurrent.close)
    _wait_file(marker)
    pids = owner.pids()
    assert pids
    receipt.write_text(json.dumps({"pids": pids, "native_marker": marker.read_text()}))
    os._exit(0)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "owner":
        _owner(Path(sys.argv[2]))
    with tempfile.TemporaryDirectory(prefix="lki-lifecycle-") as temporary:
        directory = Path(temporary)
        results = {
            "scope": "first-party working fakes and actual C++ parent watcher; no vendor model",
            "python": sys.version,
            "cold_close": cold_close(directory),
            "vad_close": public_close(directory, vad=True),
            "eot_close": public_close(directory, vad=False),
            "reopen": reopen(),
        }
        os.environ["LIVEKIT_BOUNDARY_FAKE_ADDON"] = str(
            Path(os.environ["LIVEKIT_BOUNDARY_NATIVE_DIR"]) / "source-owned-fake.node"
        )
        try:
            owner_death = [
                owner_dies(phase, directory) for phase in ("dlopen-hang", "hang", "finalizer-hang")
            ]
        finally:
            os.environ.pop("LIVEKIT_BOUNDARY_FAKE_ADDON", None)
        print(json.dumps({**results, "owner_death": owner_death}), flush=True)
