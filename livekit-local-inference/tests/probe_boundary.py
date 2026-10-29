"""Exercise public facades across real thread and owned-process boundaries."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


sys.path.insert(
    0,
    os.environ.get("LIVEKIT_BOUNDARY_PACKAGE_ROOT", str(Path(__file__).resolve().parent)),
)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _environment import worker_environment
from livekit.local_inference import EOT, VAD
from livekit.local_inference._pool import Pool, StateLostError


HERE = Path(__file__).resolve().parent


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def pool(*, timeout: float = 2.0) -> Pool:
    return Pool(
        command=(sys.executable, "-I", "-B", str(HERE / "_fake_worker.py")),
        timeout=timeout,
        worker_environment=worker_environment(),
    )


async def normal() -> dict[str, int | float | bool | str]:
    owner = pool()
    eot = EOT(_pool=owner)
    first, second = VAD(_pool=owner), VAD(_pool=owner)
    positive = np.full(512, 8192, dtype=np.int16)
    negative = np.full(512, -16384, dtype=np.int16)
    pids: tuple[int, ...] = ()
    try:
        # Construction is on the loop; calls use distinct executor threads.
        with ThreadPoolExecutor(max_workers=4) as executor:
            loop = asyncio.get_running_loop()
            a, b = await asyncio.gather(
                loop.run_in_executor(executor, first.predict, positive),
                loop.run_in_executor(executor, second.predict, negative),
            )
            assert (a, b) == (0.125, 0.25)
            a, b = await asyncio.gather(
                loop.run_in_executor(executor, first.predict, positive),
                loop.run_in_executor(executor, second.predict, negative),
            )
            assert (a, b) == (0.1875, 0.375)
            first.reset()
            assert await loop.run_in_executor(executor, first.predict, positive) == 0.125
            assert await loop.run_in_executor(executor, second.predict, negative) == 0.4375
            started = time.perf_counter()
            observed = await asyncio.gather(
                asyncio.to_thread(eot.predict, positive),
                asyncio.to_thread(eot.predict, negative),
                loop.run_in_executor(executor, first.predict, positive),
                loop.run_in_executor(executor, second.predict, negative),
            )
            parallel_s = time.perf_counter() - started
            assert observed == [0.25, 0.5, 0.1875, 0.46875]
            assert eot.predict(positive[::-1]) == 0.25
            assert eot.predict(positive.astype(">i2")) == 0.25
            readonly = positive.copy()
            readonly.flags.writeable = False
            assert eot.predict(readonly) == 0.25
            pids = owner.pids()
            assert len(pids) == 4
            assert len(set(pids)) == 4
    finally:
        eot.close()
        first.close()
        second.close()
        owner.close()
    assert all(not alive(pid) for pid in pids)
    return {
        "cross_thread_public_facades": True,
        "recurrent_affinity_and_reset_isolation": True,
        "four_parallel_lanes": True,
        "parallel_wall_s": parallel_s,
        "retired_workers": len(pids),
        "parent_dummy_secret_absent_in_workers": True,
    }


def pending_close() -> dict[str, int | float | bool | str]:
    with tempfile.TemporaryDirectory(prefix="lki-boundary-") as temporary:
        ready = Path(temporary) / "ready"
        os.environ["LIVEKIT_BOUNDARY_FAKE_READY"] = str(ready)
        os.environ["LIVEKIT_BOUNDARY_FAKE_DELAY"] = "30"
        owner = pool(timeout=2.0)
        model = VAD(_pool=owner)
        pcm = np.zeros(512, dtype=np.int16)
        pids = owner.pids()
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(model.predict, pcm)
                deadline = time.monotonic() + 2
                while not ready.exists():
                    if time.monotonic() >= deadline:
                        message = "fake did not enter its owned native call"
                        raise AssertionError(message)
                    time.sleep(0.005)
                assert int(ready.read_text()) in pids
                started = time.perf_counter()
                owner.close()
                elapsed = time.perf_counter() - started
                try:
                    future.result(timeout=1)
                except StateLostError:
                    pass
                else:
                    message = "interrupted recurrent work did not invalidate state"
                    raise AssertionError(message)
            assert all(not alive(pid) for pid in pids)
            try:
                model.reset()
            except StateLostError, RuntimeError:
                pass
            else:
                message = "retired recurrent context was reused"
                raise AssertionError(message)
        finally:
            model.close()
            owner.close()
            os.environ.pop("LIVEKIT_BOUNDARY_FAKE_READY", None)
            os.environ.pop("LIVEKIT_BOUNDARY_FAKE_DELAY", None)
    return {
        "native_entry_witnessed": True,
        "pending_close_reaped": True,
        "state_not_replayed": True,
        "close_wall_s": elapsed,
        "retired_workers": len(pids),
    }


if __name__ == "__main__":
    if sys.version_info[:3] != (3, 15, 2):
        message = "every boundary probe process must run3.15.2"
        raise RuntimeError(message)
    os.environ["BILLY_BOUNDARY_DUMMY_SECRET"] = str(time.monotonic_ns())
    try:
        result = {
            "scope": "first-party working fake at public facade/worker seam; no vendor execution",
            "python": sys.version,
            "normal": asyncio.run(normal()),
            "pending_close": pending_close(),
        }
    finally:
        os.environ.pop("BILLY_BOUNDARY_DUMMY_SECRET", None)
    print(json.dumps(result), flush=True)
