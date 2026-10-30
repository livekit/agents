"""Prove explicit public prewarm, retained ownership and restart at real IPC seams."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(
    0,
    os.environ.get("LIVEKIT_BOUNDARY_PACKAGE_ROOT", str(Path(__file__).resolve().parent)),
)
from probe_transport import Observer

from _environment import worker_environment
from livekit import local_inference as native
from livekit.local_inference._pool import Pool


TESTS = Path(__file__).resolve().parent


def pool(observer: Observer, *, drip: bool = False) -> Pool:
    return Pool(
        command=(
            sys.executable,
            "-I",
            "-B",
            str(TESTS / ("_drip_worker.py" if drip else "_fake_worker.py")),
        ),
        timeout=1,
        observer=observer,
        worker_environment=worker_environment(),
    )


def retired(observer: Observer) -> None:
    for pid in observer.pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        message = "Owned worker survived public lifecycle retirement"
        raise AssertionError(message)


def refused(operation: str) -> None:
    failure: RuntimeError | None = None
    try:
        if operation == "start":
            native.start()
        else:
            native.init_eot()
    except RuntimeError as error:
        failure = error
    assert failure is not None


def normal() -> dict[str, bool | int]:
    native.start()
    native.start()
    first = Observer()
    owner = pool(first)
    native.init_eot(_pool=owner)
    primary = owner.pids()
    assert len(primary) == 1
    model = native.EOT()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            left = executor.submit(model.predict, np.full(512, 8192, dtype=np.int16))
            right = executor.submit(model.predict, np.full(512, -16384, dtype=np.int16))
            assert left.result(timeout=2) == 0.25
            assert right.result(timeout=2) == 0.5
        assert len(owner.pids()) == 2
        model.close()
        assert owner.pids() == primary
        old = native.EOT()
        native.shutdown()
        assert not owner.pids()
        refused("warm")
        native.start()
        second = Observer()
        replacement = pool(second)
        native.init_eot(_pool=replacement)
        assert len(replacement.pids()) == 1
        failure: RuntimeError | None = None
        try:
            old.predict(np.zeros(512, dtype=np.int16))
        except RuntimeError as error:
            failure = error
        assert failure is not None
        old.close()
        with native.EOT() as current:
            assert current.predict(np.full(512, 8192, dtype=np.int16)) == 0.25
        assert len(replacement.pids()) == 1
        native.shutdown()
        assert not replacement.pids()
        retired(first)
        retired(second)
    finally:
        model.close()
        native.shutdown()
        owner.close()
    return {
        "public_prewarm_one": True,
        "parallel_capacity_two": True,
        "secondary_retired_at_last_user": True,
        "old_handle_stayed_closed": True,
        "explicit_restart_only": True,
        "retired_workers": len(first.pids) + len(second.pids),
    }


def failed_reap(directory: Path) -> dict[str, bool | int]:
    os.environ["LIVEKIT_BOUNDARY_TRANSPORT"] = "idle"
    os.environ["LIVEKIT_BOUNDARY_TRANSPORT_ONCE"] = str(directory / "failed-once")
    native.start()
    observer = Observer(fault="before_reap")
    owner = pool(observer, drip=True)
    try:
        native.init_eot(_pool=owner)
        failure: BaseExceptionGroup[BaseException] | None = None
        try:
            native.shutdown()
        except BaseExceptionGroup as error:
            failure = error
        assert failure is not None
        assert observer.tripped
        assert len(owner.pids()) == 1
        before = tuple(observer.pids)
        refused("start")
        refused("warm")
        assert tuple(observer.pids) == before
        assert len(owner.pids()) == 1
        native.shutdown()
        assert not owner.pids()
        native.start()
        native.shutdown()
        retired(observer)
    finally:
        native.shutdown()
        owner.close()
        os.environ.pop("LIVEKIT_BOUNDARY_TRANSPORT", None)
        os.environ.pop("LIVEKIT_BOUNDARY_TRANSPORT_ONCE", None)
    return {
        "failed_retirement_refuses_start": True,
        "late_warm_refused": True,
        "retry_reaps_before_reopen": True,
        "retired_workers": len(observer.pids),
    }


if __name__ == "__main__":
    if sys.version_info[:3] != (3, 15, 2):
        message = "Every lifecycle probe process must run3152"
        raise RuntimeError(message)
    with tempfile.TemporaryDirectory(prefix="lki-restart-") as temporary:
        print(
            json.dumps(
                {
                    "scope": "FIRST-PARTY working fake only; no vendor inference",
                    "normal": normal(),
                    "failed_reap": failed_reap(Path(temporary)),
                }
            ),
            flush=True,
        )
