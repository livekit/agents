"""Prove fork detachment and foreign-lifecycle refusals without vendor execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ["LIVEKIT_BOUNDARY_PACKAGE_ROOT"])
from probe_transport import Observer

from _environment import worker_environment
from livekit import local_inference as native
from livekit.local_inference._pool import OwnedProcess, Pool, SpawnEvent


HERE = Path(__file__).resolve().parent


class HeldPool(Pool):
    def hold(self, entered: threading.Event, release: threading.Event) -> None:
        with self._condition:
            entered.set()
            assert release.wait(5)


class HeldEOT(native.EOT):
    def hold(self, entered: threading.Event, release: threading.Event) -> None:
        with self._condition, self._lease.release_lock:
            entered.set()
            assert release.wait(5)

    def finalize(self) -> None:
        self._finalizer()
        assert not self._finalizer.alive

    def release(self) -> None:
        self._owner.release_eot(self._lease)


class HeldVAD(native.VAD):
    def hold(self, entered: threading.Event, release: threading.Event) -> None:
        with self._condition, self._context.owner.release_lock:
            entered.set()
            assert release.wait(5)

    def finalize(self) -> None:
        self._finalizer()
        assert not self._finalizer.alive

    def drop(self) -> None:
        self._owner.drop_vad(self._context)

    def lane_calls(self) -> tuple[Callable[[], object], ...]:
        lane = self._context.lane
        return (lane.pid, lane.retire, lambda: lane.start(time.monotonic() + 1))


def pool(observer: Observer) -> HeldPool:
    return HeldPool(
        command=(sys.executable, "-I", "-B", str(HERE / "_fake_worker.py")),
        timeout=1,
        observer=observer,
        worker_environment=worker_environment(),
    )


def refused(call: Callable[[], object]) -> None:
    failure: RuntimeError | None = None
    try:
        call()
    except RuntimeError as error:
        failure = error
    assert failure is not None
    assert "fork" in str(failure)


def wait_child(pid: int) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        actual, status = os.waitpid(pid, os.WNOHANG)
        if actual:
            assert os.waitstatus_to_exitcode(status) == 0
            return
        time.sleep(0.005)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    message = "Fork child hung on an inherited lock or finalizer"
    raise AssertionError(message)


def live_descriptors(observer: Observer) -> tuple[int, ...]:
    result: set[int] = set()
    for descriptor, device, inode in observer.descriptors:
        try:
            state = os.fstat(descriptor)
        except OSError:
            continue
        if (state.st_dev, state.st_ino) == (device, inode):
            result.add(descriptor)
    return tuple(sorted(result))


def closed(descriptors: tuple[int, ...]) -> None:
    for descriptor in descriptors:
        try:
            os.fstat(descriptor)
        except OSError:
            continue
        message = "Fork retained an owned socket/lifeline FD copy"
        raise AssertionError(message)


def child_foreign(
    owner: HeldPool,
    eot: HeldEOT,
    vad: HeldVAD,
    descriptors: tuple[int, ...],
    unrelated: int,
    process: OwnedProcess,
) -> None:
    closed(descriptors)
    assert os.fstat(unrelated).st_mode
    pcm = np.zeros(512, dtype=np.int16)
    operations: tuple[Callable[[], object], ...] = (
        native.start,
        native.shutdown,
        native.init_eot,
        native.init_vad,
        lambda: eot.predict(pcm),
        eot.close,
        eot.__enter__,
        eot.release,
        lambda: vad.predict(pcm),
        vad.reset,
        vad.close,
        vad.__enter__,
        vad.drop,
        owner.pids,
        owner.close,
        owner.acquire_eot,
        owner.create_vad,
        *vad.lane_calls(),
    )
    for operation in operations:
        refused(operation)
    # Both weakref callbacks must skip foreign release locks and process handles.
    eot.finalize()
    vad.finalize()
    calls = 0

    def profile(frame: FrameType, event: str, argument: object) -> None:
        nonlocal calls
        if event == "call" and frame.f_code is subprocess.Popen[bytes].__del__.__code__:
            calls += 1

    sys.setprofile(profile)
    process.__del__()
    gc.collect()
    sys.setprofile(None)
    assert calls == 0


@dataclass(frozen=True, slots=True)
class ForkResult:
    held_locks_refused: bool
    child_fd_copies_closed: bool
    reused_unrelated_fd_preserved: bool
    foreign_weakref_and_process_finalizers_safe: bool
    parent_workers_unchanged: bool
    parent_finalizer_preserved: bool
    workers_retired: int
    fork_pid: int


def foreign() -> ForkResult:
    assert native.FORK_SAFE_PREWARM is False
    native.start()
    observer = Observer()
    owner = pool(observer)
    native.init_eot(_pool=owner)
    eot, vad = HeldEOT(), HeldVAD(_pool=owner)
    pcm = np.full(512, 8192, dtype=np.int16)
    assert eot.predict(pcm) == 0.25
    assert vad.predict(pcm) == 0.125
    original = owner.pids()
    descriptors = live_descriptors(observer)
    assert len(descriptors) == 4
    # Reuse a retired startup FD number; child cleanup must not close the tuple.
    unrelated = os.open(os.devnull, os.O_RDONLY)
    assert unrelated in {entry[0] for entry in observer.descriptors}
    reader, writer = os.pipe()
    process = OwnedProcess(
        [sys.executable, "-I", "-B", "-c", "import time;time.sleep(10)"],
        pass_fds=(reader, writer),
        env={"PATH": os.defpath},
    )
    os.close(reader)
    os.close(writer)
    entered = (threading.Event(), threading.Event(), threading.Event())
    release = threading.Event()
    holders = tuple(
        threading.Thread(target=target, args=(event, release), daemon=True)
        for target, event in zip((owner.hold, eot.hold, vad.hold), entered, strict=True)
    )
    for thread in holders:
        thread.start()
    assert all(event.wait(1) for event in entered)
    fork_pid = os.fork()
    if fork_pid == 0:
        try:
            child_foreign(owner, eot, vad, descriptors, unrelated, process)
        except BaseException:  # noqa: BLE001 -- Report child/finalizer failures before exit.
            import traceback

            traceback.print_exc()
            os._exit(1)
        # Run normal interpreter finalization with all foreign objects retained.
        raise SystemExit(0)
    try:
        wait_child(fork_pid)
    finally:
        release.set()
        for thread in holders:
            thread.join(1)
            assert not thread.is_alive()
        os.close(unrelated)
        process.terminate()
        process.wait(timeout=1)
    assert owner.pids() == original
    assert eot.predict(pcm) == 0.25
    assert vad.predict(pcm) == 0.1875
    calls = 0

    def profile(frame: FrameType, event: str, argument: object) -> None:
        nonlocal calls
        if event == "call" and frame.f_code is subprocess.Popen[bytes].__del__.__code__:
            calls += 1

    sys.setprofile(profile)
    process.__del__()
    sys.setprofile(None)
    assert calls == 1
    eot.close()
    vad.close()
    native.shutdown()
    assert not owner.pids()
    for pid in observer.pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        message = "Parent worker was not reaped"
        raise AssertionError(message)
    return ForkResult(True, True, True, True, True, True, len(observer.pids) + 1, fork_pid)


def publication(stage: str) -> dict[str, str | bool | int]:
    recorder = Observer()
    entered, release = threading.Event(), threading.Event()
    descriptors: list[tuple[int, int]] = []
    errors: list[BaseException] = []

    def observe(event: SpawnEvent) -> None:
        recorder(event)
        if event.stage == stage:
            descriptors.append((event.descriptors[0], event.descriptors[3]))
            entered.set()
            assert release.wait(2)
            message = "Publication fault while another thread forks"
            raise InterruptedError(message)

    owner = Pool(
        command=(sys.executable, "-I", "-B", str(HERE / "_fake_worker.py")),
        timeout=1,
        observer=observe,
        worker_environment=worker_environment(),
    )

    def warm() -> None:
        try:
            owner.warm("EOT")
        except BaseException as error:  # noqa: BLE001 -- Report child/finalizer failures before exit.
            errors.append(error)

    fork_started = threading.Event()
    os.register_at_fork(before=fork_started.set)

    def unblock() -> None:
        assert fork_started.wait(1)
        release.set()

    warming = threading.Thread(target=warm, daemon=True)
    warming.start()
    assert entered.wait(1)
    unblocking = threading.Thread(target=unblock, daemon=True)
    unblocking.start()
    pid = os.fork()
    if pid == 0:
        try:
            refused(owner.pids)
            closed(descriptors[0])
        except BaseException:  # noqa: BLE001 -- Report child/finalizer failures before exit.
            import traceback

            traceback.print_exc()
            os._exit(1)
        raise SystemExit(0)
    try:
        wait_child(pid)
    finally:
        release.set()
        warming.join(2)
        unblocking.join(1)
        owner.close()
    assert not warming.is_alive()
    assert len(errors) == 1
    assert release.is_set()
    assert not owner.pids()
    for child in recorder.pids:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            continue
        message = "Partial publication retained an owned process"
        raise AssertionError(message)
    return {
        "stage": stage,
        "fd_transfer_synchronized": True,
        "partial_owner_reaped": True,
        "fork_pid": pid,
    }


def empty(*, stopped: bool) -> dict[str, bool | int]:
    if stopped:
        native.shutdown()
    else:
        native.start()
    pid = os.fork()
    if pid == 0:
        try:
            observer = Observer()
            owner = pool(observer)
            if stopped:
                failure: RuntimeError | None = None
                try:
                    native.init_eot(_pool=owner)
                except RuntimeError as error:
                    failure = error
                assert failure is not None
                assert "shut down" in str(failure)
                assert not observer.pids
                native.start()
            native.init_eot(_pool=owner)
            with native.EOT() as model:
                assert model.predict(np.full(512, 8192, dtype=np.int16)) == 0.25
            native.shutdown()
            assert not owner.pids()
        except BaseException:  # noqa: BLE001 -- Report child/finalizer failures before exit.
            import traceback

            traceback.print_exc()
            os._exit(1)
        raise SystemExit(0)
    wait_child(pid)
    return {"preserved_shutdown": stopped, "fresh_child_owner": True, "child_pid": pid}


if __name__ == "__main__":
    assert sys.version_info[:3] == (3, 15, 2)
    print(
        json.dumps(
            {
                "scope": "FIRST-PARTY fork/finalizer proof only",
                "foreign": asdict(foreign()),
                "empty_open": empty(stopped=False),
                "empty_closed": empty(stopped=True),
                "publication": [
                    publication(stage) for stage in ("spawned", "published", "closing_fds")
                ],
            }
        ),
        flush=True,
    )
