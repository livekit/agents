"""Deadline, response-custody and partial-spawn faults without module patching."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

import numpy as np


sys.path.insert(
    0, os.environ.get("LIVEKIT_BOUNDARY_PACKAGE_ROOT", str(Path(__file__).resolve().parent))
)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _environment import worker_environment
from livekit.local_inference import EOT, VAD, init_eot, shutdown
from livekit.local_inference._pool import Pool, SpawnEvent, StateLostError


HERE = Path(__file__).resolve().parent


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@dataclass(slots=True)
class Observer:
    fault: str = ""
    tripped: bool = False
    pids: list[int] = field(default_factory=list[int])
    descriptors: list[tuple[int, int, int]] = field(default_factory=list[tuple[int, int, int]])

    def __call__(self, event: SpawnEvent) -> None:
        if event.stage == "spawned":
            self.pids.append(event.pid)
            for descriptor in event.descriptors:
                state = os.fstat(descriptor)
                self.descriptors.append((descriptor, state.st_dev, state.st_ino))
        if event.stage == self.fault and not self.tripped:
            self.tripped = True
            message = "source-owned lifecycle observer failure"
            raise InterruptedError(message)


def _pool(observer: Observer, *, drip: bool = False, timeout: float = 0.12) -> Pool:
    worker = HERE / ("_drip_worker.py" if drip else "_fake_worker.py")
    return Pool(
        command=(sys.executable, "-I", "-B", str(worker)),
        timeout=timeout,
        observer=observer,
        worker_environment=worker_environment(),
    )


def _closed_descriptors(observer: Observer) -> None:
    for descriptor, device, inode in observer.descriptors:
        try:
            state = os.fstat(descriptor)
        except OSError:
            continue
        assert (state.st_dev, state.st_ino) != (device, inode)


def spawn_fault(stage: str) -> dict[str, str | bool | int]:
    observer = Observer(fault=stage)
    owner = _pool(observer)
    try:
        failure: BaseException | None = None
        try:
            EOT(_pool=owner)
        except (InterruptedError, StateLostError) as error:
            failure = error
        assert failure is not None
        assert isinstance(failure, InterruptedError) or isinstance(
            failure.__cause__, InterruptedError
        )
        assert observer.tripped
        assert not owner.pids()
        assert all(not _alive(pid) for pid in observer.pids)
        _closed_descriptors(observer)
    finally:
        owner.close()
    return {
        "stage": stage,
        "child_reaped": True,
        "all_local_descriptors_closed": True,
        "workers": len(observer.pids),
    }


def drip(mode: str) -> dict[str, str | bool | float]:
    os.environ["LIVEKIT_BOUNDARY_TRANSPORT"] = mode
    observer = Observer()
    owner = _pool(observer, drip=True)
    try:
        started = time.monotonic()
        try:
            owner.predict("EOT", b"\0\0" * 512)
        except StateLostError:
            pass
        else:
            message = "partial frame extended the absolute deadline"
            raise AssertionError(message)
        elapsed = time.monotonic() - started
        assert elapsed < 0.5
        assert all(not _alive(pid) for pid in observer.pids)
    finally:
        owner.close()
        os.environ.pop("LIVEKIT_BOUNDARY_TRANSPORT", None)
    return {"mode": mode, "one_absolute_deadline": True, "retired": True, "wall_s": elapsed}


def malformed(directory: Path) -> dict[str, bool | int]:
    os.environ["LIVEKIT_BOUNDARY_TRANSPORT"] = "malformed"
    os.environ["LIVEKIT_BOUNDARY_TRANSPORT_ONCE"] = str(directory / "first-malformed")
    observer = Observer()
    owner = _pool(observer, drip=True, timeout=1)
    model = EOT(_pool=owner)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(model.predict, np.zeros(512, dtype=np.int16))
            second = executor.submit(model.predict, np.zeros(512, dtype=np.int16))
            results: list[float] = []
            failures = 0
            for future in (first, second):
                try:
                    results.append(future.result(timeout=2))
                except StateLostError:
                    failures += 1
            assert failures == 1
            assert results == [0.25]
        for _ in range(10):
            assert model.predict(np.zeros(512, dtype=np.int16)) == 0.25
    finally:
        model.close()
        owner.close()
        os.environ.pop("LIVEKIT_BOUNDARY_TRANSPORT", None)
        os.environ.pop("LIVEKIT_BOUNDARY_TRANSPORT_ONCE", None)
    assert all(not _alive(pid) for pid in observer.pids)
    return {
        "one_bad_response_quarantined": True,
        "concurrent_good_response_preserved": True,
        "new_generation_survived": True,
        "workers": len(observer.pids),
    }


def failed_reap() -> dict[str, bool | int]:
    os.environ["LIVEKIT_BOUNDARY_TRANSPORT"] = "idle"
    observer = Observer(fault="before_reap")
    owner = _pool(observer, drip=True, timeout=1)
    owner.warm("EOT")
    try:
        failure: BaseExceptionGroup[BaseException] | None = None
        try:
            owner.close()
        except BaseExceptionGroup as error:
            failure = error
        assert failure is not None
        assert any(isinstance(item, InterruptedError) for item in failure.exceptions)
        assert observer.tripped
        assert len(owner.pids()) == 1
        owner.close()
        assert not owner.pids()
        assert all(not _alive(pid) for pid in observer.pids)
        _closed_descriptors(observer)
    finally:
        owner.close()
        os.environ.pop("LIVEKIT_BOUNDARY_TRANSPORT", None)
    return {
        "failed_reap_retains_identity": True,
        "other_lanes_retired": True,
        "retry_reaps_owned_child": True,
        "workers": len(observer.pids),
    }


def public_retry(directory: Path, *, vad: bool) -> dict[str, bool | int]:
    os.environ["LIVEKIT_BOUNDARY_TRANSPORT"] = "idle"
    os.environ["LIVEKIT_BOUNDARY_TRANSPORT_ONCE"] = str(
        directory / ("vad-once" if vad else "eot-once")
    )
    observer = Observer(fault="before_reap")
    owner = _pool(observer, drip=True, timeout=1)
    model = VAD(_pool=owner) if vad else EOT(_pool=owner)
    healthy = VAD(_pool=owner)
    pcm = np.zeros(512, dtype=np.int16)
    try:
        failure: BaseException | None = None
        try:
            model.close()
        except BaseExceptionGroup as error:
            failure = error
        assert failure is not None
        assert observer.tripped
        assert owner.pids()
        assert healthy.predict(pcm) == 0.25
        model.close()
        assert healthy.predict(pcm) == 0.25
        replacement = VAD(_pool=owner) if vad else EOT(_pool=owner)
        try:
            model.close()
            assert replacement.predict(pcm) == 0.25
        finally:
            replacement.close()
        healthy.close()
        assert not owner.pids()
        assert all(not _alive(pid) for pid in observer.pids)
        _closed_descriptors(observer)
    finally:
        model.close()
        healthy.close()
        owner.close()
        os.environ.pop("LIVEKIT_BOUNDARY_TRANSPORT", None)
        os.environ.pop("LIVEKIT_BOUNDARY_TRANSPORT_ONCE", None)
    return {
        "public_close_failed_observably": True,
        "retry_reaped": True,
        "replacement_and_other_owner_survived": True,
        "logical_release_once": True,
        "workers": len(observer.pids),
    }


def public_shutdown(directory: Path) -> dict[str, bool | int]:
    os.environ["LIVEKIT_BOUNDARY_TRANSPORT"] = "idle"
    os.environ["LIVEKIT_BOUNDARY_TRANSPORT_ONCE"] = str(directory / "shutdown-once")
    observer = Observer(fault="before_reap")
    owner = _pool(observer, drip=True, timeout=1)
    init_eot(_pool=owner)
    try:
        failure: BaseExceptionGroup[BaseException] | None = None
        try:
            shutdown()
        except BaseExceptionGroup as error:
            failure = error
        assert failure is not None
        assert observer.tripped
        assert owner.pids()
        try:
            EOT()
        except RuntimeError:
            pass
        else:
            message = "failed shutdown reopened module admission"
            raise AssertionError(message)
        shutdown()
        assert not owner.pids()
        shutdown()
        assert all(not _alive(pid) for pid in observer.pids)
        _closed_descriptors(observer)
    finally:
        shutdown()
        owner.close()
        os.environ.pop("LIVEKIT_BOUNDARY_TRANSPORT", None)
        os.environ.pop("LIVEKIT_BOUNDARY_TRANSPORT_ONCE", None)
    return {
        "shutdown_failure_observable": True,
        "retained_owner_retry_reaped": True,
        "admission_stayed_terminal": True,
        "workers": len(observer.pids),
    }


def wire_interrupt(directory: Path, *, failed_cleanup: bool) -> dict[str, bool | int]:
    marker = directory / ("interrupt-cleanup" if failed_cleanup else "interrupt")
    os.environ["LIVEKIT_BOUNDARY_TRANSPORT"] = "interrupt"
    os.environ["LIVEKIT_BOUNDARY_TRANSPORT_ONCE"] = str(marker)
    observer = Observer(fault="before_reap" if failed_cleanup else "")
    owner = _pool(observer, drip=True, timeout=1)
    model, healthy = VAD(_pool=owner), VAD(_pool=owner)
    pcm = np.zeros(512, dtype=np.int16)

    def interrupt() -> None:
        deadline = time.monotonic() + 2
        while not marker.exists() or not marker.read_bytes():
            if time.monotonic() >= deadline:
                message = "worker did not witness the sent prediction"
                raise AssertionError(message)
            time.sleep(0.005)
        assert int(marker.read_text()) in owner.pids()
        os.kill(os.getpid(), signal.SIGINT)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            sender = executor.submit(interrupt)
            failure: BaseException | None = None
            try:
                model.predict(pcm)
            except (KeyboardInterrupt, BaseExceptionGroup) as error:
                failure = error
            sender.result(timeout=2)
        assert failure is not None
        if failed_cleanup:
            assert isinstance(failure, BaseExceptionGroup)
            assert any(isinstance(error, KeyboardInterrupt) for error in failure.exceptions)
            assert any(isinstance(error, InterruptedError) for error in failure.exceptions)
            assert observer.tripped
            assert len(owner.pids()) == 2
        else:
            assert isinstance(failure, KeyboardInterrupt)
            assert len(owner.pids()) == 1
            try:
                model.reset()
            except StateLostError:
                pass
            else:
                message = "interrupted recurrent state was reused"
                raise AssertionError(message)
        if failed_cleanup:
            model.close()
        with VAD(_pool=owner) as replacement:
            if not failed_cleanup:
                try:
                    model.reset()
                except StateLostError:
                    pass
                else:
                    message = "stale context adopted a replacement generation"
                    raise AssertionError(message)
                model.close()
            assert replacement.predict(pcm) == 0.25
            assert healthy.predict(pcm) == 0.25
        healthy.close()
        assert not owner.pids()
        assert all(not _alive(pid) for pid in observer.pids)
    finally:
        model.close()
        healthy.close()
        owner.close()
        os.environ.pop("LIVEKIT_BOUNDARY_TRANSPORT", None)
        os.environ.pop("LIVEKIT_BOUNDARY_TRANSPORT_ONCE", None)
    return {
        "real_sigint_after_worker_entry": True,
        "original_interrupt_preserved": True,
        "cleanup_failure_custody": failed_cleanup,
        "uncertain_state_not_reused": True,
        "replacement_and_other_owner_survived": True,
        "retired_workers": len(observer.pids),
    }


def domain_errors() -> dict[str, bool | int]:
    os.environ["LIVEKIT_BOUNDARY_TRANSPORT"] = "domain"
    observer = Observer()
    owner = _pool(observer, drip=True, timeout=1)
    model = EOT(_pool=owner)
    try:
        pids = owner.pids()
        for expected in (ValueError, TypeError):
            failure: ValueError | TypeError | None = None
            try:
                model.predict(np.zeros(512, dtype=np.int16))
            except (ValueError, TypeError) as error:
                failure = error
            assert isinstance(failure, expected)
            assert owner.pids() == pids
        assert model.predict(np.zeros(512, dtype=np.int16)) == 0.25
        assert owner.pids() == pids
    finally:
        model.close()
        owner.close()
        os.environ.pop("LIVEKIT_BOUNDARY_TRANSPORT", None)
    assert all(not _alive(pid) for pid in observer.pids)
    return {
        "worker_value_and_type_errors_preserved": True,
        "same_generations_reused": True,
        "retired_workers": len(observer.pids),
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="lki-wire-") as temporary:
        result = {
            "scope": "first-party transport and process-ownership faults, no vendor model",
            "python": sys.version,
            "spawn_faults": [
                spawn_fault(stage) for stage in ("spawned", "published", "closing_fds")
            ],
            "drips": [drip(mode) for mode in ("hello", "header", "body")],
            "malformed": malformed(Path(temporary)),
            "failed_reap": failed_reap(),
            "public_vad_retry": public_retry(Path(temporary), vad=True),
            "public_eot_retry": public_retry(Path(temporary), vad=False),
            "wire_interrupt": wire_interrupt(Path(temporary), failed_cleanup=False),
            "wire_interrupt_cleanup": wire_interrupt(Path(temporary), failed_cleanup=True),
            "domain_errors": domain_errors(),
            "public_shutdown_retry": public_shutdown(Path(temporary)),
        }
        print(json.dumps(result), flush=True)
