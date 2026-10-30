"""Deterministic cold-close ownership faults on the installed public worker path."""

from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import select
import signal
import socket
import sys
import sysconfig
import tempfile
import threading
import time
from types import CodeType, FrameType, FunctionType
from typing import Literal, TypedDict, TypeGuard


sys.path.insert(0, os.environ["LIVEKIT_BOUNDARY_PACKAGE_ROOT"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _environment import worker_environment
from livekit import local_inference as native
from livekit.local_inference._pool import Pool, SpawnEvent, StateLostError
from livekit.local_inference._worker import receive


HERE = Path(__file__).resolve().parent


class Readiness(TypedDict):
    have_poll: str
    have_broken_poll: str
    select_poll: bool


def readiness() -> Readiness:
    have_poll: object = sysconfig.get_config_var("HAVE_POLL")
    have_broken_poll: object = sysconfig.get_config_var("HAVE_BROKEN_POLL")
    return {
        "have_poll": repr(have_poll),
        "have_broken_poll": repr(have_broken_poll),
        "select_poll": hasattr(select, "poll"),
    }


@dataclass(frozen=True, slots=True)
class Codes:
    start: CodeType
    finish: CodeType | None


class WitnessPool(Pool):
    def codes(self) -> Codes:
        lane = self._lanes["EOT"][0]
        methods = vars(type(lane))
        finish = methods.get("finish")
        return Codes(
            lane.start.__code__, finish.__code__ if isinstance(finish, FunctionType) else None
        )

    def rescue_transport(self) -> None:
        for lanes in self._lanes.values():
            for lane in lanes:
                method = vars(type(lane)).get("finish")
                if isinstance(method, FunctionType):
                    method(lane)


@dataclass(slots=True)
class Trace:
    pids: list[int] = field(default_factory=list[int])
    closing_faults: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, event: SpawnEvent) -> None:
        if event.stage == "spawned":
            self.pids.append(event.pid)
        if event.stage == "closing_transport":
            with self.lock:
                if self.closing_faults:
                    self.closing_faults -= 1
                    message = "Witnessed FD-close failure"
                    raise OSError(message)


@dataclass(frozen=True, slots=True)
class Outcome:
    mode: str
    retained_during_borrow: bool
    post_close_births: int
    close_failed: bool
    retained_before_retry: bool
    grouped_errors: bool
    state_lost_preserved: bool
    start_refused_failed_cleanup: bool
    final_fds_closed: bool
    reaped: bool
    joined: bool
    close_seconds: float


def group(error: BaseException) -> TypeGuard[BaseExceptionGroup[BaseException]]:
    return isinstance(error, BaseExceptionGroup)


def state_lost(error: BaseException) -> bool:
    if isinstance(error, StateLostError):
        return True
    if group(error):
        return any(state_lost(member) for member in error.exceptions)
    return False


def wait_file(path: Path) -> None:
    end = time.monotonic() + 3
    while not path.exists() or not path.read_bytes():
        if time.monotonic() >= end:
            message = "Actual first-party cold initializer was not witnessed"
            raise TimeoutError(message)
        time.sleep(0.002)


def exercise(
    mode: Literal["hello", "admit", "settle", "fd-retry"], directory: Path, *, v1: bool
) -> Outcome:
    native.start()
    marker = directory / (mode + "-init")
    os.environ["LIVEKIT_BOUNDARY_FAKE_INIT_READY"] = str(marker)
    trace = Trace(closing_faults=3 if mode == "fd-retry" else 0)
    owner = WitnessPool(
        command=(sys.executable, "-I", "-B", str(HERE / "_fake_worker.py")),
        observer=trace,
        worker_environment=worker_environment(),
    )
    codes = owner.codes()
    entered, release, quiescing = threading.Event(), threading.Event(), threading.Event()
    failures: list[BaseException] = []
    closes: list[BaseException] = []
    descriptors: list[int] = []
    countdown = 1 if mode == "settle" else 0
    fault_lock = threading.Lock()
    elapsed: list[float] = []
    retained = False
    retained_retry = False
    refused = False

    def profile(frame: FrameType, event: str, argument: object) -> None:
        nonlocal countdown
        if event != "call":
            return
        if (
            frame.f_code is threading.Condition.wait.__code__
            and frame.f_back is not None
            and frame.f_back.f_code is Pool.close.__code__
        ):
            quiescing.set()
        boundary = codes.start if mode == "admit" else receive.__code__
        if (
            frame.f_code is boundary
            and threading.current_thread() is caller
            and not entered.is_set()
        ):
            if mode != "admit":
                connection = frame.f_locals.get("connection")
                if not isinstance(connection, socket.socket):
                    message = "Readiness boundary did not own a socket"
                    raise TypeError(message)
                descriptors.append(connection.fileno())
            entered.set()
            if not release.wait(5):
                message = "Cold ownership checkpoint was not released"
                raise TimeoutError(message)
        fault = codes.finish if mode == "settle" else None
        if fault is not None and frame.f_code is fault:
            with fault_lock:
                if countdown:
                    countdown -= 1
                    if mode == "settle":
                        message = "Witnessed interruption before transport settlement"
                        raise InterruptedError(message)
                    message = "Witnessed FD-close failure"
                    raise OSError(message)

    def warm() -> None:
        sys.setprofile(profile)
        try:
            native.init_eot(_pool=owner)
        except BaseException as error:
            failures.append(error)
        finally:
            sys.setprofile(None)

    def close() -> None:
        sys.setprofile(profile)
        started = time.monotonic()
        try:
            native.shutdown()
        except BaseException as error:
            closes.append(error)
        finally:
            elapsed.append(time.monotonic() - started)
            sys.setprofile(None)

    caller = threading.Thread(target=warm)
    closer = threading.Thread(target=close)
    caller.start()
    try:
        if not entered.wait(3):
            message = "Admitted caller did not reach its real code boundary"
            raise TimeoutError(message)
        if mode != "admit":
            wait_file(marker)
        closer.start()
        if not quiescing.wait(2):
            message = "Close did not reach post-reap caller settlement"
            raise TimeoutError(message)
        if descriptors:
            try:
                os.fstat(descriptors[0])
                retained = True
            except OSError:
                pass
        release.set()
        if mode == "admit":
            # The previous implementation spawns after the scan. Reap that owned worker before its ten-second
            # startup timeout; the recorded post-close birth is the regression.
            deadline = time.monotonic() + 1
            while caller.is_alive() and not trace.pids and time.monotonic() < deadline:
                time.sleep(0.002)
            if trace.pids:
                wait_file(marker)
                for pid in owner.pids():
                    os.kill(pid, signal.SIGTERM)
        caller.join(3)
        closer.join(4)
        if caller.is_alive() or closer.is_alive():
            message = "Cold close failed its unchanged caller/retirement bound"
            raise TimeoutError(message)
        if mode == "fd-retry" or v1:
            retained_retry = bool(owner.pids())
            if descriptors:
                os.fstat(descriptors[0])
            if mode == "fd-retry":
                try:
                    native.start()
                except RuntimeError:
                    refused = True
    finally:
        release.set()
        if v1:
            owner.rescue_transport()
        native.shutdown()
        caller.join(2)
        if closer.ident is not None:
            closer.join(2)
        os.environ.pop("LIVEKIT_BOUNDARY_FAKE_INIT_READY", None)
    final_closed = True
    for descriptor in descriptors:
        try:
            os.fstat(descriptor)
        except OSError:
            continue
        final_closed = False
    if owner.pids():
        message = "Retained transport survived cleanup retry"
        raise AssertionError(message)
    for pid in trace.pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        message = "Owned worker survived cleanup retry"
        raise AssertionError(message)
    return Outcome(
        mode,
        retained,
        len(trace.pids) if mode == "admit" else 0,
        bool(closes),
        retained_retry,
        any(isinstance(error, BaseExceptionGroup) for error in failures),
        any(state_lost(error) for error in failures),
        refused,
        final_closed,
        True,
        not caller.is_alive() and not closer.is_alive(),
        elapsed[0],
    )


def completed_reply_interruption() -> dict[str, bool | int]:
    os.environ.pop("LIVEKIT_BOUNDARY_FAKE_INIT_READY", None)
    trace = Trace()
    owner = WitnessPool(
        command=(sys.executable, "-I", "-B", str(HERE / "_fake_worker.py")),
        observer=trace,
        worker_environment=worker_environment(),
    )
    first = owner.create_vad()
    codes = owner.codes()
    pcm = array("h", [8192] * 512).tobytes()
    interrupted = False
    calls = 0

    def interrupt(frame: FrameType, event: str, argument: object) -> None:
        if event == "call" and frame.f_code is codes.finish:
            raise KeyboardInterrupt

    def stale_guard(frame: FrameType, event: str, argument: object) -> None:
        nonlocal calls
        if event == "call" and frame.f_code is codes.finish:
            calls += 1
            message = "Stale context reached another session settlement"
            raise InterruptedError(message)

    try:
        sys.setprofile(interrupt)
        try:
            owner.predict("VAD", pcm, first)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            sys.setprofile(None)
        if not interrupted or owner.pids():
            message = "Completed but undelivered recurrent state was reusable"
            raise AssertionError(message)
        other = owner.create_vad()
        replacement = owner.create_vad()
        active = owner.pids()
        failure: StateLostError | None = None
        sys.setprofile(stale_guard)
        try:
            owner.predict("VAD", pcm, first)
        except StateLostError as error:
            failure = error
        finally:
            sys.setprofile(None)
        if failure is None or calls or owner.pids() != active:
            message = "Stale preflight touched healthy replacement custody"
            raise AssertionError(message)
        if (
            owner.predict("VAD", pcm, replacement) != 0.125
            or owner.predict("VAD", pcm, other) != 0.125
        ):
            message = "Settlement interruption changed another fresh context"
            raise AssertionError(message)
        owner.drop_vad(first)
        owner.drop_vad(other)
        owner.drop_vad(replacement)
    finally:
        sys.setprofile(None)
        owner.close()
    if owner.pids():
        message = "Completed-reply interruption leaked native ownership"
        raise AssertionError(message)
    return {
        "original_keyboard_interrupt": interrupted,
        "undelivered_state_quarantined": True,
        "stale_preflight_skipped_settlement": calls == 0,
        "replacement_and_other_context_unchanged": True,
        "reaped_workers": len(trace.pids),
    }


def validate(result: Outcome, *, v1: bool) -> None:
    mode = result.mode
    if v1:
        if result.close_failed or not result.retained_before_retry or result.grouped_errors:
            message = "V1 did not demonstrate silent retained custody after interruption"
            raise AssertionError(message)
    elif (mode != "admit" and not result.retained_during_borrow) or result.post_close_births:
        message = "Retirement closed borrowed IO or admitted a post-close worker"
        raise AssertionError(message)
    elif result.close_failed != (mode == "fd-retry") or (
        mode in ("settle", "fd-retry") and not result.grouped_errors
    ):
        message = "Cleanup failure was lost or its physical custody was not retryable"
        raise AssertionError(message)
    elif mode == "fd-retry" and (
        not result.retained_before_retry or not result.start_refused_failed_cleanup
    ):
        message = "Failed public shutdown released the old owner before retry"
        raise AssertionError(message)
    if not result.final_fds_closed or not result.reaped or not result.joined:
        message = "Qualification left owned resources after cleanup"
        raise AssertionError(message)


def main() -> None:
    if sys.version_info[:3] != (3, 15, 2):
        message = "Exact CPython3152 is required"
        raise RuntimeError(message)
    if len(sys.argv) == 1:
        modes: tuple[Literal["hello", "admit", "settle", "fd-retry"], ...] = (
            "hello",
            "admit",
            "settle",
            "fd-retry",
        )
        results: list[Outcome] = []
        for mode in modes:
            with tempfile.TemporaryDirectory(prefix="lki-cold-final-") as temporary:
                result = exercise(mode, Path(temporary), v1=False)
                validate(result, v1=False)
                results.append(result)
        print(
            json.dumps(
                {
                    "python": sys.version,
                    "readiness": readiness(),
                    "cold_retirement": [asdict(result) for result in results],
                    "completed": completed_reply_interruption(),
                }
            ),
            flush=True,
        )
        return
    mode = sys.argv[1]
    if mode == "completed":
        print(
            json.dumps(
                {
                    "python": sys.version,
                    "readiness": readiness(),
                    "completed": completed_reply_interruption(),
                }
            ),
            flush=True,
        )
        return
    if mode not in ("hello", "admit", "settle", "fd-retry"):
        message = "Unknown cold-close qualification mode"
        raise ValueError(message)
    v1 = len(sys.argv) > 2 and sys.argv[2] == "v1"
    with tempfile.TemporaryDirectory(prefix="lki-cold-final-") as temporary:
        result = exercise(mode, Path(temporary), v1=v1)
        validate(result, v1=v1)
        print(
            json.dumps({"python": sys.version, "readiness": readiness(), "result": asdict(result)}),
            flush=True,
        )


if __name__ == "__main__":
    main()
