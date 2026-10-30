"""Owned process lanes with independent EOT and recurrent VAD admission."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
import io
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Literal
import weakref

from ._worker import (
    HEADER,
    HELLO,
    MAX_PCM_BYTES,
    UINT,
    Kind,
    Operation,
    decode_ack,
    decode_context,
    decode_probability,
    decode_reply,
    receive,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence


# Fork never copies an FD transfer halfway through publication or closure.
_fork_gate = threading.RLock()
_pools: weakref.WeakSet[Pool] = weakref.WeakSet()
_foreign_pools: list[Pool] = []


class OwnedProcess(subprocess.Popen[bytes]):
    """The normal Popen finalizer is valid only in the process that spawned it."""

    _creator_pid: int = 0

    def __init__(
        self,
        command: list[str],
        *,
        pass_fds: tuple[int, int],
        env: dict[str, str],
    ) -> None:
        self._creator_pid = os.getpid()
        # Fixed first-party argv and creator-owned POSIX descriptors.
        super().__init__(
            command,
            pass_fds=pass_fds,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=None,
            env=env,
        )

    def __del__(self, _getpid: Callable[[], int] = os.getpid) -> None:
        if _getpid() == self._creator_pid:
            super().__del__()


class StateLostError(RuntimeError):
    """A worker retired; recurrent state must be recreated rather than retried."""


@dataclass(eq=False, slots=True)
class Owner:
    """Closing an instance rejects queued work before joining active calls."""

    closed: bool = False
    released: bool = False
    release_lock: threading.Lock = field(default_factory=threading.Lock)


type Stage = Literal["spawned", "published", "closing_fds", "before_reap", "closing_transport"]


@dataclass(frozen=True, slots=True)
class SpawnEvent:
    stage: Stage
    pid: int
    descriptors: tuple[int, int, int, int]


type Observer = Callable[[SpawnEvent], None]


@dataclass(slots=True)
class _Session:
    process: subprocess.Popen[bytes]
    connection: socket.socket
    lifeline: io.FileIO
    descriptors: tuple[int, int, int, int]
    borrowed: bool = False
    retired: bool = False


@dataclass(slots=True)
class _Admission:
    lane: _Lane
    generation: int
    session: _Session | None = None


class _Lane:
    def __init__(
        self,
        kind: Kind,
        command: tuple[str, ...],
        environment: dict[str, str],
        observer: Observer | None = None,
    ) -> None:
        self.kind = kind
        self.command = command
        self.environment = environment
        self.busy = False
        self.active_owner: Owner | None = None
        self.retiring = False
        self.users = 0
        self.generation = 0
        self._lifetime = threading.RLock()
        self._session: _Session | None = None
        self._observer = observer
        self._owner = os.getpid()

    def ensure_owner(self) -> None:
        if os.getpid() != self._owner:
            message = "native inference lane cannot be inherited across fork"
            raise RuntimeError(message)

    def detach_after_fork(self) -> None:
        if os.getpid() == self._owner:
            message = "fork detachment requires a foreign lane"
            raise RuntimeError(message)
        # Close child-local copies, never shutdown the shared socket endpoint.
        # Historical descriptor numbers can already refer to unrelated files.
        session = self._session
        if session is not None:
            session.connection.close()
            session.lifeline.close()

    def _observe(
        self,
        stage: Stage,
        session: _Session,
    ) -> None:
        if self._observer is not None:
            self._observer(SpawnEvent(stage, session.process.pid, session.descriptors))

    def start(
        self,
        deadline: float,
        expected_generation: int | None = None,
        *,
        admission: _Admission | None = None,
    ) -> tuple[socket.socket, int]:
        self.ensure_owner()
        with self._lifetime:
            if admission is not None and admission.generation != self.generation:
                message = "native inference admission was retired"
                raise StateLostError(message)
            current = self._session
            if expected_generation is not None and (
                expected_generation != self.generation
                or current is None
                or current.process.poll() is not None
            ):
                message = "native VAD state was lost; create a new VAD instance"
                raise StateLostError(message)
            if current is not None and current.process.poll() is None:
                if admission is not None:
                    admission.session = current
                current.borrowed = True
                return current.connection, self.generation
            self.retire()
            try:
                with _fork_gate:
                    parent, child = socket.socketpair()
                    read_descriptor, write_descriptor = os.pipe()
                    reader = io.FileIO(read_descriptor, "rb", closefd=True)
                    writer = io.FileIO(write_descriptor, "wb", closefd=True)
                    process: subprocess.Popen[bytes] | None = None
                    session: _Session | None = None
                    try:
                        process = OwnedProcess(
                            [*self.command, str(child.fileno()), self.kind, str(reader.fileno())],
                            pass_fds=(child.fileno(), reader.fileno()),
                            env=self.environment,
                        )
                        session = _Session(
                            process,
                            parent,
                            writer,
                            (parent.fileno(), child.fileno(), reader.fileno(), writer.fileno()),
                        )
                        if admission is not None:
                            admission.session = session
                        session.borrowed = True
                        self._observe("spawned", session)
                        # One assignment transfers the complete process/socket/lifeline.
                        self._session = session
                        self.generation += 1
                        generation = self.generation
                        self._observe("published", session)
                        self._observe("closing_fds", session)
                        child.close()
                        reader.close()
                    except BaseException:
                        if process is not None:
                            # A failure before publication still owns the returned Popen.
                            self._session = (
                                session
                                if session is not None
                                else _Session(
                                    process,
                                    parent,
                                    writer,
                                    (
                                        parent.fileno(),
                                        child.fileno(),
                                        reader.fileno(),
                                        writer.fileno(),
                                    ),
                                )
                            )
                        else:
                            parent.close()
                            writer.close()
                        raise
                    finally:
                        child.close()
                        reader.close()
            except BaseException as error:
                self.retire_after_failure(error)
                raise
        # A cold vendor initialization must not exclude concurrent retirement.
        try:
            _hello(parent, process.pid, deadline)
        except BaseException as error:
            self.retire_after_failure(error)
            raise
        self.ensure_owner()
        with self._lifetime:
            if self._session is session and self.generation == generation:
                return parent, generation
        message = "native inference startup was retired"
        raise StateLostError(message)

    def retire_after_failure(self, error: BaseException) -> None:
        try:
            self.retire()
        except BaseException as cleanup:  # noqa: BLE001 -- Keep both operation and retirement failures.
            message = "native inference operation and retirement failed"
            raise BaseExceptionGroup(message, [error, cleanup]) from None

    def retire(self) -> None:
        """Retain complete custody until public wait confirms the child reaped."""
        self.ensure_owner()
        with self._lifetime:
            self.retiring = True
            session = self._session
            self.generation += 1
            if session is None:
                self.retiring = False
                return
            with suppress(OSError):
                session.connection.shutdown(socket.SHUT_RDWR)
            try:
                session.process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(session.process.pid, signal.SIGTERM)
                try:
                    session.process.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(session.process.pid, signal.SIGKILL)
                    self._observe("before_reap", session)
                    session.process.wait(timeout=2.0)
            # Reap the peer before closing a socket another thread may select.
            # Failed waits retain this session and keep admission quarantined.
            session.retired = True
            if not session.borrowed or not self.busy:
                self._close_session(session)

    def _close_session(self, session: _Session) -> None:
        self._observe("closing_transport", session)
        with _fork_gate:
            session.connection.close()
            session.lifeline.close()
            self._session = None
        self.retiring = False

    def finish(self, session: _Session) -> None:
        self.ensure_owner()
        with self._lifetime:
            if self._session is session and session.retired:
                self._close_session(session)

    def pid(self) -> int | None:
        self.ensure_owner()
        with self._lifetime:
            return self._session.process.pid if self._session is not None else None


def _hello(connection: socket.socket, pid: int, deadline: float) -> None:
    match HELLO.unpack(receive(connection, HELLO.size, deadline)):
        case (b"LKI3152!", int() as actual) if actual == pid:
            return
        case _:
            message = "native inference worker identity mismatch"
            raise RuntimeError(message)


def _domain_error(status: int, response: bytes) -> str:
    message = response.decode("utf-8", errors="strict")
    if status not in (1, 2):
        failure = "native inference state was lost: " + message
        raise RuntimeError(failure)
    return message


@dataclass(frozen=True, slots=True)
class Context:
    """A native VAD token cannot move between process generations."""

    lane: _Lane
    generation: int
    token: int
    owner: Owner = field(default_factory=Owner)


class Pool:
    """Own two bounded lanes per model kind; preserve VAD process affinity."""

    def __init__(
        self,
        *,
        command: Sequence[str] | None = None,
        timeout: float = 2.0,
        observer: Observer | None = None,
        worker_environment: Mapping[str, str] | None = None,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            msg = "native inference timeout must be finite and positive"
            raise ValueError(msg)
        if sys.version_info[:3] != (3, 15, 2):
            msg = "native inference requires CPython 3.15.2"
            raise RuntimeError(msg)
        selected = (
            tuple(command)
            if command is not None
            else (
                sys.executable,
                "-I",
                "-B",
                str(Path(__file__).with_name("_worker.py")),
            )
        )
        environment = {
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "TZ": "UTC",
            "TMPDIR": tempfile.gettempdir(),
            "EOT_PARALLELISM": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTHON_CPU_COUNT": "1",
        }
        if worker_environment:
            if command is None:
                message = "worker environment injection requires an explicit owned worker"
                raise ValueError(message)
            environment.update(worker_environment)
        kinds: tuple[Kind, ...] = ("EOT", "VAD")
        self._lanes = {
            kind: tuple(_Lane(kind, selected, environment, observer) for _ in range(2))
            for kind in kinds
        }
        self._condition = threading.Condition()
        self._timeout = timeout
        self._closed = False
        self._owner = os.getpid()
        self._eot_users = 0
        self._warm_holds: set[_Lane] = set()
        with _fork_gate:
            _pools.add(self)

    @property
    def settle_timeout(self) -> float:
        return self._timeout + 3.0

    def detach_after_fork(self) -> None:
        if self.owns_current_process():
            message = "fork detachment requires a foreign pool"
            raise RuntimeError(message)
        for lanes in self._lanes.values():
            for lane in lanes:
                lane.detach_after_fork()

    def owns_current_process(self) -> bool:
        return os.getpid() == self._owner

    def ensure_owner(self) -> None:
        if not self.owns_current_process():
            msg = "native inference pool cannot be inherited across fork"
            raise RuntimeError(msg)

    def _active(self) -> None:
        self.ensure_owner()
        if self._closed:
            msg = "native inference pool is closed"
            raise RuntimeError(msg)

    @contextmanager
    def _admit(
        self,
        kind: Kind,
        deadline: float,
        preferred: _Lane | None = None,
        owner: Owner | None = None,
    ) -> Generator[_Admission]:
        self.ensure_owner()
        with self._condition:
            while True:
                self._active()
                if owner is not None and owner.closed:
                    message = "native inference instance is closed"
                    raise ValueError(message)
                candidates = (preferred,) if preferred is not None else self._lanes[kind]
                available = next(
                    (lane for lane in candidates if not lane.busy and not lane.retiring), None
                )
                if available is not None:
                    available.busy = True
                    available.active_owner = owner
                    admission = _Admission(available, available.generation)
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    msg = "native inference admission deadline exceeded"
                    raise TimeoutError(msg)
                self._condition.wait(remaining)

        lane = admission.lane
        operation_error: BaseException | None = None
        try:
            yield admission
        except BaseException as error:
            operation_error = error
            raise
        finally:
            try:
                self.ensure_owner()
                session = admission.session
                if session is not None:
                    # No socket access remains after this owned lease release.
                    session.borrowed = False
                    lane.finish(session)
            except BaseException as cleanup:  # noqa: BLE001 -- Preserve both failures.
                settlement_error: BaseException = cleanup
                if operation_error is not None:
                    message = "native inference operation and settlement failed"
                    settlement_error = BaseExceptionGroup(message, [operation_error, cleanup])
                # A completed reply can still be lost to settlement interruption;
                # quarantine its recurrent state before another caller can enter.
                lane.retiring = True
                lane.retire_after_failure(settlement_error)
                raise settlement_error from None
            finally:
                self.ensure_owner()
                with self._condition:
                    lane.busy = False
                    lane.active_owner = None
                    self._condition.notify_all()

    def _exchange[T](
        self,
        kind: Kind,
        operation: Operation,
        *,
        decode: Callable[[bytes], T],
        context: Context | None = None,
        pcm: bytes = b"",
        preferred: _Lane | None = None,
        timeout: float | None = None,
        owner: Owner | None = None,
    ) -> tuple[T, _Lane, int]:
        if len(pcm) > MAX_PCM_BYTES or len(pcm) % 2:
            msg = "native inference PCM extent exceeds its bound"
            raise ValueError(msg)
        deadline = time.monotonic() + (self._timeout if timeout is None else timeout)
        request_owner = (
            context.owner if context is not None and operation != Operation.DROP else owner
        )
        with self._admit(
            kind, deadline, context.lane if context is not None else preferred, request_owner
        ) as admission:
            lane = admission.lane
            if context is not None and context.generation != lane.generation:
                msg = "native VAD state was lost; create a new VAD instance"
                raise StateLostError(msg)
            try:
                connection, generation = lane.start(
                    deadline,
                    context.generation if context is not None else None,
                    admission=admission,
                )
            except (OSError, EOFError, UnicodeError, RuntimeError) as error:
                msg = "native inference startup failed; create a fresh context"
                raise StateLostError(msg) from error
            if context is not None and context.generation != generation:
                msg = "native VAD state was lost; create a new VAD instance"
                raise StateLostError(msg)
            try:
                connection.settimeout(max(0.000001, deadline - time.monotonic()))
                handle = context.token if context is not None else 0
                connection.sendall(
                    HEADER.pack(b"LKI1", operation, 1, b"<i2", handle, len(pcm))
                    + UINT.pack(len(pcm) // 2)
                    + pcm
                )
                status, response = decode_reply(connection, deadline)
                if status == 0:
                    return decode(response), lane, generation
                message = _domain_error(status, response)
            except BaseException as error:
                lane.retire_after_failure(error)
                if isinstance(error, (OSError, EOFError, UnicodeError, RuntimeError)):
                    msg = "native inference state was lost; create a fresh context"
                    raise StateLostError(msg) from error
                raise
            # These validated worker replies are recoverable domain errors.
            if status == 1:
                raise ValueError(message)
            raise TypeError(message)

    def warm(self, kind: Kind, *, retain: bool = True) -> None:
        self.ensure_owner()
        lanes = self._lanes[kind][:1] if kind == "EOT" else self._lanes[kind]
        if retain:
            self.ensure_owner()
            with self._condition:
                self._active()
                # Keep explicit prewarm intent through retry; ordinary model close
                # cannot retire a lane while its public prewarm is in flight.
                self._warm_holds.update(lanes)
        for lane in lanes:
            self._exchange(kind, Operation.WARM, decode=decode_ack, preferred=lane, timeout=10.0)

    def acquire_eot(self) -> Owner:
        self.ensure_owner()
        owner = Owner()
        self.ensure_owner()
        with self._condition:
            self._active()
            self._eot_users += 1
        try:
            self.warm("EOT", retain=False)
        except BaseException:
            self.release_eot(owner)
            raise
        return owner

    def release_eot(self, owner: Owner) -> None:
        self.ensure_owner()
        with owner.release_lock:
            self.close_owner(owner)
            self.ensure_owner()
            with self._condition:
                if not owner.released:
                    self._eot_users -= 1
                    owner.released = True
                lanes = tuple(
                    lane
                    for lane in self._lanes["EOT"]
                    if lane.retiring or (self._eot_users == 0 and lane not in self._warm_holds)
                )
                for lane in lanes:
                    lane.retiring = True
            self._retire_lanes(lanes)

    def close_owner(self, owner: Owner, *, interrupt: bool = False) -> None:
        self.ensure_owner()
        with self._condition:
            owner.closed = True
            lanes = (
                tuple(
                    lane
                    for values in self._lanes.values()
                    for lane in values
                    if lane.busy and lane.active_owner is owner
                )
                if interrupt
                else ()
            )
            for lane in lanes:
                lane.retiring = True
            self._condition.notify_all()
        if lanes:
            self._retire_lanes(lanes)

    def _retire_lanes(self, lanes: tuple[_Lane, ...]) -> None:
        self.ensure_owner()
        errors: list[BaseException] = []
        for lane in lanes:
            try:
                lane.retire()
            except BaseException as error:  # noqa: BLE001 -- Retire every lane even after interruption.
                errors.append(error)
        self.ensure_owner()
        with self._condition:
            for lane in lanes:
                if lane.pid() is None:
                    lane.retiring = False
            self._condition.notify_all()
        if errors:
            message = "native inference worker retirement failed"
            raise BaseExceptionGroup(message, errors)

    def create_vad(self) -> Context:
        self.ensure_owner()
        with self._condition:
            self._active()
            lane = min(self._lanes["VAD"], key=lambda item: item.users)
            if lane.users >= 128:
                msg = "native VAD context capacity exhausted"
                raise RuntimeError(msg)
            lane.users += 1
        try:
            token, _, generation = self._exchange(
                "VAD",
                Operation.CREATE,
                decode=decode_context,
                preferred=lane,
                timeout=10.0,
            )
            return Context(lane, generation, token)
        except BaseException:
            self.ensure_owner()
            with self._condition:
                lane.users -= 1
            raise

    def drop_vad(self, context: Context) -> None:
        self.ensure_owner()
        with context.owner.release_lock:
            self.close_owner(context.owner)
            try:
                if (
                    not context.owner.released
                    and context.generation == context.lane.generation
                    and not self._closed
                ):
                    self._exchange("VAD", Operation.DROP, decode=decode_ack, context=context)
            finally:
                self.ensure_owner()
                with self._condition:
                    if not context.owner.released:
                        context.lane.users -= 1
                        context.owner.released = True
                    retire = context.lane.retiring or (
                        context.lane.users == 0 and context.lane not in self._warm_holds
                    )
                    if retire:
                        context.lane.retiring = True
                if retire:
                    self._retire_lanes((context.lane,))

    def reset_vad(self, context: Context) -> None:
        self._exchange("VAD", Operation.RESET, decode=decode_ack, context=context)

    def predict(
        self,
        kind: Kind,
        pcm: bytes,
        context: Context | None = None,
        *,
        owner: Owner | None = None,
    ) -> float:
        probability, _, _ = self._exchange(
            kind,
            Operation.PREDICT,
            decode=decode_probability,
            context=context,
            pcm=pcm,
            owner=owner,
        )
        return probability

    def pids(self) -> tuple[int, ...]:
        """Return this owner's process identities for lifecycle qualification."""
        self.ensure_owner()
        return tuple(
            pid
            for lanes in self._lanes.values()
            for lane in lanes
            if (pid := lane.pid()) is not None
        )

    def close(self) -> None:
        self.ensure_owner()
        with self._condition:
            self._closed = True
            lanes = tuple(lane for values in self._lanes.values() for lane in values)
            for lane in lanes:
                lane.retiring = True
            self._condition.notify_all()
        self._retire_lanes(lanes)
        deadline = time.monotonic() + 3.0
        self.ensure_owner()
        with self._condition:
            while any(lane.busy for lanes in self._lanes.values() for lane in lanes):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    msg = "native inference callers did not settle after worker retirement"
                    raise RuntimeError(msg)
                self._condition.wait(remaining)
        # The borrower may have failed while releasing its FD lease. Retry only
        # after every admitted caller has relinquished socket access.
        self._retire_lanes(lanes)


def _before_fork() -> None:
    _fork_gate.acquire()


def _after_fork_parent() -> None:
    _fork_gate.release()


def _after_fork_child() -> None:
    global _fork_gate, _pools
    # No inherited Python lock, Popen wait/poll, or foreign signal is safe here.
    try:
        for pool in tuple(_pools):
            pool.detach_after_fork()
            _foreign_pools.append(pool)
    except BaseException:  # noqa: BLE001 -- A failed fork detach must terminate this child.
        # A failed detach cannot leave a running child with foreign lifelines.
        os._exit(75)
    _pools = weakref.WeakSet()
    _fork_gate = threading.RLock()


os.register_at_fork(
    before=_before_fork,
    after_in_parent=_after_fork_parent,
    after_in_child=_after_fork_child,
)
