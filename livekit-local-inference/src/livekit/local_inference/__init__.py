"""The original LiveKit models, owned by bounded CPython 3.15.2 workers."""

from __future__ import annotations

from array import array
import atexit
from collections.abc import Buffer
import sys
import threading
import time
from typing import TYPE_CHECKING, Self
import weakref

from ._pool import Owner, Pool


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    import numpy as np
    from numpy.typing import NDArray


__all__ = [
    "EOT",
    "EOT_MAX_SAMPLES",
    "VAD",
    "VAD_WINDOW_SAMPLES",
    "init_eot",
    "init_vad",
    "shutdown",
]

EOT_MAX_SAMPLES = 19_200
VAD_WINDOW_SAMPLES = 512
_owner_lock = threading.Lock()
_owner: Pool | None = None
_shutdown = False


def _default_pool(*, factory: Callable[[], Pool] = Pool) -> Pool:
    global _owner
    with _owner_lock:
        if _shutdown:
            msg = "native inference owner has shut down"
            raise RuntimeError(msg)
        if _owner is None:
            _owner = factory()
        return _owner


def _pcm(pcm: object, *, vad: bool) -> bytes:
    # PEP3118 validates dtype/shape without importing NumPy in model workers.
    # Preserve logical values for strided/read-only/big-endian int16 arrays.
    if not isinstance(pcm, Buffer):
        message = "predict requires an int16 array buffer"
        raise TypeError(message)
    with memoryview(pcm) as view:
        if view.itemsize != 2 or view.format not in {"h", "<h", ">h", "=h", "@h"}:
            message = "predict requires an int16 array buffer"
            raise TypeError(message)
        if view.ndim != 1:
            message = f"array has incorrect number of dimensions: {view.ndim}; expected 1"
            raise ValueError(message)
        samples = view.nbytes // 2
        if vad and samples != VAD_WINDOW_SAMPLES:
            message = (
                f"vad.predict expects exactly 512 int16 samples (32 ms @ 16 kHz), got {samples}"
            )
            raise ValueError(message)
        if not vad and samples > EOT_MAX_SAMPLES:
            message = f"pcm too long: {samples} samples (max 19200, i.e. 1.2 s @ 16 kHz)"
            raise ValueError(message)
        payload = view.tobytes(order="C")
        if (view.format.startswith(">") and sys.byteorder == "little") or (
            view.format.startswith("<") and sys.byteorder == "big"
        ):
            normalized = array("h")
            normalized.frombytes(payload)
            normalized.byteswap()
            return normalized.tobytes()
        return payload


def init_eot(*, _pool: Pool | None = None) -> None:
    """Prewarm the EOT workers; their owner must eventually call shutdown()."""
    owner = _default_pool() if _pool is None else _default_pool(factory=lambda: _pool)
    owner.warm("EOT")


def init_vad() -> None:
    """Prewarm the VAD workers without creating recurrent stream contexts."""
    _default_pool().warm("VAD")


def shutdown() -> None:
    """Close admission and reap all workers, invalidating surviving contexts."""
    global _owner, _shutdown
    with _owner_lock:
        _shutdown = True
        if _owner is not None:
            _owner.close()
            _owner = None


atexit.register(shutdown)


def _close_model(
    pool: Pool,
    owner: Owner,
    condition: threading.Condition,
    idle: Callable[[], bool],
    release: Callable[[], None],
) -> None:
    pool.close_owner(owner)
    errors: list[BaseException] = []
    try:
        with condition:
            if not condition.wait_for(idle, timeout=pool.settle_timeout):
                errors.append(TimeoutError("native inference calls did not settle before close"))
    except BaseException as error:
        errors.append(error)
    if errors:
        try:
            pool.close_owner(owner, interrupt=True)
        except BaseException as error:
            errors.append(error)
    try:
        release()
    except BaseException as error:
        errors.append(error)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        message = "native inference close and cleanup failed"
        raise BaseExceptionGroup(message, errors)


class EOT:
    """Thread-safe stateless v1-mini predictions, with explicit close ownership."""

    def __init__(self, *, _pool: Pool | None = None) -> None:
        self._owner = _pool if _pool is not None else _default_pool()
        self._condition = threading.Condition()
        self._active = 0
        self._closed = False
        self._lease = self._owner.acquire_eot()
        self._finalizer = weakref.finalize(self, self._owner.release_eot, self._lease)

    def predict(self, pcm: NDArray[np.int16]) -> float:
        """Predict from at most 1.2 seconds of mono, 16 kHz int16 samples."""
        with self._condition:
            if self._closed:
                msg = "EOT instance is closed"
                raise ValueError(msg)
            self._active += 1
        try:
            return self._owner.predict("EOT", _pcm(pcm, vad=False), owner=self._lease)
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def close(self) -> None:
        """Reject queued calls, settle active work, and release this instance."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        _close_model(
            self._owner,
            self._lease,
            self._condition,
            lambda: self._active == 0,
            lambda: self._owner.release_eot(self._lease),
        )
        self._finalizer.detach()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class VAD:
    """One ordered recurrent context per stream, pinned to a worker generation."""

    def __init__(self, *, _pool: Pool | None = None) -> None:
        self._owner = _pool if _pool is not None else _default_pool()
        self._condition = threading.Condition()
        self._busy = False
        self._closing = False
        self._context = self._owner.create_vad()
        self._finalizer = weakref.finalize(self, self._owner.drop_vad, self._context)

    def _active(self) -> None:
        if self._closing or not self._finalizer.alive:
            msg = "VAD instance is closed"
            raise ValueError(msg)

    def _begin(self) -> None:
        deadline = time.monotonic() + self._owner.settle_timeout
        with self._condition:
            while True:
                self._active()
                if not self._busy:
                    self._busy = True
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    message = "VAD context admission deadline exceeded"
                    raise TimeoutError(message)
                self._condition.wait(remaining)

    def _settled(self) -> None:
        with self._condition:
            self._busy = False
            self._condition.notify_all()

    def predict(self, pcm: NDArray[np.int16]) -> float:
        """Advance this stream once from exactly 512 mono, 16 kHz samples."""
        self._begin()
        try:
            return self._owner.predict("VAD", _pcm(pcm, vad=True), self._context)
        finally:
            self._settled()

    def reset(self) -> None:
        """Reset this context; lost worker state requires a fresh VAD instance."""
        self._begin()
        try:
            self._owner.reset_vad(self._context)
        finally:
            self._settled()

    def close(self) -> None:
        """Reject queued calls, settle this context, and retire uncertain work."""
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        _close_model(
            self._owner,
            self._context.owner,
            self._condition,
            lambda: not self._busy,
            lambda: self._owner.drop_vad(self._context),
        )
        self._finalizer.detach()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
