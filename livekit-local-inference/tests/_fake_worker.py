"""Working recurrent backend at the same owned-worker protocol seam."""

from array import array
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback


sys.path.insert(
    0, os.environ.get("LIVEKIT_BOUNDARY_PACKAGE_ROOT", str(Path(__file__).resolve().parent))
)
from livekit.local_inference._worker import Kind, serve


flags = sys.getdlopenflags()
try:
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
    from livekit.local_inference import _napi_host315 as native
finally:
    sys.setdlopenflags(flags)


class FakeBackend:
    def __init__(self, kind: Kind) -> None:
        if "LIVEKIT_BOUNDARY_DUMMY_SECRET" in os.environ:
            message = "parent application credential crossed the owned worker boundary"
            raise RuntimeError(message)
        self._owner = threading.get_ident()
        self._kind = kind
        self._sequence = 0
        self._contexts: dict[int, float] = {}
        before_hello = os.environ.get("LIVEKIT_BOUNDARY_FAKE_INIT_READY")
        if before_hello:
            Path(before_hello).write_text(str(os.getpid()))
            time.sleep(30)

    def _owned(self) -> None:
        if threading.get_ident() != self._owner:
            message = "fake native backend crossed its owner thread"
            raise RuntimeError(message)

    def warm(self) -> None:
        self._owned()

    def create(self) -> int:
        self._owned()
        if self._kind == "EOT":
            return 0
        if len(self._contexts) >= 128:
            message = "fake native context capacity exhausted"
            raise RuntimeError(message)
        self._sequence += 1
        self._contexts[self._sequence] = 0.0
        return self._sequence

    def predict(self, handle: int, pcm: memoryview) -> float:
        self._owned()
        ready = os.environ.get("LIVEKIT_BOUNDARY_FAKE_READY")
        if ready:
            Path(ready).write_text(str(os.getpid()))
        delay = float(os.environ.get("LIVEKIT_BOUNDARY_FAKE_DELAY", "0.01"))
        time.sleep(delay)
        values = array("h")
        values.frombytes(pcm.tobytes())
        mean = sum(abs(value) for value in values) / (32768 * len(values)) if values else 0.0
        if self._kind == "EOT":
            return mean
        previous = self._contexts[handle]
        current = previous * 0.5 + mean * 0.5
        self._contexts[handle] = current
        return current

    def reset(self, handle: int) -> None:
        self._owned()
        self._contexts[handle] = 0.0

    def drop(self, handle: int) -> None:
        self._owned()
        if self._kind == "VAD":
            del self._contexts[handle]

    def close(self) -> None:
        self._owned()
        self._contexts.clear()


class NativeFakeBackend:
    """The actual C++ host and first-party fake addon, never the vendor engine."""

    def __init__(self, kind: Kind) -> None:
        self._kind = kind
        self._host = native.open(os.environ["LIVEKIT_BOUNDARY_FAKE_ADDON"])

    def warm(self) -> None:
        if self._kind == "EOT":
            native.init_eot(self._host)
        else:
            native.init_vad(self._host)

    def create(self) -> int:
        self.warm()
        return native.create_vad(self._host) if self._kind == "VAD" else 0

    def predict(self, handle: int, pcm: memoryview) -> float:
        return (
            native.predict_eot(self._host, pcm)
            if self._kind == "EOT"
            else native.predict_vad(self._host, handle, pcm)
        )

    def reset(self, handle: int) -> None:
        native.reset_vad(self._host, handle)

    def drop(self, handle: int) -> None:
        if self._kind == "VAD":
            native.drop_vad(self._host, handle)

    def close(self) -> None:
        native.close(self._host)


if __name__ == "__main__":
    status = 0
    try:
        descriptor, raw_kind, owner_descriptor = sys.argv[-3:]
        kind: Kind = "EOT" if raw_kind == "EOT" else "VAD"
        native.start_parent_watch(int(owner_descriptor))
        os.close(int(owner_descriptor))
        marker = os.environ.get("LIVEKIT_BOUNDARY_NATIVE_HANG_LOG")
        if marker:
            with Path(marker).open("wb", buffering=0) as output:
                os.dup2(output.fileno(), 1)
        with socket.socket(fileno=int(descriptor)) as connection:
            backend = (
                NativeFakeBackend(kind)
                if os.environ.get("LIVEKIT_BOUNDARY_FAKE_ADDON")
                else FakeBackend(kind)
            )
            serve(connection, kind, backend)
    except BaseException:
        traceback.print_exc()
        status = 1
    finally:
        native.stop_parent_watch()
    os._exit(status)
