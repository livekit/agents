"""Bounded inference frames and the child that owns one native environment."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import struct
import sys
import sysconfig
import time
import traceback
from typing import Literal, Protocol


type Kind = Literal["EOT", "VAD"]

# Preserve the qualified PCM bridge's framing; sockets are inherited capabilities.
HEADER = struct.Struct("!4sBB8sQI")
REPLY = struct.Struct("!BI")
UINT = struct.Struct("!Q")
FLOAT = struct.Struct("!d")
HELLO = struct.Struct("!8sQ")
MAX_PCM_BYTES = 38_400
MAX_REPLY_BYTES = 4096


class Operation(IntEnum):
    CREATE = 0
    PREDICT = 1
    RESET = 2
    DROP = 3
    WARM = 4
    STOP = 5


class Backend(Protocol):
    """The process-main-thread seam shared by native code and working fakes."""

    def warm(self) -> None: ...
    def create(self) -> int: ...
    def predict(self, handle: int, pcm: memoryview) -> float: ...
    def reset(self, handle: int) -> None: ...
    def drop(self, handle: int) -> None: ...
    def close(self) -> None: ...


def receive(connection: socket.socket, count: int, deadline: float | None = None) -> bytes:
    """Read exactly an already-validated extent, failing on a short frame."""
    result = bytearray(count)
    target = memoryview(result)
    cursor = 0
    while cursor < count:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                message = "native inference frame deadline exceeded"
                raise TimeoutError(message)
            connection.settimeout(remaining)
        size = connection.recv_into(target[cursor:])
        if not size:
            msg = "native inference connection closed"
            raise EOFError(msg)
        cursor += size
    return bytes(result)


def decode_uint(payload: bytes) -> int:
    if len(payload) != UINT.size:
        msg = "invalid native inference handle response"
        raise RuntimeError(msg)
    match UINT.unpack(payload):
        case (int() as value,):
            return value
        case _:
            msg = "invalid native inference handle response"
            raise RuntimeError(msg)


def decode_reply(connection: socket.socket, deadline: float) -> tuple[int, bytes]:
    match REPLY.unpack(receive(connection, REPLY.size, deadline)):
        case (int() as status, int() as count) if count <= MAX_REPLY_BYTES:
            return status, receive(connection, count, deadline)
        case _:
            msg = "invalid native inference response"
            raise RuntimeError(msg)


def decode_context(payload: bytes) -> int:
    token = decode_uint(payload)
    if token <= 0:
        message = "invalid native VAD context token"
        raise RuntimeError(message)
    return token


def decode_probability(payload: bytes) -> float:
    if len(payload) == FLOAT.size:
        match FLOAT.unpack(payload):
            case (float() as probability,) if math.isfinite(probability) and 0 <= probability <= 1:
                return probability
            case _:
                pass
    message = "invalid native inference probability response"
    raise RuntimeError(message)


def decode_ack(payload: bytes) -> None:
    if payload:
        message = "invalid native inference acknowledgement"
        raise RuntimeError(message)


def _artifact() -> Path:
    directory = Path(__file__).with_name("_vendor")
    value: object = json.loads((directory / "metadata.json").read_text())
    expected_platform = {
        ("darwin", "arm64"): "darwin-arm64",
        ("linux", "x86_64"): "linux-x64-gnu",
        ("linux", "aarch64"): "linux-arm64-gnu",
    }.get((sys.platform, platform.machine()))
    match value:
        case {
            "package": str(package),
            "version": "0.2.7",
            "sha256": str(digest),
            "source_commit": "c749f5a77488ca9bfb63b142c112c09a44349689",
            "license": str(license_name),
            "platform": str(target),
        } if (
            target == expected_platform
            and package == "@livekit/local-inference-" + target
            and license_name == "Apache-2.0"
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        ):
            path = directory / "engine.node"
            with path.open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != digest:
                msg = "native inference artifact identity changed"
                raise RuntimeError(msg)
            return path
        case _:
            msg = "invalid native inference artifact metadata"
            raise RuntimeError(msg)


class NativeBackend:
    """Keep the native capsule and recurrent contexts on this child thread."""

    def __init__(self, kind: Kind, owner_descriptor: int) -> None:
        flags = sys.getdlopenflags()
        try:
            sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
            from livekit.local_inference import _napi_host315
        finally:
            sys.setdlopenflags(flags)
        self._native = _napi_host315
        _napi_host315.start_parent_watch(owner_descriptor)
        os.close(owner_descriptor)
        try:
            self._host = _napi_host315.open(str(_artifact()))
        except BaseException:
            _napi_host315.stop_parent_watch()
            raise
        self._kind = kind
        self._ready = False

    def warm(self) -> None:
        if not self._ready:
            if self._kind == "EOT":
                self._native.init_eot(self._host)
            else:
                self._native.init_vad(self._host)
            self._ready = True

    def create(self) -> int:
        self.warm()
        return self._native.create_vad(self._host) if self._kind == "VAD" else 0

    def predict(self, handle: int, pcm: memoryview) -> float:
        self.warm()
        if self._kind == "EOT":
            return self._native.predict_eot(self._host, pcm)
        return self._native.predict_vad(self._host, handle, pcm)

    def reset(self, handle: int) -> None:
        self._native.reset_vad(self._host, handle)

    def drop(self, handle: int) -> None:
        if self._kind == "VAD":
            self._native.drop_vad(self._host, handle)

    def close(self) -> None:
        try:
            self._native.close(self._host)
        finally:
            self._native.stop_parent_watch()


@dataclass(frozen=True, slots=True)
class Request:
    operation: Operation
    handle: int
    payload: bytes

    @classmethod
    def read(cls, connection: socket.socket, kind: Kind) -> Request:
        header = receive(connection, HEADER.size)
        match HEADER.unpack(header):
            case (
                b"LKI1",
                int() as raw_operation,
                1,
                bytes() as dtype,
                int() as handle,
                int() as count,
            ) if dtype.rstrip(b"\0") == b"<i2" and count <= MAX_PCM_BYTES and count % 2 == 0:
                operation = Operation(raw_operation)
            case _:
                message = "invalid native inference frame"
                raise ValueError(message)
        samples = decode_uint(receive(connection, UINT.size))
        if samples * 2 != count:
            message = "invalid native inference sample extent"
            raise ValueError(message)
        if operation == Operation.PREDICT and kind == "VAD" and samples != 512:
            message = "VAD requires exactly 512 int16 samples"
            raise ValueError(message)
        if operation != Operation.PREDICT and samples:
            message = "native inference control frame cannot contain PCM"
            raise ValueError(message)
        if operation == Operation.RESET and kind != "VAD":
            message = "only VAD contexts can be reset"
            raise ValueError(message)
        return cls(operation, handle, receive(connection, count))


def execute(request: Request, backend: Backend) -> bytes:
    match request.operation:
        case Operation.PREDICT:
            values = array("h")
            values.frombytes(request.payload)
            probability = backend.predict(request.handle, memoryview(values))
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                message = "invalid native inference probability"
                raise RuntimeError(message)
            return FLOAT.pack(probability)
        case Operation.CREATE:
            return UINT.pack(backend.create())
        case Operation.RESET:
            backend.reset(request.handle)
        case Operation.DROP:
            backend.drop(request.handle)
        case Operation.WARM:
            backend.warm()
        case Operation.STOP:
            pass
    return b""


def serve(connection: socket.socket, kind: Kind, backend: Backend) -> None:
    """Run the checked protocol on the same thread that constructed the backend."""
    if sys.version_info[:3] != (3, 15, 2) or sysconfig.get_config_var("Py_GIL_DISABLED"):
        message = "native inference requires normal-GIL CPython 3.15.2"
        raise RuntimeError(message)
    try:
        connection.sendall(HELLO.pack(b"LKI3152!", os.getpid()))
        while True:
            try:
                request = Request.read(connection, kind)
            except EOFError:
                return
            try:
                response = execute(request, backend)
                connection.sendall(REPLY.pack(0, len(response)) + response)
                if request.operation == Operation.STOP:
                    return
            except (ValueError, TypeError, RuntimeError) as error:
                # Native failure may already have advanced recurrent state.
                # Retire this process; never replay a call or reconstruct LSTM.
                message = str(error).encode("utf-8")
                if len(message) > MAX_REPLY_BYTES:
                    message = b"native inference returned an oversized error"
                connection.sendall(REPLY.pack(4, len(message)) + message)
                return
    finally:
        backend.close()


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[2] not in {"EOT", "VAD"}:
        msg = "private native inference worker"
        raise SystemExit(msg)
    descriptor = int(sys.argv[1])
    owner_descriptor = int(sys.argv[3])
    kind: Kind = "EOT" if sys.argv[2] == "EOT" else "VAD"
    with socket.socket(fileno=descriptor) as connection:
        serve(connection, kind, NativeBackend(kind, owner_descriptor))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    status = 0
    try:
        main()
    except BaseException:  # noqa: BLE001 -- Child exit owns opaque native finalization.
        traceback.print_exc()
        status = 1
    # The process owns the opaque addon's singleton lifetime. Do not enter its
    # unqualified global destructors after stopping the parent-loss watchdog.
    os._exit(status)
