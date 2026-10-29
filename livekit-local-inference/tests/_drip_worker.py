"""Controlled transport failures on the real inherited-socket seam."""

import os
from pathlib import Path
import signal
import socket
import sys
import time


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fake_worker import native
from livekit.local_inference._worker import (
    FLOAT,
    HELLO,
    REPLY,
    UINT,
    Kind,
    Operation,
    Request,
)


def run(connection: socket.socket, kind: Kind) -> None:
    mode = os.environ["LIVEKIT_BOUNDARY_TRANSPORT"]
    hello = HELLO.pack(b"LKI3152!", os.getpid())
    if mode == "hello":
        for value in hello:
            connection.sendall(bytes((value,)))
            time.sleep(0.04)
    else:
        connection.sendall(hello)
    predictions = 0
    while True:
        try:
            request = Request.read(connection, kind)
        except EOFError:
            return
        payload = (
            FLOAT.pack(0.25)
            if request.operation == Operation.PREDICT
            else UINT.pack(1)
            if request.operation == Operation.CREATE
            else b""
        )
        if request.operation == Operation.PREDICT:
            predictions += 1
            if mode == "domain" and predictions <= 2:
                error = b"first-party recoverable argument error"
                connection.sendall(REPLY.pack(predictions, len(error)) + error)
                continue
            if mode == "interrupt":
                marker = Path(os.environ["LIVEKIT_BOUNDARY_TRANSPORT_ONCE"])
                try:
                    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    pass
                else:
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    os.write(descriptor, str(os.getpid()).encode())
                    os.close(descriptor)
                    time.sleep(30)
            if mode == "header":
                for value in REPLY.pack(0, len(payload)):
                    connection.sendall(bytes((value,)))
                    time.sleep(0.04)
                connection.sendall(payload)
                continue
            if mode == "body":
                connection.sendall(REPLY.pack(0, len(payload)))
                time.sleep(0.08)
                connection.sendall(payload[:4])
                time.sleep(0.08)
                connection.sendall(payload[4:])
                continue
            if mode == "malformed":
                marker = Path(os.environ["LIVEKIT_BOUNDARY_TRANSPORT_ONCE"])
                try:
                    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    pass
                else:
                    os.close(descriptor)
                    payload = b"bad"
        connection.sendall(REPLY.pack(0, len(payload)) + payload)
        if mode == "idle" and (
            (kind == "EOT" and request.operation == Operation.WARM)
            or (kind == "VAD" and request.operation == Operation.DROP)
        ):
            once = os.environ.get("LIVEKIT_BOUNDARY_TRANSPORT_ONCE")
            if once:
                try:
                    descriptor = os.open(once, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    continue
                else:
                    os.close(descriptor)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            while True:
                time.sleep(1)


if __name__ == "__main__":
    descriptor, raw_kind, owner_descriptor = sys.argv[-3:]
    kind: Kind = "EOT" if raw_kind == "EOT" else "VAD"
    native.start_parent_watch(int(owner_descriptor))
    os.close(int(owner_descriptor))
    try:
        with socket.socket(fileno=int(descriptor)) as connection:
            run(connection, kind)
    except OSError, EOFError:
        pass
    finally:
        native.stop_parent_watch()
    os._exit(0)
