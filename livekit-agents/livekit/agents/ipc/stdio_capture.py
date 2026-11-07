from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import struct
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger("livekit.agents.stdio")

_SO_TIMESTAMPNS: int = getattr(socket, "SO_TIMESTAMPNS", 35)
_SO_PASSCRED: int = getattr(socket, "SO_PASSCRED", 16)
_SCM_CREDENTIALS: int = getattr(socket, "SCM_CREDENTIALS", 2)
_MAX_LINE_BYTES = 64 * 1024
_CLOSE_TIMEOUT = 2.0

Stream = Literal["stdout", "stderr"]


def capture_enabled() -> bool:
    flag = os.environ.get("LIVEKIT_CAPTURE_JOB_STDIO")
    if flag is not None:
        return flag.strip().lower() in ("1", "true", "yes")
    return bool(os.environ.get("LIVEKIT_AGENT_ID"))


@dataclass
class ChildStdio:
    stdout: socket.socket
    stderr: socket.socket

    def close(self) -> None:
        for s in (self.stdout, self.stderr):
            with contextlib.suppress(OSError):
                s.close()


def create_stdio_pairs() -> tuple[ChildStdio, ChildStdio]:
    out_parent, out_child = socket.socketpair()
    err_parent, err_child = socket.socketpair()
    for s in (out_parent, err_parent):
        with contextlib.suppress(OSError):
            s.setsockopt(socket.SOL_SOCKET, _SO_PASSCRED, 1)
        with contextlib.suppress(OSError):
            s.setsockopt(socket.SOL_SOCKET, _SO_TIMESTAMPNS, 1)
    return ChildStdio(out_parent, err_parent), ChildStdio(out_child, err_child)


def redirect_stdio(stdio: ChildStdio) -> None:
    for stream, sock, fd in ((sys.stdout, stdio.stdout, 1), (sys.stderr, stdio.stderr, 2)):
        with contextlib.suppress(Exception):
            stream.flush()
        os.dup2(sock.fileno(), fd)
        sock.close()
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(Exception):
                reconfigure(line_buffering=True)


def _parse_cmsgs(anc: list[tuple[int, int, bytes]]) -> tuple[int | None, float | None]:
    pid: int | None = None
    ts: float | None = None
    for level, typ, payload in anc:
        if level != socket.SOL_SOCKET:
            continue
        if typ == _SCM_CREDENTIALS and len(payload) >= 12:
            cred_pid = struct.unpack("iii", payload[:12])[0]
            if cred_pid > 0:
                pid = cred_pid
        elif typ == _SO_TIMESTAMPNS and len(payload) >= 16:
            sec, nsec = struct.unpack("ll", payload[:16])
            if sec > 0:
                ts = sec + nsec / 1e9
    return pid, ts


class StdioReader:
    def __init__(
        self,
        sock: socket.socket,
        stream: Stream,
        extra_fnc: Callable[[], dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._sock = sock
        self._stream: Stream = stream
        self._extra_fnc = extra_fnc
        self._loop = loop
        self._buf = bytearray()
        self._buf_pid: int | None = None
        self._buf_ts: float | None = None
        self._closed_fut: asyncio.Future[None] = loop.create_future()
        self._cmsg_space = socket.CMSG_SPACE(16) + socket.CMSG_SPACE(12)

    def start(self) -> None:
        self._sock.setblocking(False)
        self._loop.add_reader(self._sock.fileno(), self._on_readable)

    async def aclose(self) -> None:
        try:
            await asyncio.wait_for(asyncio.shield(self._closed_fut), timeout=_CLOSE_TIMEOUT)
        except asyncio.TimeoutError:
            self._finish()

    def _on_readable(self) -> None:
        while True:
            try:
                data, anc, _, _ = self._sock.recvmsg(65536, self._cmsg_space)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                self._finish()
                return
            if not data:
                self._finish()
                return
            pid, ts = _parse_cmsgs(anc)
            self._feed(data, pid, ts if ts is not None else time.time())

    def _feed(self, data: bytes, pid: int | None, ts: float) -> None:
        start = 0
        while True:
            nl = data.find(b"\n", start)
            if nl == -1:
                break
            chunk = data[start:nl]
            if self._buf:
                self._buf.extend(chunk)
                self._flush_buf()
            else:
                self._emit(chunk, pid, ts)
            start = nl + 1

        rest = data[start:]
        if rest:
            if not self._buf:
                self._buf_pid, self._buf_ts = pid, ts
            self._buf.extend(rest)
            if len(self._buf) >= _MAX_LINE_BYTES:
                self._flush_buf()

    def _flush_buf(self) -> None:
        if not self._buf:
            return
        self._emit(bytes(self._buf), self._buf_pid, self._buf_ts or time.time())
        self._buf.clear()
        self._buf_pid = self._buf_ts = None

    def _emit(self, raw: bytes, pid: int | None, ts: float) -> None:
        text = raw.decode("utf-8", errors="replace").rstrip("\r")
        if not text.strip() or not logger.isEnabledFor(logging.INFO):
            return
        extra = dict(self._extra_fnc())
        extra["stream"] = self._stream
        if pid is not None:
            extra["pid"] = pid
        record = logger.makeRecord(logger.name, logging.INFO, "", 0, text, (), None, extra=extra)
        record.created = ts
        record.msecs = (ts - int(ts)) * 1000.0
        logger.handle(record)

    def _finish(self) -> None:
        if self._closed_fut.done():
            return
        with contextlib.suppress(Exception):
            self._loop.remove_reader(self._sock.fileno())
        self._flush_buf()
        with contextlib.suppress(OSError):
            self._sock.close()
        self._closed_fut.set_result(None)
