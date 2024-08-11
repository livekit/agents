from __future__ import annotations
import asyncio
import socket


class DuplexConn:

    def __init__(self):
        s1, s2 = socket.socketpair()
        s1.setblocking(False)
        s2.setblocking(False)



"""
class Connection:
    def __init__(self, conn: mpc.Connection, loop: asyncio.AbstractEventLoop | None = None):
        self._conn = conn
        self._loop = loop or asyncio.get_event_loop()

        self._event = asyncio.Event()
        self._write_event = asyncio.Event()
        self._closed = False

        self._loop.add_reader(self._conn.fileno(), self._event.set)

    async def _wait_for_write(self):
        self._write_event.clear()
        self._loop.add_writer(self._conn.fileno(), self._write_event.set)
        try:
            await self._write_event.wait()
        finally:
            if not self._conn.closed:
                self._loop.remove_writer(self._conn.fileno())

    async def _wait_for_input(self):
        while not self._conn.poll():
            await self._event.wait()
            self._event.clear()

    async def send(self, obj):
        await self._wait_for_write()
        self._conn.send(obj)

    async def recv(self):
        await self._wait_for_input()
        return self._conn.recv()

    def fileno(self):
        return self._conn.fileno()

    def close(self):
        if self._conn.closed:
            return

        self._loop.remove_reader(self._conn.fileno())
        self._conn.close()

    async def poll(self, timeout: float = 0.0):
        if self._conn.poll():
            return True

        try:
            await asyncio.wait_for(self._wait_for_input(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return self._conn.poll()

    async def send_bytes(self, buf, offset=0, size=None):
        await self._wait_for_write()
        self._conn.send_bytes(buf, offset, size)

    async def recv_bytes(self, maxlength=None):
        await self._wait_for_input()
        return self._conn.recv_bytes(maxlength)

    async def recv_bytes_into(self, buf, offset=0):
        await self._wait_for_input()
        return self._conn.recv_bytes_into(buf, offset)
"""