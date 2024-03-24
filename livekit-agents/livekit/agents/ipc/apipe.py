import asyncio
import queue
import threading

from .. import aio
from . import protocol


# TODO(theomonnom): More efficient implementation without additional threads
class AsyncPipe:
    """Wraps a ProcessPipe to provide async I/O"""

    def __init__(
        self, pipe: protocol.ProcessPipe, loop: asyncio.AbstractEventLoop
    ) -> None:
        self._loop = loop
        self._p = pipe

        self._read_ch = aio.Chan(32, loop=self._loop)
        self._write_q = queue.Queue(32)

        self._exit_ev = threading.Event()
        self._read_t = threading.Thread(target=self._read_thread, daemon=True)
        self._write_t = threading.Thread(target=self._write_thread, daemon=True)
        self._read_t.start()
        self._write_t.start()

    def _read_thread(self) -> None:
        while not self._exit_ev.is_set():
            msg = protocol.read_msg(self._p)

            def _put_msg(msg):
                _ = asyncio.ensure_future(self._read_ch.send(msg))

            self._loop.call_soon_threadsafe(_put_msg, msg)

    def _write_thread(self) -> None:
        while not self._exit_ev.is_set():
            msg = self._write_q.get()
            protocol.write_msg(self._p, msg)

    async def read(self) -> protocol.Message:
        return await self._read_ch.recv()

    async def write(self, msg: protocol.Message) -> None:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("write must be called from the same loop as the pipe")

        await asyncio.to_thread(self._write_q.put, msg)

    def __aiter__(self):
        return self

    async def __anext__(self) -> protocol.Message:
        return await self.read()

    def close(self) -> None:
        self._read_ch.close()
        self._exit_ev.set()
