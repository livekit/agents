import asyncio
import queue
import threading

from . import aio, ipc_enc


# TODO(theomonnom): More efficient implementation without additional threads
class AsyncPipe:
    """Wraps a ProcessPipe to provide async I/O"""

    def __init__(
        self,
        pipe: ipc_enc.ProcessPipe,
        loop: asyncio.AbstractEventLoop,
        messages: dict[int, type[ipc_enc.Message]],
    ) -> None:
        self._loop = loop
        self._p = pipe
        self._messages = messages

        self._read_ch = aio.Chan(32, loop=self._loop)
        self._write_q = queue.Queue(32)

        self._exit_ev = threading.Event()
        self._read_t = threading.Thread(target=self._read_thread, daemon=True)
        self._write_t = threading.Thread(target=self._write_thread, daemon=True)
        self._read_t.start()
        self._write_t.start()

    def _read_thread(self) -> None:
        while not self._exit_ev.is_set():
            try:
                msg = ipc_enc.read_msg(self._p, self._messages)

                def _put_msg(msg):
                    _ = asyncio.ensure_future(self._read_ch.send(msg))

                self._loop.call_soon_threadsafe(_put_msg, msg)
            except (OSError, EOFError, BrokenPipeError):
                break

        self._loop.call_soon_threadsafe(self.close)

    def _write_thread(self) -> None:
        while not self._exit_ev.is_set():
            try:
                msg = self._write_q.get()
                ipc_enc.write_msg(self._p, msg)
            except (OSError, BrokenPipeError):
                break

        self._loop.call_soon_threadsafe(self.close)

    async def read(self) -> ipc_enc.Message:
        return await self._read_ch.recv()

    async def write(self, msg: ipc_enc.Message) -> None:
        await asyncio.to_thread(self._write_q.put, msg)

    def __aiter__(self) -> "AsyncPipe":
        return self

    async def __anext__(self) -> ipc_enc.Message:
        return await self.read()

    def close(self) -> None:
        self._p.close()
        self._read_ch.close()
        self._exit_ev.set()
