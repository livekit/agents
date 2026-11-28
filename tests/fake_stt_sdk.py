"""SDK transport adapters for the common STT lifecycle tests."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from livekit.agents import utils


class Server:
    def __init__(self):
        self.connected = asyncio.Event()
        self.stopped = asyncio.Event()
        self.messages = asyncio.Queue()
        self.audio = []
        self.fail_stop = False
        self.consumer = None
        self.final = None
        self.terminal = None
        self.closed = False
        self.stream_closed = False
        self.connections = []

    def track_connection(self):
        connection = SimpleNamespace(closed=False)
        self.connections.append(connection)
        return connection

    def stop(self):
        self.stopped.set()
        if self.fail_stop:
            raise ConnectionResetError("test stop write failed")

    async def send_audio(self, data):
        if data:
            self.audio.append(data)
        else:
            self.stop()

    async def end_audio(self):
        self.stop()

    async def events(self):
        while True:
            message = await self.messages.get()
            if message is None:
                return
            if isinstance(message, Exception):
                raise message
            yield message

    def finish(self):
        self.messages.put_nowait(self.final)
        if self.terminal is not None:
            self.messages.put_nowait(self.terminal)
        self.messages.put_nowait(None)

    def drop(self):
        self.messages.put_nowait(None)

    async def close(self):
        self.closed = True
        self.messages.put_nowait(None)
        if self.consumer is not None:
            await utils.aio.gracefully_cancel(self.consumer)

    def cancel(self):
        self.stream_closed = True
        if self.consumer is not None:
            self.consumer.cancel()


class Pool:
    last_acquire_time = 0.0
    last_connection_reused = False

    def __init__(self, server, client=None):
        self.server = server
        self.client = server if client is None else client
        self.resource = server.track_connection()

    @asynccontextmanager
    async def connection(self, *, timeout):
        self.server.connected.set()
        yield self.client

    async def aclose(self):
        self.resource.closed = True
        await self.server.close()

    def remove(self, connection):
        pass
