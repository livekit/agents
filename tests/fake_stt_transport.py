"""In-memory aiohttp transport for STT stream contract tests."""

import asyncio
import json
from types import SimpleNamespace

import aiohttp


class _Context:
    def __await__(self):
        async def connect():
            return self

        return connect().__await__()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


class Socket(_Context):
    def __init__(self, transport, kwargs):
        self.transport = transport
        self.kwargs = kwargs
        self.closed = False
        self.close_code = 1000
        self._response = SimpleNamespace(headers={}, status=101)
        self.messages = asyncio.Queue()
        self.writes = []
        self.stops = asyncio.Queue()
        self.pings = 0
        self.receives = 0
        self.errors = 0
        for greeting in transport.case.greetings:
            self.emit(greeting)

    def emit(self, data):
        if isinstance(data, aiohttp.WSMessage):
            self.messages.put_nowait(data)
        elif isinstance(data, bytes):
            self.messages.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, data, ""))
        else:
            self.messages.put_nowait(
                aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(data), "")
            )

    def server_close(self):
        self.emit(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, 1000, ""))

    async def receive(self):
        self.receives += 1
        if self.messages.empty() and self.transport.failure in ("close", "error"):
            # Deliver setup frames first, then fail an active connection. Yield so
            # the sender can consume audio and the test can exercise no-replay policies.
            await asyncio.sleep(0.02)
            self.closed = True
            if self.transport.failure == "close":
                return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, 1000, "")
            self.errors += 1
            return aiohttp.WSMessage(aiohttp.WSMsgType.ERROR, self.exception(), "")
        message = await self.messages.get()
        if message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
            self.closed = True
        return message

    def is_stop(self, data):
        expected = self.transport.case.stop
        if expected is None:
            return False
        if isinstance(expected, dict):
            if not isinstance(data, str):
                return False
            try:
                actual = json.loads(data)
            except json.JSONDecodeError:
                return False
            return isinstance(actual, dict) and all(actual.get(k) == v for k, v in expected.items())
        return data == expected

    def record_write(self, data):
        if self.is_stop(data):
            if self.transport.fail_stop:
                if self.transport.fail_stop == "close":
                    self.server_close()
                raise self.transport.error
            self.stops.put_nowait(data)
        self.writes.append(data)

    async def _before_write(self):
        if self.transport.block_writes:
            self.transport.write_blocked.set()
            await self.transport.release_writes.wait()

    async def send_str(self, data):
        if self.closed:
            raise aiohttp.ClientConnectionError("test write failed")
        await self._before_write()
        self.record_write(data)
        if self.transport.auto_reply and self.transport.case.reply is not None:
            self.transport.case.reply(self, data)

    async def send_json(self, data):
        await self.send_str(json.dumps(data))

    async def send_bytes(self, data):
        if self.closed:
            raise aiohttp.ClientConnectionError("test write failed")
        await self._before_write()
        self.record_write(bytes(data))

    async def ping(self, data=b""):
        self.pings += 1
        if self.transport.failure != "silent":
            self.emit(aiohttp.WSMessage(aiohttp.WSMsgType.PONG, data, ""))

    async def pong(self, data=b""):
        pass

    def exception(self):
        return self.transport.error

    async def close(self, **kwargs):
        self.closed = True


class Response(_Context):
    status = 200
    ok = True
    headers = {}

    async def json(self, **kwargs):
        return {"url": "wss://unused", "id": "test", "access_token": "test", "expire_at": 1e12}

    async def text(self):
        return json.dumps(await self.json())

    def raise_for_status(self):
        pass

    async def close(self):
        pass


class Transport:
    def __init__(self, case):
        self.case = case
        self.failure = None
        self.auto_reply = False
        self.fail_stop = None
        self.error = aiohttp.ServerTimeoutError("test connection lost")
        self.sockets = []
        self.connected = asyncio.Event()
        self.progress = asyncio.Event()
        self.block_writes = False
        self.write_blocked = asyncio.Event()
        self.release_writes = asyncio.Event()

    def session(self, *args, **kwargs):
        transport = self

        class Session:
            closed = False

            def ws_connect(self, *args, **kwargs):
                socket = Socket(transport, kwargs)
                transport.sockets.append(socket)
                transport.connected.set()
                return socket

            def post(self, *args, **kwargs):
                return Response()

            async def close(self):
                self.closed = True

        return Session()
