"""In-process fakes for testing the cloud transport without aiohttp.

Used by cloud-transport tests to drive ``_CloudTransport.run`` deterministically:

- ``FakeTurnDetectorWS`` impersonates an ``aiohttp.ClientWebSocketResponse``.
  Captures outbound ``send_bytes`` payloads as parsed ``ClientMessage``
  protobufs, and yields scripted server frames from ``receive()``.
- ``ControlledCloudTransport`` overrides ``_connect_ws`` so each connect
  attempt is scripted: an exception is raised, ``None`` returns the fake ws.
- ``make_stream(...)`` constructs an ``_BaseStreamingTurnDetectorStream`` in cloud
  mode bound to a controlled transport.
- ``wait_until_connected(...)`` blocks until the transport's ``_ws`` is set,
  so tests can assert against post-connect state without arbitrary sleeps.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import aiohttp

from livekit.agents.inference.eot.base import (
    TurnDetectorOptions,
    _BaseStreamingTurnDetectorStream,
)
from livekit.agents.inference.eot.languages import ThresholdOptions
from livekit.agents.inference.eot.transports import _CloudTransport, _CloudTransportOptions
from livekit.agents.types import APIConnectOptions
from livekit.protocol.agent_pb.agent_inference import ClientMessage


class FakeTurnDetectorWS:
    """Fake ``aiohttp.ClientWebSocketResponse`` for cloud-transport tests."""

    def __init__(self) -> None:
        self.sent: list[ClientMessage] = []
        self._incoming: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        self.closed = False
        self.close_code: int | None = None

    async def send_bytes(self, data: bytes) -> None:
        if self.closed:
            raise ConnectionResetError("ws closed")
        msg = ClientMessage()
        msg.ParseFromString(data)
        self.sent.append(msg)

    async def receive(self) -> aiohttp.WSMessage:
        return await self._incoming.get()

    async def close(self) -> None:
        self.closed = True

    def feed_unexpected_close(self, code: int = 1006) -> None:
        """Simulate a server-initiated drop. ``recv_task`` will raise."""
        self.close_code = code
        self._incoming.put_nowait(
            aiohttp.WSMessage(type=aiohttp.WSMsgType.CLOSE, data=code, extra=None)
        )
        self.closed = True


class ControlledCloudTransport(_CloudTransport):
    """``_CloudTransport`` subclass that scripts the ``_connect_ws`` outcome.

    ``connect_script`` is consumed left-to-right. An exception is raised; a
    ``None`` returns the fake ws. Once exhausted, subsequent calls return
    the fake ws.
    """

    def __init__(
        self,
        *,
        fake_ws: FakeTurnDetectorWS,
        connect_script: list[BaseException | None] | None = None,
        **kwargs: Any,
    ) -> None:
        self._fake_ws = fake_ws
        self._connect_script: list[BaseException | None] = list(connect_script or [])
        self._connect_calls = 0
        super().__init__(**kwargs)

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        self._connect_calls += 1
        if self._connect_script:
            r = self._connect_script.pop(0)
            if isinstance(r, BaseException):
                raise r
        return self._fake_ws  # type: ignore[return-value]


def make_stream(
    *,
    fake_ws: FakeTurnDetectorWS | None = None,
    connect_script: list[BaseException | None] | None = None,
    max_retry: int = 3,
    retry_interval: float = 0.0,
) -> tuple[_BaseStreamingTurnDetectorStream, FakeTurnDetectorWS, ControlledCloudTransport]:
    """Construct a cloud-mode stream with a controlled transport.

    Returns the stream, the fake ws, and the transport so callers can read
    post-connect state from either side.
    """
    fake_ws = fake_ws or FakeTurnDetectorWS()
    detector = MagicMock()
    detector.model = "turn-detector-v1"
    detector.provider = "livekit"
    session_mock = MagicMock()
    session_mock.closed = False
    opts = TurnDetectorOptions(sample_rate=16000, thresholds=ThresholdOptions("turn-detector-v1"))
    conn_options = APIConnectOptions(max_retry=max_retry, retry_interval=retry_interval)
    cloud_opts = _CloudTransportOptions(
        base_url="",
        api_key="x",
        api_secret="x",
        conn_options=conn_options,
    )
    transport = ControlledCloudTransport(
        fake_ws=fake_ws,
        connect_script=connect_script,
        detector=detector,
        opts=opts,
        cloud_opts=cloud_opts,
        http_session=session_mock,
    )
    stream = _BaseStreamingTurnDetectorStream(
        detector=detector,
        opts=opts,
        transport=transport,
        model="turn-detector-v1",
    )
    return stream, fake_ws, transport


async def wait_until_connected(
    transport: ControlledCloudTransport, *, timeout: float = 1.0
) -> None:
    """Block until ``transport._ws`` is set."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while transport._ws is None:
        if loop.time() > deadline:
            raise TimeoutError("transport did not connect within timeout")
        await asyncio.sleep(0)


async def drain_send_queue(transport: ControlledCloudTransport, *, timeout: float = 1.0) -> None:
    """Yield until the outbound channel is empty (sender task has drained it)."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        ch = transport._send_ch
        if ch is None or ch.qsize() == 0:
            # Give the sender task one more tick to flush whatever it was awaiting.
            await asyncio.sleep(0)
            return
        if loop.time() > deadline:
            raise TimeoutError("send queue did not drain within timeout")
        await asyncio.sleep(0)
