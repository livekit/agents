"""Unit tests for the Gladia STT websocket receive loop."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from livekit.agents import APIConnectionError

pytestmark = pytest.mark.unit


async def test_an_error_frame_raises_so_the_stream_can_fail_over() -> None:
    """A heartbeat timeout arrives as an ERROR frame, not as a close frame.

    Gladia's only other ERROR branch is in the batch path. The streaming receive
    loop logged it as an unexpected message type and then ended normally, so `_run`
    broke out instead of reconnecting and the base stream never saw a failure --
    leaving a FallbackAdapter nothing to react to.
    """
    from livekit.plugins.gladia.stt import SpeechStream

    class _ErrorWebSocket:
        def __init__(self) -> None:
            self._sent = False

        def __aiter__(self) -> _ErrorWebSocket:
            return self

        async def __anext__(self) -> Any:
            if self._sent:
                raise StopAsyncIteration
            self._sent = True
            return SimpleNamespace(type=aiohttp.WSMsgType.ERROR, data=None, extra=None)

        def exception(self) -> Exception:
            return RuntimeError("no PONG received")

    stream = object.__new__(SpeechStream)
    stream._ws = _ErrorWebSocket()

    with pytest.raises(APIConnectionError):
        await stream._recv_messages_task()
