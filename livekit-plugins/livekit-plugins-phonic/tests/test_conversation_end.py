from collections.abc import AsyncIterator
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from livekit.agents import llm
from livekit.plugins.phonic.realtime import PhonicConversationEndedError, RealtimeModel

pytestmark = pytest.mark.unit


class _ClosedSocket:
    def __init__(self, code: int, reason: str) -> None:
        self._close = Close(code, reason)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        raise ConnectionClosedError(self._close, None)
        yield


@pytest.mark.parametrize("code", [4500, 4505, 4600])
async def test_deliberate_end_emits_one_typed_error_without_raising(code: int) -> None:
    session = RealtimeModel(api_key="test").session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    try:
        await session._recv_task(_ClosedSocket(code, "ended"))  # type: ignore[arg-type]

        assert len(errors) == 1
        assert not errors[0].recoverable
        assert isinstance(errors[0].error, PhonicConversationEndedError)
        assert errors[0].error.close_code == code
    finally:
        await session.aclose()


async def test_other_close_codes_still_raise() -> None:
    session = RealtimeModel(api_key="test").session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    try:
        with pytest.raises(ConnectionClosedError):
            await session._recv_task(_ClosedSocket(4501, "downstream error"))  # type: ignore[arg-type]

        assert not isinstance(errors[0].error, PhonicConversationEndedError)
    finally:
        await session.aclose()
