import pytest

from livekit.agents.inference.tts import TTS
from livekit.agents.utils import ConnectionPool

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]


async def test_release_closes_idle_connections_and_reconnects_on_next_request() -> None:
    tts = TTS(model="cartesia/sonic-3", api_key="k", api_secret="s")
    connects = 0
    closed: list[int] = []

    async def connect(timeout: float) -> int:
        nonlocal connects
        connects += 1
        return connects

    async def close(conn: int) -> None:
        closed.append(conn)

    pool = ConnectionPool[int](connect_cb=connect, close_cb=close)
    tts._pool = pool

    pool.put(await pool.get(timeout=1.0))
    await tts.release()

    assert closed == [1]
    assert await pool.get(timeout=1.0) == 2
