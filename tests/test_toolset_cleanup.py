import asyncio

import pytest

from livekit.agents.llm import Toolset

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


class SlowClosingToolset(Toolset):
    def __init__(self, *, id: str) -> None:
        super().__init__(id=id)
        self.started = asyncio.Event()
        self.finished = asyncio.Event()

    async def aclose(self) -> None:
        self.started.set()
        for _ in range(10):
            await asyncio.sleep(0)
        self.finished.set()


class FailingToolset(Toolset):
    def __init__(self, peer: SlowClosingToolset, error: BaseException) -> None:
        super().__init__(id="failing")
        self.peer = peer
        self.error = error

    async def aclose(self) -> None:
        await self.peer.started.wait()
        raise self.error


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("failing_first", [False, True])
async def test_close_waits_for_sibling_cleanup_before_raising(
    error_type: type[BaseException], failing_first: bool
) -> None:
    slow = SlowClosingToolset(id="slow")
    error = error_type("close failed")
    failing = FailingToolset(slow, error)
    parent = Toolset(id="parent", tools=[failing, slow] if failing_first else [slow, failing])

    try:
        with pytest.raises(error_type) as raised:
            await parent.aclose()
        assert raised.value is error
        assert slow.finished.is_set()
    finally:
        # Let the baseline's outstanding close finish, even when the assertion fails.
        await asyncio.wait_for(slow.finished.wait(), timeout=1)


async def test_close_waits_for_all_successful_nested_toolsets() -> None:
    first = SlowClosingToolset(id="first")
    second = SlowClosingToolset(id="second")
    parent = Toolset(id="parent", tools=[first, Toolset(id="nested", tools=[second])])

    await parent.aclose()

    assert first.finished.is_set()
    assert second.finished.is_set()
