import pytest
from interrupt_handler.middleware import InterruptFilteringMiddleware


@pytest.mark.asyncio
async def test_filler_ignored():
    mw = InterruptFilteringMiddleware()
    result = await mw.should_interrupt(
        text="uh umm hmm",
        confidence=0.95,
        agent_is_speaking=True
    )
    assert result is False


@pytest.mark.asyncio
async def test_low_confidence_ignored():
    mw = InterruptFilteringMiddleware()
    result = await mw.should_interrupt(
        text="hello",
        confidence=0.2,
        agent_is_speaking=True
    )
    assert result is False


@pytest.mark.asyncio
async def test_command_interrupts():
    mw = InterruptFilteringMiddleware()
    result = await mw.should_interrupt(
        text="please stop",
        confidence=0.99,
        agent_is_speaking=True
    )
    assert result is True


@pytest.mark.asyncio
async def test_agent_silent_accepts_all():
    mw = InterruptFilteringMiddleware()
    result = await mw.should_interrupt(
        text="umm",
        confidence=0.1,
        agent_is_speaking=False
    )
    assert result is True
