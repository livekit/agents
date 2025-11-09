import pytest
from agents.extensions.interrupt_handler import InterruptFilter

@pytest.mark.asyncio
async def test_ignore_filler_when_agent_speaks():
    handler = InterruptFilter(None, ignored_words=["uh", "umm", "hmm"])
    await handler.set_agent_speaking(True)
    result = handler.is_filler_only("uh")
    assert result is True

@pytest.mark.asyncio
async def test_detect_valid_interrupt():
    handler = InterruptFilter(None, ignored_words=["uh", "umm", "hmm"])
    await handler.set_agent_speaking(True)
    result = handler.is_filler_only("stop")
    assert result is False
import pytest
from agents.extensions.interrupt_handler import InterruptFilter

@pytest.mark.asyncio
async def test_ignore_filler_when_agent_speaks():
    # Define a dummy callback to simulate stopping the agent
    async def dummy_stop():
        return True

    handler = InterruptFilter(None, stop_callback=dummy_stop, ignored_words=["uh", "umm", "hmm"])
    await handler.set_agent_speaking(True)
    result = handler.is_filler_only("uh")
    assert result is True

@pytest.mark.asyncio
async def test_detect_valid_interrupt():
    async def dummy_stop():
        return True

    handler = InterruptFilter(None, stop_callback=dummy_stop, ignored_words=["uh", "umm", "hmm"])
    await handler.set_agent_speaking(True)
    result = handler.is_filler_only("stop")
    assert result is False
