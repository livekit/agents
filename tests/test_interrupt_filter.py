import pytest
from agents.extensions.interrupt_handler import InterruptFilter

@pytest.mark.asyncio
async def test_ignore_filler_when_agent_speaking():
    f = InterruptFilter(ignored_words=["uh","umm","hmm"], conf_threshold=0.5)
    res = await f.on_asr_event("uh umm", confidence=0.9, agent_speaking=True)
    assert res["should_stop_agent"] is False
    assert res["reason"] == "filler_only"

@pytest.mark.asyncio
async def test_mixed_stops_agent_when_agent_speaking():
    f = InterruptFilter(ignored_words=["uh","umm"], conf_threshold=0.5)
    res = await f.on_asr_event("uh stop", confidence=0.8, agent_speaking=True)
    assert res["should_stop_agent"] is True

@pytest.mark.asyncio
async def test_low_confidence_ignored():
    f = InterruptFilter(ignored_words=["hmm"], conf_threshold=0.8)
    res = await f.on_asr_event("hmm", confidence=0.6, agent_speaking=True)
    assert res["should_stop_agent"] is False
    assert res["reason"].startswith("low_confidence")

@pytest.mark.asyncio
async def test_agent_quiet_registers_speech():
    f = InterruptFilter(ignored_words=["uh"], conf_threshold=0.5)
    res = await f.on_asr_event("uh", confidence=0.9, agent_speaking=False)
    assert res["should_stop_agent"] is False
    assert res["reason"] == "agent_quiet"
